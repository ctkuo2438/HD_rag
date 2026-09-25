"""One structured Responses request with locally validated evidence references."""

from __future__ import annotations

import json
import re
from typing import Protocol, TypedDict, cast

from human_design.rag.config import AppConfig
from human_design.reading.models import (
    AnswerResult,
    AnswerStatus,
    PromptContext,
    _reject_absolute_paths,
)
from human_design.reading.prompt import build_source_citations, render_reading_prompt
from human_design.reading.query_builder import validate_query


class GenerationError(ValueError):
    """A safe configuration or provider error without private request/response data."""


class GenerationValidationError(GenerationError):
    """The sole response failed local validation; no repair request is made."""


class _Response(Protocol):
    status: str
    output_text: str


class _Responses(Protocol):
    def create(self, **kwargs: object) -> _Response: ...


class _Client(Protocol):
    responses: _Responses

    def close(self) -> None: ...


class _AnswerPayload(TypedDict):
    answer_markdown: str
    used_source_ids: list[str]
    used_chart_fact_paths: list[str]
    limitations: list[str]


_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_markdown": {"type": "string"},
        "used_source_ids": {"type": "array", "items": {"type": "string"}},
        "used_chart_fact_paths": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer_markdown", "used_source_ids", "used_chart_fact_paths", "limitations"],
    "additionalProperties": False,
}
_CITATION = re.compile(r"(?<!\\)\[(S[^\[\]\s]*)\]")


def _validate_config(config: AppConfig, client: _Client | None) -> None:
    # This marker is an internal Python test hook, never a config/request option.
    offline_fake = client is not None and getattr(client, "_test_only", False) is True
    if not offline_fake:
        if config.real_generation is not True:
            raise GenerationError("Enable HD_RAG_REAL_GENERATION=1 for real generation")
        if not isinstance(config.openai_api_key, str) or not config.openai_api_key.strip():
            raise GenerationError("Real generation requires OPENAI_API_KEY")
    if not isinstance(config.generation_model, str) or not config.generation_model.strip():
        raise GenerationError("Generation requires HD_RAG_GENERATION_MODEL")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate structured field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError("Invalid JSON constant")


def _parse_response(response: _Response) -> _AnswerPayload:
    try:
        if response.status != "completed" or not isinstance(response.output_text, str):
            raise ValueError("Incomplete response")
        payload = json.loads(response.output_text, object_pairs_hook=_unique_object,
                             parse_constant=_invalid_constant)
        if not isinstance(payload, dict) or set(payload) != set(_SCHEMA["required"]):
            raise ValueError("Invalid fields")
        if not isinstance(payload["answer_markdown"], str) or not payload["answer_markdown"].strip():
            raise ValueError("Blank answer")
        for name in ("used_source_ids", "used_chart_fact_paths", "limitations"):
            if not isinstance(payload[name], list) or any(not isinstance(v, str) for v in payload[name]):
                raise ValueError("Invalid field type")
    except Exception:
        # JSON/SDK errors can retain the complete private response in their text.
        raise GenerationValidationError("Generation returned invalid structured output") from None
    return cast(_AnswerPayload, payload)


def _private_echo(value: str, context: PromptContext, config: AppConfig, rendered: str) -> bool:
    private_values = [config.openai_api_key, config.cohere_api_key, rendered]
    private_values.extend(source.chunk.text for source in context.sources)
    return any(private and private in value for private in private_values)


def _validated_answer(
    payload: _AnswerPayload, context: PromptContext, config: AppConfig, rendered: str,
) -> AnswerResult:
    supplied_ids = {source.citation_id for source in context.sources}
    used_ids = set(payload["used_source_ids"])
    if not used_ids <= supplied_ids:
        raise GenerationValidationError("Generation referenced an unknown source ID")
    used_paths = set(payload["used_chart_fact_paths"])
    if not used_paths <= {fact.source_path for fact in context.chart_facts}:
        raise GenerationValidationError("Generation referenced an unsupplied chart fact")
    answer = payload["answer_markdown"]
    cited_ids = set(_CITATION.findall(answer))
    if not cited_ids or not cited_ids <= supplied_ids:
        raise GenerationValidationError("Generation requires valid supplied bracket citations")
    if not cited_ids <= used_ids:
        raise GenerationValidationError("Every bracket citation must be listed in used_source_ids")
    if _private_echo(answer, context, config, rendered):
        raise GenerationValidationError("Generation returned private evidence or credentials")

    warnings = []
    for limitation in payload["limitations"]:
        try:
            _reject_absolute_paths(limitation)
            if _private_echo(limitation, context, config, rendered):
                raise ValueError("Private limitation")
        except ValueError:
            limitation = "A private provider limitation was omitted."
        if limitation.strip() and limitation not in warnings:
            warnings.append(limitation)
    if used_ids - cited_ids:
        warnings.append("Unused source IDs were omitted from public citations.")
    try:
        return AnswerResult(
            status=AnswerStatus.OK, answer_markdown=answer,
            citations=tuple(c for c in build_source_citations(context) if c.citation_id in cited_ids),
            chart_facts_used=tuple(f for f in context.chart_facts if f.source_path in used_paths),
            warnings=tuple(warnings),
        )
    except (TypeError, ValueError):
        raise GenerationValidationError("Generation returned unsafe public output") from None


def generate_answer(
    prompt_context: PromptContext, config: AppConfig, *, client: _Client | None = None,
) -> AnswerResult:
    """Generate once, or return insufficient evidence before rendering/provider work.

    Injected clients are caller-owned. Only Python test doubles explicitly marked
    ``_test_only = True`` can run offline; no user request/config enables that hook.
    Real SDK retries are disabled so validation/failure never causes another call.
    """
    validate_query(prompt_context.user_question)
    if not prompt_context.sources:
        return AnswerResult(AnswerStatus.INSUFFICIENT_EVIDENCE,
                            "There is insufficient retrieved evidence to answer this question.")
    _validate_config(config, client)
    rendered = render_reading_prompt(prompt_context)
    owned = client is None
    if owned:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=config.openai_api_key, max_retries=0)
        except Exception:
            raise GenerationError("Unable to initialize the generation provider") from None
    assert client is not None
    try:
        response = client.responses.create(
            model=config.generation_model, input=rendered, store=False,
            text={"format": {"type": "json_schema", "name": "human_design_answer",
                             "strict": True, "schema": _SCHEMA}},
        )
    except Exception:
        raise GenerationError("Generation provider request failed") from None
    finally:
        if owned:
            try:
                client.close()
            except Exception:
                # Cleanup failures must not disclose provider state or cause retries.
                pass
    return _validated_answer(_parse_response(response), prompt_context, config, rendered)
