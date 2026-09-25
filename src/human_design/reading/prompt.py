"""Assemble final evidence and render escaped data using a packaged grounded prompt."""

from collections.abc import Sequence
from dataclasses import replace
from html import escape
from importlib.resources import files

from human_design.rag.models import RetrievedChunk
from human_design.reading.models import (
    ChartFact,
    PromptContext,
    PromptSource,
    SourceCitation,
    _reject_absolute_paths,
)
from human_design.reading.query_builder import validate_query


_PROMPT_RESOURCE = "human_design_answer.txt"


def load_reading_prompt() -> str:
    resource = files("human_design.reading") / "prompts" / _PROMPT_RESOURCE
    if not resource.is_file():
        raise FileNotFoundError(f"Reading answer prompt not found: {_PROMPT_RESOURCE}")
    return resource.read_text(encoding="utf-8")


def _safe_chunk(chunk: RetrievedChunk) -> RetrievedChunk:
    public = {}
    for name in ("document_title", "page_label"):
        value = getattr(chunk, name)
        try:
            _reject_absolute_paths(value, name)
        except ValueError:
            value = None
        public[name] = value
    # Keep the canonical chunk/provenance, but no arbitrary local/provider metadata.
    return replace(chunk, **public, metadata={})


def _require_final_sources(context: PromptContext) -> None:
    if not context.sources:
        raise ValueError("A generation-bound prompt requires at least one final source")
    if [s.citation_id for s in context.sources] != [f"S{i}" for i in range(1, len(context.sources) + 1)]:
        raise ValueError("Prompt citation IDs must follow final source order: S1, S2, ...")


def build_prompt_context(
    user_question: str, selected_facts: Sequence[ChartFact], final_chunks: Sequence[RetrievedChunk],
) -> PromptContext:
    """Accept only Task 36 selected facts and sources already selected after fusion/reranking."""
    validate_query(user_question)
    context = PromptContext(user_question, tuple(selected_facts), tuple(
        PromptSource(f"S{index}", _safe_chunk(chunk)) for index, chunk in enumerate(final_chunks, 1)
    ))
    _require_final_sources(context)
    return context


def build_source_citations(context: PromptContext) -> tuple[SourceCitation, ...]:
    """Project only public display fields, never paths, passages, diagnostics, or scores."""
    _require_final_sources(context)
    return tuple(SourceCitation(
        source.citation_id, source.chunk.chunk_id, source.chunk.source_file,
        source.chunk.document_title, source.chunk.page_label, source.chunk.page_number,
    ) for source in context.sources)


def render_reading_prompt(context: PromptContext) -> str:
    validate_query(context.user_question)
    _require_final_sources(context)
    chart_facts = "\n".join(
        f'  <fact field="{escape(f.field, quote=True)}" source_path="{escape(f.source_path, quote=True)}">'
        f'{escape(str(f.value), quote=True)}</fact>' for f in context.chart_facts
    )
    sources = []
    for source in context.sources:
        chunk = source.chunk
        attributes = {"id": source.citation_id, "file": chunk.source_file,
                      "title": chunk.document_title, "page": chunk.page_label, "page_number": chunk.page_number}
        rendered_attributes = " ".join(
            f'{key}="{escape(str(value), quote=True)}"' for key, value in attributes.items() if value is not None
        )
        sources.append(f"  <source {rendered_attributes}>{escape(chunk.text, quote=True)}</source>")
    return load_reading_prompt().format(
        user_question=escape(context.user_question, quote=True),
        chart_facts=chart_facts, reference_sources="\n".join(sources),
    )
