"""Frozen request, chart, prompt, and public output models for focused Q&A."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TypeVar

from human_design.rag.models import JsonScalar, RetrievedChunk
from human_design.vision.models import BodyGraphExtractionResult


_T = TypeVar("_T")
_ABSOLUTE_PATH_IN_TEXT = re.compile(
    r"\bfile://|(?<![\w:/\\])(?:[A-Za-z]:[\\/]|[\\/])[^\s<>\"']+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KnowledgeQuestionRequest:
    """A focused knowledge question without chart context."""

    query: str

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")


@dataclass(frozen=True)
class ChartImageQuestionRequest:
    """Outer entry: run Phase 2 before constructing a ChartQuestionRequest."""

    query: str
    bodygraph_image: Path

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.bodygraph_image, Path):
            raise TypeError("bodygraph_image must be a Path")


@dataclass(frozen=True)
class ChartQuestionRequest:
    """Chart Q&A core entry using a parsed and validated Phase 2 result."""

    query: str
    bodygraph_result: BodyGraphExtractionResult

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.bodygraph_result, BodyGraphExtractionResult):
            raise TypeError("bodygraph_result must be BodyGraphExtractionResult")


@dataclass(frozen=True)
class ChartFact:
    """One scalar chart fact; source_path is a Phase 2 field path, never a file.

    List members use stable selectors, e.g. active_gates[gate=42], so each path
    continues to identify the same fact after selecting or ordering facts.
    """

    field: str
    value: JsonScalar
    source_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.field, str):
            raise TypeError("field must be a string")
        if self.value is not None and type(self.value) not in (str, int, float, bool):
            raise TypeError("value must be a JSON scalar")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("value must be finite")
        if not isinstance(self.source_path, str) or not re.fullmatch(
            r"(?:derived_chart_data|raw_vision)\.[a-z_]+(?:\.[a-z_]+)*"
            r"(?:\[[a-z_]+=[A-Za-z0-9 _.-]+\])?",
            self.source_path,
        ):
            raise ValueError("source_path must be a non-empty Phase 2 model-field path")
        _reject_absolute_paths(asdict(self))


@dataclass(frozen=True)
class ChartContext:
    """Normalized chart facts and safe validation warnings only."""

    facts: tuple[ChartFact, ...] = field(default_factory=tuple)
    validation_warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        facts = _typed_tuple(self.facts, ChartFact, "facts")
        if len({fact.source_path for fact in facts}) != len(facts):
            raise ValueError("facts must have unique source_path values")
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "validation_warnings", _typed_tuple(
            self.validation_warnings, str, "validation_warnings",
        ))
        _reject_absolute_paths(asdict(self))


@dataclass(frozen=True)
class PromptSource:
    """One final source; passage text is data, while provenance stays path-checked."""

    citation_id: str
    chunk: RetrievedChunk

    def __post_init__(self) -> None:
        if not isinstance(self.citation_id, str):
            raise TypeError("citation_id must be a string")
        if not isinstance(self.chunk, RetrievedChunk):
            raise TypeError("chunk must be RetrievedChunk")
        provenance = asdict(self)
        del provenance["chunk"]["text"]
        _reject_absolute_paths(provenance)


@dataclass(frozen=True)
class PromptContext:
    """Question, selected facts, and final sources; no image or provider state."""

    user_question: str
    chart_facts: tuple[ChartFact, ...] = field(default_factory=tuple)
    sources: tuple[PromptSource, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.user_question, str):
            raise TypeError("user_question must be a string")
        object.__setattr__(self, "chart_facts", _typed_tuple(
            self.chart_facts, ChartFact, "chart_facts",
        ))
        object.__setattr__(self, "sources", _typed_tuple(self.sources, PromptSource, "sources"))
        # Question/passage text is untrusted data, escaped by the renderer. The
        # typed ChartFact/PromptSource boundaries validate structured provenance.


@dataclass(frozen=True)
class SourceCitation:
    """Public display provenance only: no passage, relative path, metadata, or scores."""

    citation_id: str
    chunk_id: str
    source_file: str
    document_title: str | None = None
    page_label: str | None = None
    page_number: int | None = None

    def __post_init__(self) -> None:
        for name in ("citation_id", "chunk_id", "source_file"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        for name in ("document_title", "page_label"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if self.page_number is not None and type(self.page_number) is not int:
            raise TypeError("page_number must be an integer or None")
        if (not self.source_file.strip() or self.source_file in {".", ".."}
                or "/" in self.source_file or "\\" in self.source_file):
            raise ValueError("source_file must be a safe basename")
        _reject_absolute_paths(asdict(self))


class AnswerStatus(StrEnum):
    """Focused Q&A outcomes; REFUSED is reserved for an explicit pipeline refusal."""

    OK = "ok"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_CHART = "invalid_chart"
    NEEDS_FOCUS = "needs_focus"
    REFUSED = "refused"
    ERROR = "error"


@dataclass(frozen=True)
class AnswerResult:
    """Public answer with cited sources and used facts, without retrieval internals."""

    status: AnswerStatus
    answer_markdown: str
    citations: tuple[SourceCitation, ...] = field(default_factory=tuple)
    chart_facts_used: tuple[ChartFact, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AnswerStatus(self.status))
        if not isinstance(self.answer_markdown, str):
            raise TypeError("answer_markdown must be a string")
        object.__setattr__(self, "citations", _typed_tuple(
            self.citations, SourceCitation, "citations",
        ))
        object.__setattr__(self, "chart_facts_used", _typed_tuple(
            self.chart_facts_used, ChartFact, "chart_facts_used",
        ))
        object.__setattr__(self, "warnings", _typed_tuple(self.warnings, str, "warnings"))
        _reject_absolute_paths(asdict(self))


def _typed_tuple(values: Iterable[_T], item_type: type[_T], name: str) -> tuple[_T, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of {item_type.__name__}")
    result = tuple(values)
    if any(not isinstance(item, item_type) for item in result):
        raise TypeError(f"{name} must contain only {item_type.__name__} values")
    return result


def _reject_absolute_paths(value: object, name: str = "context") -> None:
    """Reject filesystem provenance at prompt/public model boundaries on any OS."""
    if isinstance(value, str):
        normalized = value.strip()
        if (PurePosixPath(normalized).is_absolute() or PureWindowsPath(normalized).anchor
                or normalized.lower().startswith("file:")
                or _ABSOLUTE_PATH_IN_TEXT.search(normalized)):
            raise ValueError(f"{name} must not contain an absolute filesystem path")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_absolute_paths(key, "metadata key")
            _reject_absolute_paths(item, str(key))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_absolute_paths(item, name)
