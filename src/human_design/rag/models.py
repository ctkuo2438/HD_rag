"""Typed result models for the local Human Design RAG pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class PdfLoadResult:
    pdf_count: int # PDF files number
    document_count: int # documents number after splitting PDF files
    source_files: tuple[Path, ...]


@dataclass(frozen=True)
class LowTextDocument:
    source_path: str | None
    file_name: str | None
    page_label: str | None
    page_number: int | None
    text_length: int


@dataclass(frozen=True)
class TextExtractionReport:
    pdf_count: int
    document_count: int
    source_files: tuple[Path, ...]
    total_text_characters: int
    low_text_threshold: int
    low_text_document_count: int
    low_text_documents: tuple[LowTextDocument, ...]


@dataclass(frozen=True)
class ChunkingResult:
    document_count: int
    chunk_count: int
    chunk_size: int # 800 by default
    chunk_overlap: int # 80 by default


@dataclass(frozen=True)
class IngestionResult:
    pdf_count: int
    document_count: int
    chunk_count: int
    persisted_count: int
    collection_name: str
    chroma_dir: Path


@dataclass(frozen=True)
class VectorStoreResult:
    collection_name: str
    chroma_dir: Path
    persisted_count: int


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    score: float | None
    source_file: str | None
    page_label: str | None
    page_number: int | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class HybridSearchRequest:
    """Separate semantic and exact-term queries for one focused question."""

    original_query: str
    dense_query: str
    sparse_query: str

    def __post_init__(self) -> None:
        for name in ("original_query", "dense_query", "sparse_query"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")


@dataclass(frozen=True)
class RetrievedChunk:
    """Internal provenance; absent ranks/scores stay None, never sentinel zeroes.

    Metadata may retain local diagnostics. Prompt boundaries must reject absolute
    paths, and public citations must select only safe display fields.
    """

    chunk_id: str
    text: str
    source_file: str
    source_relpath: str | None = None
    document_title: str | None = None
    page_label: str | None = None
    page_number: int | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("chunk_id", "text", "source_file"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        for name in ("source_relpath", "document_title", "page_label"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        _validate_relative_source(self.source_file, "source_file", basename=True)
        if self.source_relpath is not None:
            _validate_relative_source(self.source_relpath, "source_relpath")
        if self.page_number is not None and type(self.page_number) is not int:
            raise TypeError("page_number must be an integer or None")
        for name in ("dense_rank", "sparse_rank"):
            rank = getattr(self, name)
            if rank is not None:
                if type(rank) is not int:
                    raise TypeError(f"{name} must be an integer or None")
                if rank < 1:
                    raise ValueError(f"{name} must be at least 1")
        for name in ("dense_score", "sparse_score", "rrf_score", "rerank_score"):
            score = getattr(self, name)
            if score is not None:
                if type(score) not in (int, float):
                    raise TypeError(f"{name} must be a finite number or None")
                if not math.isfinite(score):
                    raise ValueError(f"{name} must be finite")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a JSON-safe mapping")
        object.__setattr__(self, "metadata", _copy_json_value(dict(self.metadata)))


@dataclass(frozen=True)
class HybridRetrievalResult:
    """Ordered fused candidates and the configuration/index identity that produced them."""

    request: HybridSearchRequest
    ingestion_version: str
    ingestion_id: str
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    final_top_k: int
    rrf_k: int
    candidates: tuple[RetrievedChunk, ...] = field(default_factory=tuple)
    rerank_provider: str | None = None
    rerank_model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, HybridSearchRequest):
            raise TypeError("request must be HybridSearchRequest")
        candidates = tuple(self.candidates)
        if any(not isinstance(chunk, RetrievedChunk) for chunk in candidates):
            raise TypeError("candidates must contain RetrievedChunk objects")
        if any(chunk.rrf_score is None for chunk in candidates):
            raise ValueError("fused candidates must have an rrf_score")
        object.__setattr__(self, "candidates", candidates)
        for name in ("ingestion_version", "ingestion_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("dense_top_k", "sparse_top_k", "fusion_top_k", "final_top_k", "rrf_k"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.final_top_k > self.fusion_top_k:
            raise ValueError("final_top_k must be less than or equal to fusion_top_k")
        for name in ("rerank_provider", "rerank_model"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")


def _validate_relative_source(value: str, name: str, *, basename: bool = False) -> None:
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if (
        not value.strip() or value.lower().startswith("file:")
        or posix.is_absolute() or windows.anchor or ".." in windows.parts
        or (basename and ("/" in value or "\\" in value or value == "."))
    ):
        raise ValueError(f"{name} must be a safe {'basename' if basename else 'relative path'}")


def _copy_json_value(value: object) -> JsonValue:
    """Validate and detach JSON metadata without lossy string/default conversions."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("metadata keys must be strings")
        return {key: _copy_json_value(item) for key, item in value.items()}
    raise TypeError("metadata must contain only JSON-safe values")
