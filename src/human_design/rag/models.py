"""Typed result models for the local Human Design RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PdfLoadResult:
    pdf_count: int # PDF files number
    document_count: int # documents number after splitting PDF files
    source_files: tuple[Path, ...]


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
