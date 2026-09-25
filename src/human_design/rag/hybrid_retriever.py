"""Separate-query retrieval and deterministic reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from human_design.rag.config import AppConfig
from human_design.rag.hybrid_index import HybridIndexError, safe_source_metadata
from human_design.rag.models import HybridRetrievalResult, HybridSearchRequest, RetrievedChunk


class Retriever(Protocol):
    ingestion_id: str

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


_SOURCE_FIELDS = ("source_file", "source_relpath", "document_title", "page_label", "page_number")


def _safe_candidate(chunk: RetrievedChunk, backend: str) -> RetrievedChunk:
    if getattr(chunk, f"{backend}_rank") is None:
        raise HybridIndexError(f"Index consistency error: missing {backend} rank")
    metadata = safe_source_metadata(chunk.metadata)
    public = safe_source_metadata({name: getattr(chunk, name) for name in _SOURCE_FIELDS})
    for key, value in {"chunk_id": chunk.chunk_id, **public}.items():
        if key in metadata and metadata[key] != value:
            raise HybridIndexError("Index consistency error: conflicting source metadata")
        metadata[key] = value
    other = "sparse" if backend == "dense" else "dense"
    return replace(
        chunk, **{name: public.get(name) for name in _SOURCE_FIELDS}, metadata=metadata,
        **{f"{other}_rank": None, f"{other}_score": None}, rrf_score=None, rerank_score=None,
    )


def _merge(existing: RetrievedChunk, incoming: RetrievedChunk) -> RetrievedChunk:
    if existing.text != incoming.text:
        raise HybridIndexError("Index consistency error: conflicting canonical chunk text")
    metadata = dict(existing.metadata)
    for key, value in incoming.metadata.items():
        if key in metadata and metadata[key] != value:
            raise HybridIndexError("Index consistency error: conflicting source metadata")
        metadata[key] = value
    ranks = {}
    for backend in ("dense", "sparse"):
        rank = getattr(incoming, f"{backend}_rank")
        previous = getattr(existing, f"{backend}_rank")
        if rank is not None and (previous is None or rank < previous):
            ranks[f"{backend}_rank"] = rank
            ranks[f"{backend}_score"] = getattr(incoming, f"{backend}_score")
    return replace(existing, **ranks, **{name: metadata.get(name) for name in _SOURCE_FIELDS}, metadata=metadata)


def fuse_rrf(
    dense: Sequence[RetrievedChunk], sparse: Sequence[RetrievedChunk], *, rrf_k: int, fusion_top_k: int,
) -> tuple[RetrievedChunk, ...]:
    """Deduplicate by canonical ID; scores remain diagnostics and never enter RRF."""
    for value in (rrf_k, fusion_top_k):
        if type(value) is not int or value <= 0:
            raise ValueError("RRF K and fusion Top-K must be positive integers")
    merged: dict[str, RetrievedChunk] = {}
    for backend, chunks in (("dense", dense), ("sparse", sparse)):
        for chunk in chunks:
            candidate = _safe_candidate(chunk, backend)
            existing = merged.get(candidate.chunk_id)
            merged[candidate.chunk_id] = _merge(existing, candidate) if existing else candidate
    fused = [replace(chunk, rrf_score=sum(
        1 / (rrf_k + rank) for rank in (chunk.dense_rank, chunk.sparse_rank) if rank is not None
    )) for chunk in merged.values()]
    fused.sort(key=lambda chunk: (
        -chunk.rrf_score,
        min(rank for rank in (chunk.dense_rank, chunk.sparse_rank) if rank is not None),
        chunk.chunk_id,
    ))
    return tuple(fused[:fusion_top_k])


@dataclass(frozen=True)
class HybridRetriever:
    dense_adapter: Retriever = field(repr=False)
    sparse_adapter: Retriever = field(repr=False)
    config: AppConfig

    def retrieve(self, request: HybridSearchRequest) -> HybridRetrievalResult:
        identity = getattr(self.dense_adapter, "ingestion_id", None)
        if (not isinstance(identity, str) or not identity.strip()
                or identity != getattr(self.sparse_adapter, "ingestion_id", None)):
            raise HybridIndexError("Dense/sparse ingestion identity is missing or mismatched")
        settings = self.config
        result = HybridRetrievalResult(
            request=request, ingestion_version=settings.ingestion_version, ingestion_id=identity,
            dense_top_k=settings.dense_top_k, sparse_top_k=settings.sparse_top_k,
            fusion_top_k=settings.fusion_top_k, final_top_k=settings.final_top_k, rrf_k=settings.rrf_k,
        )
        try:
            dense = self.dense_adapter.retrieve(request.dense_query, settings.dense_top_k)
        except Exception:
            raise HybridIndexError("Phase 3 dense retrieval failed; check the index") from None
        try:
            sparse = self.sparse_adapter.retrieve(request.sparse_query, settings.sparse_top_k)
        except Exception:
            raise HybridIndexError("Phase 3 sparse retrieval failed; check the BM25 index") from None
        return replace(result, candidates=fuse_rrf(
            dense, sparse, rrf_k=settings.rrf_k, fusion_top_k=settings.fusion_top_k,
        ))
