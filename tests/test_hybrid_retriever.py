"""Hybrid fusion contracts, using only in-memory retrieval fakes."""

from dataclasses import replace

import pytest

from human_design.rag.config import load_config
from human_design.rag.hybrid_index import HybridIndexError
from human_design.rag.hybrid_retriever import HybridRetriever
from human_design.rag.models import HybridSearchRequest, RetrievedChunk


REQUEST = HybridSearchRequest("original question", "semantic question", "Gate 42 exact terms")


class FakeRetriever:
    def __init__(self, chunks=(), ingestion_id="canonical-build", error=None):
        self.chunks = list(chunks)
        self.ingestion_id = ingestion_id
        self.calls = []
        self.error = error

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        if self.error:
            raise self.error
        return self.chunks


def chunk(identity, *, dense=None, sparse=None, **changes):
    return replace(RetrievedChunk(
        identity, "Synthetic Gate 42 passage", "reference.pdf",
        source_relpath="books/reference.pdf", page_label="7", page_number=7,
        dense_rank=dense, dense_score=-0.2 if dense else None,
        sparse_rank=sparse, sparse_score=150.0 if sparse else None,
    ), **changes)


def config(**changes):
    return replace(load_config(env={}), **changes)


def test_separate_queries_top_k_identity_and_full_provenance():
    dense = FakeRetriever([chunk("shared", dense=1), chunk("dense", dense=2)])
    sparse = FakeRetriever([chunk("sparse", sparse=1), chunk("shared", sparse=2)])
    settings = config(dense_top_k=3, sparse_top_k=7, fusion_top_k=9, rrf_k=60)
    result = HybridRetriever(dense, sparse, settings).retrieve(REQUEST)
    assert dense.calls == [(REQUEST.dense_query, 3)]
    assert sparse.calls == [(REQUEST.sparse_query, 7)]
    assert result.request is REQUEST
    assert result.ingestion_id == "canonical-build"
    assert result.ingestion_version == settings.ingestion_version
    assert (result.dense_top_k, result.sparse_top_k, result.fusion_top_k,
            result.final_top_k, result.rrf_k) == (3, 7, 9, 4, 60)
    assert result.rerank_provider is result.rerank_model is None
    shared, sparse_only, dense_only = result.candidates
    assert [c.chunk_id for c in result.candidates] == ["shared", "sparse", "dense"]
    assert (shared.dense_rank, shared.sparse_rank) == (1, 2)
    assert (shared.dense_score, shared.sparse_score) == (-0.2, 150.0)
    assert shared.rrf_score == 1 / 61 + 1 / 62
    assert sparse_only.rrf_score == 1 / 61
    assert sparse_only.dense_rank is sparse_only.dense_score is None
    assert dense_only.rrf_score == 1 / 62
    assert dense_only.sparse_rank is dense_only.sparse_score is None
    assert all(c.rerank_score is None for c in result.candidates)


def test_raw_scores_do_not_change_fusion_order():
    def retrieve(score):
        return HybridRetriever(
            FakeRetriever([chunk("b", dense=1, dense_score=score)]),
            FakeRetriever([chunk("a", sparse=1, sparse_score=-score)]), config(),
        ).retrieve(REQUEST)
    assert [c.chunk_id for c in retrieve(10000).candidates] == ["a", "b"]
    assert [c.rrf_score for c in retrieve(10000).candidates] == [
        c.rrf_score for c in retrieve(-10000).candidates
    ]


def test_exact_tie_breaks_by_best_rank_then_lexical_id():
    # k=1: rank (2, 2) and rank (1, 5) both sum to exactly 2/3.
    dense = FakeRetriever([chunk("z", dense=1), chunk("a", dense=2), chunk("b", dense=2)])
    sparse = FakeRetriever([chunk("b", sparse=2), chunk("a", sparse=2), chunk("z", sparse=5)])
    result = HybridRetriever(dense, sparse, config(rrf_k=1)).retrieve(REQUEST)
    assert [c.chunk_id for c in result.candidates] == ["z", "a", "b"]


def test_truncates_only_after_fusion_and_deduplicates_only_by_id():
    dense = FakeRetriever([chunk("a", dense=1), chunk("shared", dense=2)])
    sparse = FakeRetriever([chunk("b", sparse=1), chunk("shared", sparse=2)])
    result = HybridRetriever(dense, sparse, config(fusion_top_k=1, final_top_k=1)).retrieve(REQUEST)
    assert [c.chunk_id for c in result.candidates] == ["shared"]
    result = HybridRetriever(dense, sparse, config()).retrieve(REQUEST)
    assert len(result.candidates) == 3  # identical text does not collapse different IDs


def test_repeated_hit_in_one_backend_counts_once_using_best_rank():
    dense = FakeRetriever([chunk("a", dense=3), chunk("a", dense=1)])
    result = HybridRetriever(dense, FakeRetriever(), config()).retrieve(REQUEST)
    assert len(result.candidates) == 1
    assert result.candidates[0].dense_rank == 1
    assert result.candidates[0].rrf_score == 1 / 61


@pytest.mark.parametrize("change", [
    {"text": "conflicting synthetic text"}, {"source_file": "other.pdf"},
    {"source_relpath": "other/reference.pdf"}, {"page_label": "8"}, {"page_number": 8},
    {"metadata": {"source_sha256": "other-fingerprint"}},
])
def test_conflicting_canonical_hits_fail(change):
    base = chunk("a", dense=1, metadata={"source_sha256": "original-fingerprint"})
    other = replace(base, dense_rank=None, dense_score=None, sparse_rank=1, **change)
    with pytest.raises(HybridIndexError, match="consistency"):
        HybridRetriever(FakeRetriever([base]), FakeRetriever([other]), config()).retrieve(REQUEST)


def test_sanitizes_metadata_and_preserves_safe_fields():
    candidate = chunk("a", dense=1, metadata={
        "source_file": "reference.pdf", "page_number": 7,
        "source_path": "/private/local/book.pdf", "private": "secret", "chunk_id": "a",
    })
    result = HybridRetriever(FakeRetriever([candidate]), FakeRetriever(), config()).retrieve(REQUEST)
    actual = result.candidates[0]
    assert actual.source_relpath == "books/reference.pdf"
    assert actual.metadata["page_number"] == 7
    assert "source_path" not in actual.metadata and "private" not in actual.metadata
    assert "/private/" not in repr(actual)


@pytest.mark.parametrize("identity", ["different-build", "", None])
def test_identity_checked_before_either_query(identity):
    dense, sparse = FakeRetriever(), FakeRetriever(ingestion_id=identity)
    with pytest.raises(HybridIndexError, match="identity"):
        HybridRetriever(dense, sparse, config()).retrieve(REQUEST)
    assert not dense.calls and not sparse.calls


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_backend_failure_has_no_fallback_or_private_path(backend):
    dense, sparse = FakeRetriever([chunk("a", dense=1)]), FakeRetriever([chunk("b", sparse=1)])
    target = dense if backend == "dense" else sparse
    target.error = FileNotFoundError("/private/local/index")
    with pytest.raises(HybridIndexError, match=backend) as caught:
        HybridRetriever(dense, sparse, config()).retrieve(REQUEST)
    assert "/private" not in str(caught.value)


def test_missing_adapter_is_not_a_production_fallback():
    dense = FakeRetriever()
    with pytest.raises(HybridIndexError, match="identity"):
        HybridRetriever(dense, None, config()).retrieve(REQUEST)
    assert not dense.calls


def test_empty_results_are_valid():
    assert HybridRetriever(FakeRetriever(), FakeRetriever(), config()).retrieve(REQUEST).candidates == ()


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_adapter_must_supply_its_one_based_rank(backend):
    missing_rank = FakeRetriever([chunk("a")])
    dense, sparse = (missing_rank, FakeRetriever()) if backend == "dense" else (FakeRetriever(), missing_rank)
    with pytest.raises(HybridIndexError, match="rank"):
        HybridRetriever(dense, sparse, config()).retrieve(REQUEST)
