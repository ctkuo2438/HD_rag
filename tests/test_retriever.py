from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from human_design.rag.config import load_config
from human_design.rag import retriever
from human_design.rag.retriever import DenseRetrieverAdapter, load_dense_retriever


def test_dense_adapter_preserves_provenance_and_changes_only_top_k(tmp_path: Path) -> None:
    from llama_index.core.schema import NodeWithScore, TextNode
    nodes = [TextNode(id_=str(i), text=f"Synthetic source {i}", metadata={
        "chunk_id": str(i), "source_file": "synthetic.pdf", "source_relpath": "books/synthetic.pdf",
        "page_label": "7", "page_number": 7, "document_title": "Synthetic reference",
        "source_path": str(tmp_path / "private.pdf"), "private_cache_path": str(tmp_path / "cache"),
        "ingestion_id": "identity",
    }) for i in range(2)]
    class FakeRetriever:
        similarity_top_k = 20
        def retrieve(self, query):
            assert query == "Gate 42"
            assert self.similarity_top_k == 2
            return [NodeWithScore(node=nodes[0], score=0.75), NodeWithScore(node=nodes[1], score=None)]
    backend = FakeRetriever()
    adapter = DenseRetrieverAdapter(backend, "identity")
    results = adapter.retrieve("Gate 42", 2)
    assert adapter.ingestion_id == "identity"
    with pytest.raises(AttributeError):
        adapter.ingestion_id = "other"
    assert backend.similarity_top_k == 20
    assert [result.chunk_id for result in results] == ["0", "1"]
    assert [result.dense_rank for result in results] == [1, 2]
    assert [result.dense_score for result in results] == [0.75, None]
    for result in results:
        assert result.source_relpath == "books/synthetic.pdf"
        assert result.page_label == "7" and result.page_number == 7
        assert result.sparse_rank is result.sparse_score is result.rrf_score is result.rerank_score is None
        assert "source_path" not in result.metadata and "private_cache_path" not in result.metadata


@pytest.mark.parametrize("mode", ["disabled", "no-key", "missing"])
def test_dense_loader_gates_before_provider_or_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    config = replace(load_config(env={}), index_dir=tmp_path / "hybrid",
                     real_embeddings=mode != "disabled", openai_api_key=None if mode == "no-key" else "fake-key")
    monkeypatch.setattr(retriever, "create_openai_embedding_model_from_config", lambda config: pytest.fail("no provider"))
    with pytest.raises(ValueError, match="HD_RAG_REAL_EMBEDDINGS|OPENAI_API_KEY|manifest"):
        load_dense_retriever(config)
    assert not config.index_dir.exists()


def test_dense_loader_opens_validated_store_without_rebuilding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from human_design.rag import hybrid_index as hybrid
    config = replace(load_config(env={}), index_dir=tmp_path / "hybrid", real_embeddings=True, openai_api_key="fake")
    backend = SimpleNamespace(similarity_top_k=2, retrieve=lambda query: [])
    manifest = SimpleNamespace(ingestion_id="validated-identity")
    monkeypatch.setattr(hybrid, "open_phase3_chroma", lambda config: ("store", manifest))
    monkeypatch.setattr(retriever, "create_openai_embedding_model_from_config", lambda config: "fake-embedding")
    class ReadOnlyIndex:
        def __init__(self, *args, **kwargs):
            pytest.fail("must not build an index")
        @classmethod
        def from_vector_store(cls, *, vector_store, embed_model):
            assert vector_store == "store" and embed_model == "fake-embedding"
            return SimpleNamespace(as_retriever=lambda **kwargs: backend)
    monkeypatch.setattr(retriever, "VectorStoreIndex", ReadOnlyIndex)
    adapter = load_dense_retriever(config)
    assert adapter.ingestion_id == "validated-identity"
    assert adapter.retrieve("Question", 1) == []
