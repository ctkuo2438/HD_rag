from dataclasses import replace
from pathlib import Path

import pytest

from human_design.rag.config import load_config
from human_design.rag import retriever


def _config(tmp_path: Path):
    return replace(
        load_config(env={}),
        chroma_dir=tmp_path / "chroma",
        collection_name="human_design_test",
        embedding_model="text-embedding-3-small",
        openai_api_key=None,
    )


def test_load_existing_chroma_index_uses_config_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_create_chroma_vector_store(**kwargs: object) -> str:
        calls.append(
            "vector_store:"
            f"{kwargs['chroma_dir']}:"
            f"{kwargs['collection_name']}:"
            f"{kwargs['embedding_model']}"
        )
        return "vector-store"

    def fake_create_embedding_model_from_config(received_config: object) -> str:
        assert received_config is config
        calls.append("embedding-model")
        return "embed-model"

    class FakeVectorStoreIndex:
        @classmethod
        def from_vector_store(
            cls,
            *,
            vector_store: object,
            embed_model: object,
        ) -> str:
            calls.append(f"from_vector_store:{vector_store}:{embed_model}")
            return "index"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("retriever reload must not construct an index from nodes")

    monkeypatch.setattr(retriever, "create_chroma_vector_store", fake_create_chroma_vector_store)
    monkeypatch.setattr(
        retriever,
        "create_openai_embedding_model_from_config",
        fake_create_embedding_model_from_config,
    )
    monkeypatch.setattr(retriever, "VectorStoreIndex", FakeVectorStoreIndex)

    index = retriever.load_existing_chroma_index(config)

    assert index == "index"
    assert calls == [
        f"vector_store:{config.chroma_dir}:human_design_test:text-embedding-3-small",
        "embedding-model",
        "from_vector_store:vector-store:embed-model",
    ]


def test_create_retriever_from_config_passes_similarity_top_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    class FakeIndex:
        def as_retriever(self, *, similarity_top_k: int) -> str:
            return f"retriever:{similarity_top_k}"

    monkeypatch.setattr(
        retriever,
        "load_existing_chroma_index",
        lambda received_config: FakeIndex(),
    )

    result = retriever.create_retriever_from_config(config, similarity_top_k=7)

    assert result == "retriever:7"


def test_embedding_model_mismatch_raises_rebuild_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    def fake_create_chroma_vector_store(**kwargs: object) -> object:
        raise ValueError("Embedding model mismatch")

    monkeypatch.setattr(retriever, "create_chroma_vector_store", fake_create_chroma_vector_store)

    with pytest.raises(ValueError, match="rebuild the Chroma collection"):
        retriever.load_existing_chroma_index(config)


def test_retriever_does_not_load_pdfs_or_chunk_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    from human_design.rag import chunking, ingestion

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("retriever reload must not ingest or chunk documents")

    monkeypatch.setattr(ingestion, "load_pdf", fail_if_called)
    monkeypatch.setattr(ingestion, "load_pdfs", fail_if_called)
    monkeypatch.setattr(ingestion, "build_text_extraction_report", fail_if_called)
    monkeypatch.setattr(chunking, "chunk_documents", fail_if_called)
    monkeypatch.setattr(retriever, "create_chroma_vector_store", lambda **kwargs: "vector-store")
    monkeypatch.setattr(
        retriever,
        "create_openai_embedding_model_from_config",
        lambda received_config: "embed-model",
    )

    class FakeVectorStoreIndex:
        @classmethod
        def from_vector_store(
            cls,
            *,
            vector_store: object,
            embed_model: object,
        ) -> str:
            return "index"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("retriever reload must not persist nodes")

    monkeypatch.setattr(retriever, "VectorStoreIndex", FakeVectorStoreIndex)

    assert retriever.load_existing_chroma_index(config) == "index"


def test_retriever_default_tests_do_not_require_openai_api_key_or_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(retriever, "create_chroma_vector_store", lambda **kwargs: "vector-store")
    monkeypatch.setattr(
        retriever,
        "create_openai_embedding_model_from_config",
        lambda received_config: "embed-model",
    )

    class FakeVectorStoreIndex:
        @classmethod
        def from_vector_store(
            cls,
            *,
            vector_store: object,
            embed_model: object,
        ) -> str:
            return "index"

    monkeypatch.setattr(retriever, "VectorStoreIndex", FakeVectorStoreIndex)

    assert retriever.load_existing_chroma_index(config) == "index"
    assert not (tmp_path / "storage" / "chroma").exists()


def test_retriever_does_not_append_or_delete_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(retriever, "create_chroma_vector_store", lambda **kwargs: "vector-store")
    monkeypatch.setattr(
        retriever,
        "create_openai_embedding_model_from_config",
        lambda received_config: "embed-model",
    )

    class FakeVectorStoreIndex:
        @classmethod
        def from_vector_store(
            cls,
            *,
            vector_store: object,
            embed_model: object,
        ) -> str:
            return "index"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("retriever reload must not append vectors")

        def delete_ref_doc(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("retriever reload must not delete vectors")

    monkeypatch.setattr(retriever, "VectorStoreIndex", FakeVectorStoreIndex)

    assert retriever.load_existing_chroma_index(config) == "index"
