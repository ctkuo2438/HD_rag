from pathlib import Path

import pytest
from llama_index.vector_stores.chroma import ChromaVectorStore

from human_design.rag import vector_store
from human_design.rag.config import DEFAULT_COLLECTION_NAME


def test_create_chroma_client_uses_configured_directory(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma"

    client = vector_store.create_chroma_client(chroma_dir)
    collection = client.get_or_create_collection(name="client_probe")

    assert chroma_dir.exists()
    assert collection.name == "client_probe"


def test_get_or_create_chroma_collection_uses_default_collection_name(
    tmp_path: Path,
) -> None:
    client = vector_store.create_chroma_client(tmp_path / "chroma")

    collection = vector_store.get_or_create_chroma_collection(client)

    assert collection.name == DEFAULT_COLLECTION_NAME


def test_existing_collection_can_be_reopened_without_deleting_data(
    tmp_path: Path,
) -> None:
    chroma_dir = tmp_path / "chroma"
    client = vector_store.create_chroma_client(chroma_dir)
    collection = vector_store.get_or_create_chroma_collection(client, "persistent_test")
    collection.add(
        ids=["test-id"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["test document"],
        metadatas=[{"source_file": "test.pdf"}],
    )

    reopened_client = vector_store.create_chroma_client(chroma_dir)
    reopened_collection = vector_store.get_or_create_chroma_collection(
        reopened_client,
        "persistent_test",
    )

    assert reopened_collection.count() == 1


def test_create_chroma_vector_store_returns_llamaindex_chroma_vector_store(
    tmp_path: Path,
) -> None:
    chroma_store = vector_store.create_chroma_vector_store(
        chroma_dir=tmp_path / "chroma",
        collection_name="vector_store_test",
    )

    assert isinstance(chroma_store, ChromaVectorStore)


def test_empty_collection_records_embedding_model_metadata(tmp_path: Path) -> None:
    client = vector_store.create_chroma_client(tmp_path / "chroma")
    collection = vector_store.get_or_create_chroma_collection(client, "empty_metadata")

    vector_store.ensure_collection_embedding_model(collection, "model-a")

    assert collection.metadata == {
        vector_store.EMBEDDING_MODEL_METADATA_KEY: "model-a",
    }


def test_existing_collection_with_matching_embedding_model_passes(
    tmp_path: Path,
) -> None:
    client = vector_store.create_chroma_client(tmp_path / "chroma")
    collection = client.get_or_create_collection(
        name="matching_metadata",
        metadata={vector_store.EMBEDDING_MODEL_METADATA_KEY: "model-a"},
    )

    vector_store.ensure_collection_embedding_model(collection, "model-a")

    assert collection.metadata[vector_store.EMBEDDING_MODEL_METADATA_KEY] == "model-a"


def test_existing_collection_with_mismatched_embedding_model_raises(
    tmp_path: Path,
) -> None:
    client = vector_store.create_chroma_client(tmp_path / "chroma")
    collection = client.get_or_create_collection(
        name="mismatched_metadata",
        metadata={vector_store.EMBEDDING_MODEL_METADATA_KEY: "model-a"},
    )

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        vector_store.ensure_collection_embedding_model(collection, "model-b")


def test_non_empty_collection_without_embedding_model_metadata_raises(
    tmp_path: Path,
) -> None:
    client = vector_store.create_chroma_client(tmp_path / "chroma")
    collection = vector_store.get_or_create_chroma_collection(client, "unknown_model")
    collection.add(
        ids=["test-id"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["test document"],
        metadatas=[{"source_file": "test.pdf"}],
    )

    with pytest.raises(ValueError, match="unknown embedding model"):
        vector_store.ensure_collection_embedding_model(collection, "model-a")


def test_create_chroma_vector_store_enforces_embedding_model_when_provided(
    tmp_path: Path,
) -> None:
    chroma_store = vector_store.create_chroma_vector_store(
        chroma_dir=tmp_path / "chroma",
        collection_name="enforced_model",
        embedding_model="model-a",
    )
    client = vector_store.create_chroma_client(tmp_path / "chroma")
    collection = vector_store.get_or_create_chroma_collection(client, "enforced_model")

    assert isinstance(chroma_store, ChromaVectorStore)
    assert collection.metadata == {
        vector_store.EMBEDDING_MODEL_METADATA_KEY: "model-a",
    }


def test_vector_store_does_not_call_pdf_loading_or_chunking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from human_design.rag import chunking, ingestion

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("vector store setup must not load PDFs or chunk documents")

    monkeypatch.setattr(ingestion, "load_pdf", fail_if_called)
    monkeypatch.setattr(ingestion, "load_pdfs", fail_if_called)
    monkeypatch.setattr(chunking, "chunk_documents", fail_if_called)

    chroma_store = vector_store.create_chroma_vector_store(
        chroma_dir=tmp_path / "chroma",
        collection_name="no_pipeline_calls",
    )

    assert isinstance(chroma_store, ChromaVectorStore)


def test_vector_store_does_not_require_openai_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    chroma_store = vector_store.create_chroma_vector_store(
        chroma_dir=tmp_path / "chroma",
        collection_name="no_openai_key",
    )

    assert isinstance(chroma_store, ChromaVectorStore)


def test_vector_store_tests_do_not_create_repo_level_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    vector_store.create_chroma_vector_store(
        chroma_dir=tmp_path / "isolated_chroma",
        collection_name="isolated_collection",
    )

    assert not (tmp_path / "storage").exists()
    assert not Path("storage/chroma").exists()
