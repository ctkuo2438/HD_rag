from pathlib import Path
from unittest.mock import Mock

import pytest

from human_design.rag import vector_store


def test_chroma_client_requires_explicit_storage_path(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = Mock()
    monkeypatch.setattr(vector_store.chromadb, "PersistentClient", constructor)
    with pytest.raises(TypeError):
        vector_store.create_chroma_client()
    constructor.assert_not_called()


def test_create_chroma_client_uses_configured_directory(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma"
    client = vector_store.create_chroma_client(chroma_dir)
    collection = client.create_collection(name="client_probe", embedding_function=None)
    assert chroma_dir.exists()
    assert collection.name == "client_probe"


def test_existing_collection_can_be_reopened_without_deleting_data(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma"
    client = vector_store.create_chroma_client(chroma_dir)
    collection = client.create_collection(name="persistent_test", embedding_function=None)
    collection.add(
        ids=["test-id"], embeddings=[[0.1, 0.2, 0.3]], documents=["test document"],
        metadatas=[{"source_file": "test.pdf"}],
    )
    before = collection.get(include=["documents", "metadatas"])
    reopened_client = vector_store.create_chroma_client(chroma_dir)
    reopened_collection = reopened_client.get_collection("persistent_test", embedding_function=None)
    assert reopened_collection.get(include=["documents", "metadatas"]) == before
