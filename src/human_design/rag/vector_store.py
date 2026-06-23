"""Local Chroma vector store setup for the Phase 1 RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from human_design.rag.config import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION_NAME


EMBEDDING_MODEL_METADATA_KEY = "embedding_model"


def create_chroma_client(chroma_dir: Path = DEFAULT_CHROMA_DIR) -> Any:
    """
    Create a persistent local Chroma client.
    """
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_or_create_chroma_collection(
    client: Any,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    """
    Return an existing Chroma collection or create it if missing.
    """
    return client.get_or_create_collection(name=collection_name)


def create_chroma_vector_store(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str | None = None,
) -> ChromaVectorStore:
    """
    Create a LlamaIndex wrapper around a persistent Chroma collection.
    """
    client = create_chroma_client(chroma_dir)
    collection = get_or_create_chroma_collection(client, collection_name)
    if embedding_model is not None:
        ensure_collection_embedding_model(collection, embedding_model)
    # I have a Chroma collection, wrap it into a vector store that LlamaIndex can work with
    return ChromaVectorStore(chroma_collection=collection)


def ensure_collection_embedding_model(collection: Any, embedding_model: str) -> None:
    """
    Ensure a Chroma collection is compatible with an embedding model.
    """
    metadata = dict(collection.metadata or {})
    existing_model = metadata.get(EMBEDDING_MODEL_METADATA_KEY)

    if existing_model == embedding_model:
        return
    if existing_model is not None:
        raise ValueError(
            "Embedding model mismatch: "
            f"collection uses {existing_model!r}, requested {embedding_model!r}"
        )
    if collection.count() > 0:
        raise ValueError(
            "Cannot use non-empty Chroma collection with unknown embedding model"
        )

    metadata[EMBEDDING_MODEL_METADATA_KEY] = embedding_model
    collection.modify(metadata=metadata)
