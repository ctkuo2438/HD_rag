"""Local Chroma vector store setup for the Phase 1 RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
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


def open_existing_chroma_vector_store(
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
) -> ChromaVectorStore:
    """
    Open an existing non-empty Chroma collection for read-only retrieval.
    """
    if not chroma_dir.exists():
        raise ValueError(f"Chroma directory not found: {chroma_dir}. Run ingestion first.")
    if not chroma_dir.is_dir():
        raise ValueError(f"Chroma path is not a directory: {chroma_dir}")

    client = create_chroma_client(chroma_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except NotFoundError as exc:
        raise ValueError(
            f"Chroma collection not found: {collection_name!r}. Run or rebuild ingestion."
        ) from exc

    if collection.count() == 0:
        raise ValueError(
            f"Chroma collection is empty: {collection_name!r}. Run or rebuild ingestion."
        )
    ensure_collection_embedding_model(collection, embedding_model)
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
