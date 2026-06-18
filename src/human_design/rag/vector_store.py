"""Local Chroma vector store setup for the Phase 1 RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from human_design.rag.config import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION_NAME


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
) -> ChromaVectorStore:
    """
    Create a LlamaIndex wrapper around a persistent Chroma collection.
    """
    client = create_chroma_client(chroma_dir)
    collection = get_or_create_chroma_collection(client, collection_name)
    return ChromaVectorStore(chroma_collection=collection)
