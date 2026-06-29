"""Reload existing Chroma-backed indexes and create retrievers."""

from __future__ import annotations

from typing import Any

from llama_index.core import VectorStoreIndex

from human_design.rag.config import AppConfig
from human_design.rag.embeddings import create_openai_embedding_model_from_config
from human_design.rag.vector_store import open_existing_chroma_vector_store


def load_existing_chroma_index(config: AppConfig) -> VectorStoreIndex:
    """
    Load an existing Chroma-backed index without re-ingesting documents.
    """
    try:
        vector_store = open_existing_chroma_vector_store(
            chroma_dir=config.chroma_dir,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc}. Please rebuild the Chroma collection with the configured "
            "embedding model before retrieval."
        ) from exc

    embed_model = create_openai_embedding_model_from_config(config)
    return load_existing_chroma_index_from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def load_existing_chroma_index_from_vector_store(
        vector_store: Any, 
        embed_model: Any,
    ) -> VectorStoreIndex:
    """
    Create a LlamaIndex object from an existing vector store.
    """
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def create_retriever_from_config(
    config: AppConfig,
    similarity_top_k: int = 5,
) -> Any:
    """
    Create a retriever from an existing configured Chroma collection.
    """
    index = load_existing_chroma_index(config)
    return index.as_retriever(similarity_top_k=similarity_top_k)
