"""Embedding model factory for the Phase 1 RAG pipeline."""

from __future__ import annotations

from llama_index.embeddings.openai import OpenAIEmbedding

from human_design.rag.config import AppConfig, DEFAULT_EMBEDDING_MODEL


def create_openai_embedding_model(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
) -> OpenAIEmbedding:
    """
    Create an OpenAI embedding model without making embedding requests.
    """
    return OpenAIEmbedding(model=embedding_model, api_key=api_key)


def create_openai_embedding_model_from_config(config: AppConfig) -> OpenAIEmbedding:
    """
    Create an OpenAI embedding model from app configuration.
    """
    return create_openai_embedding_model(
        embedding_model=config.embedding_model,
        api_key=config.openai_api_key,
    )
