"""Reload existing Chroma-backed indexes and create retrievers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode

from human_design.rag.config import AppConfig
from human_design.rag.embeddings import create_openai_embedding_model_from_config
from human_design.rag.models import RetrievedChunk
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


@dataclass(frozen=True)
class DenseRetrieverAdapter:
    """Small adapter over an already-open retriever; querying never builds an index."""

    _retriever: VectorIndexRetriever = field(repr=False)
    ingestion_id: str

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        from human_design.rag.hybrid_index import HybridIndexError, safe_source_metadata

        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        previous = self._retriever.similarity_top_k
        try:
            self._retriever.similarity_top_k = top_k
            matches = self._retriever.retrieve(query)
        finally:
            self._retriever.similarity_top_k = previous
        results = []
        for rank, match in enumerate(matches, 1):
            metadata = safe_source_metadata(match.node.metadata)
            if (metadata.get("chunk_id") != match.node.node_id or "source_file" not in metadata
                    or match.node.metadata.get("ingestion_id") != self.ingestion_id):
                raise HybridIndexError("Dense result lacks matching canonical ingestion identity")
            results.append(RetrievedChunk(
                chunk_id=match.node.node_id, text=match.node.get_content(metadata_mode=MetadataMode.NONE),
                source_file=metadata["source_file"], source_relpath=metadata.get("source_relpath"),
                document_title=metadata.get("document_title"), page_label=metadata.get("page_label"),
                page_number=metadata.get("page_number"), dense_rank=rank, dense_score=match.score, metadata=metadata,
            ))
        return results


def load_dense_retriever(config: AppConfig, *, embed_model: BaseEmbedding | None = None) -> DenseRetrieverAdapter:
    """Open only the validated Phase 3 collection with explicitly opted-in query embeddings."""
    from human_design.rag.hybrid_index import open_phase3_chroma, require_real_embeddings

    require_real_embeddings(config)
    vector_store, manifest = open_phase3_chroma(config)
    embedding = embed_model if embed_model is not None else create_openai_embedding_model_from_config(config)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedding)
    return DenseRetrieverAdapter(index.as_retriever(similarity_top_k=config.dense_top_k), manifest.ingestion_id)
