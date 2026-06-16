"""Chunk already-loaded LlamaIndex documents for the Phase 1 RAG pipeline."""

from __future__ import annotations

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from human_design.rag.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INGESTION_VERSION,
)
from human_design.rag.models import ChunkingResult


def chunk_documents(
        documents: list[Document],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, 
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        ingestion_version: str = DEFAULT_INGESTION_VERSION,
    ) -> tuple[list[BaseNode], ChunkingResult]:

    # use LlamaIndex's SentenceSplitter to chunk the documents into nodes
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes:
        _normalize_node_metadata(
            node,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            ingestion_version=ingestion_version,
        )

    return nodes, ChunkingResult(
        document_count=len(documents),
        chunk_count=len(nodes),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _normalize_node_metadata(
        node: BaseNode, 
        *,
        chunk_size: int, 
        chunk_overlap: int, 
        embedding_model: str, 
        ingestion_version: str
    ) -> None:
    metadata = dict(node.metadata)

    file_name = metadata.get("file_name")
    if file_name is not None:
        metadata["source_file"] = str(file_name)

    if metadata.get("page_label") is None and metadata.get("source") is not None:
        metadata["page_label"] = str(metadata["source"])

    if metadata.get("page_number") is None:
        source_page_number = _int_like(metadata.get("source"))
        if source_page_number is not None:
            metadata["page_number"] = source_page_number

    metadata["chunk_size"] = chunk_size
    metadata["chunk_overlap"] = chunk_overlap
    metadata["embedding_model"] = embedding_model
    metadata["ingestion_version"] = ingestion_version

    node.metadata = metadata


def _int_like(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return int(stripped_value)
        except ValueError:
            return None

    return None
