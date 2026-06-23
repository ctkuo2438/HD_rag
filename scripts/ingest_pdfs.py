"""Opt-in local PDF ingestion into the Phase 1 Chroma vector store."""

from __future__ import annotations

import os

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma.base import ChromaVectorStore

from human_design.rag.chunking import chunk_documents
from human_design.rag.config import load_config
from human_design.rag.embeddings import create_openai_embedding_model_from_config
from human_design.rag.ingestion import (
    NoPdfFilesFoundError,
    build_text_extraction_report,
    load_pdfs,
)
from human_design.rag.vector_store import create_chroma_vector_store


REAL_EMBEDDINGS_ENV_VAR = "HD_RAG_REAL_EMBEDDINGS"
DISABLED_MESSAGE = (
    "Real embedding ingestion is disabled by default. "
    "Set HD_RAG_REAL_EMBEDDINGS=1 to run ingestion."
)

# pipeline
'''
load_config()
→ load_pdfs()
→ build_text_extraction_report()
→ chunk_documents()
→ create_openai_embedding_model_from_config()
→ create_chroma_vector_store()
→ StorageContext.from_defaults()
→ VectorStoreIndex()
'''
def main() -> None:
    """
    Run the local ingestion pipeline when real embeddings are explicitly enabled.
    """
    # HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py
    if os.environ.get(REAL_EMBEDDINGS_ENV_VAR) != "1":
        raise SystemExit(DISABLED_MESSAGE)
    
    try:
        config = load_config()
        if config.openai_api_key is None:
            raise SystemExit("OPENAI_API_KEY is not set. Please set it in your environment or .env file.")
        
        documents = load_pdfs(config.pdf_dir) # step 1 from ingestion.py
        extraction_report = build_text_extraction_report(config.pdf_dir) # step 2 from ingestion.py
    except NoPdfFilesFoundError as exc:
        raise SystemExit(str(exc)) from exc

    # step 3 from chunking.py
    nodes, chunk_result = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedding_model=config.embedding_model,
        ingestion_version=config.ingestion_version,
    )
    # step 4 from embeddings.py
    embed_model = create_openai_embedding_model_from_config(config)
    # step 5 from vector_store.py, get LlamaIndex ChromaVectorStore adapter
    # ChromaVectorStore = LlamaIndex adapter for Chroma DB
    vector_store: ChromaVectorStore = create_chroma_vector_store(
        chroma_dir=config.chroma_dir,
        collection_name=config.collection_name,
        embedding_model=config.embedding_model,
    )
    # step 6, create a LlamaIndex StorageContext
    # tell LlamaIndex: when the VectorStoreIndex needs to store the vectors, please store them in this ChromaVectorStore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # VectorStoreIndex = LlamaIndex's index / retrieval abstraction
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # summary
    _print_summary(
        pdf_count=extraction_report.pdf_count,
        document_count=extraction_report.document_count,
        chunk_count=chunk_result.chunk_count,
        persisted_count=len(nodes),
        collection_name=config.collection_name,
        chroma_dir=str(config.chroma_dir),
        embedding_model=config.embedding_model,
        ingestion_version=config.ingestion_version,
    )


def _print_summary(
    *,
    pdf_count: int,
    document_count: int,
    chunk_count: int,
    persisted_count: int,
    collection_name: str,
    chroma_dir: str,
    embedding_model: str,
    ingestion_version: str,
) -> None:
    print("Ingestion complete.")
    print(f"PDFs found: {pdf_count}")
    print(f"Documents/pages loaded: {document_count}")
    print(f"Chunks/nodes created: {chunk_count}")
    print(f"Chunks persisted: {persisted_count}")
    print(f"Collection name: {collection_name}")
    print(f"Chroma directory: {chroma_dir}")
    print(f"Embedding model: {embedding_model}")
    print(f"Ingestion version: {ingestion_version}")


if __name__ == "__main__":
    main()
