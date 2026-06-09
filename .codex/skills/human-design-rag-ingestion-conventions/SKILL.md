---
name: human-design-rag-ingestion-conventions
description: |
  Use when writing or reviewing PDF ingestion, chunking, embedding, vector store,
  or retrieval code for Human Design RAG Phase 1. This locks the Phase 1 pipeline:
  PyMuPDFReader, text-only extraction, SentenceSplitter, OpenAI embeddings, and Chroma.
---

# Human Design RAG ingestion conventions

## Purpose

Use this skill when writing or reviewing ingestion, chunking, embedding, vector store, or retrieval code for Human Design RAG Phase 1.

## Fixed Phase 1 pipeline

The Phase 1 ingestion pipeline is fixed:

Local PDFs
   ↓
LlamaIndex PyMuPDFReader
   ↓
Extract embedded PDF text only
   ↓
SentenceSplitter(chunk_size=800, chunk_overlap=80)
   ↓
OpenAIEmbedding(text-embedding-3-small)
   ↓
Chroma persistent vector store

## Required implementation choices

Use:
- LlamaIndex PyMuPDFReader for PDF loading
- LlamaIndex SentenceSplitter for chunking
- chunk_size=800
- chunk_overlap=80
- OpenAIEmbedding with model text-embedding-3-small
- Chroma persistent local vector store

Do not use:
- SimpleDirectoryReader unless the user explicitly changes the decision
- LangChain
- Pinecone
- S3
- LlamaParse
- OCR
- image interpretation
- BodyGraph diagram parsing inside PDFs

## Text-only extraction

Phase 1 extracts embedded PDF text only.

Embedded images, BodyGraph diagrams, sample chart visuals, scanned pages, and image-heavy content are out of scope.

If low-text pages are detected, record them in an extraction report if practical.
Do not add OCR or vision parsing in Phase 1.

## Required metadata on each chunk/node

Each chunk/node should preserve metadata where available:

- source_file
- page_label or page_number if available
- document_title if available
- chunk_size
- chunk_overlap
- embedding_model
- ingestion_version

Metadata is important because retrieval smoke tests must show where each chunk came from.

## Ingestion report

The ingestion flow should produce a clear report with:

- number of PDFs found
- number of documents/pages loaded
- number of chunks/nodes created
- number of chunks persisted
- collection name
- Chroma directory
- skipped or low-text pages if tracked

## Retrieval smoke test output

The retrieval smoke-test script should return:

- text snippet
- similarity score if available
- source_file
- page metadata if available

Example query:

    uv run python scripts/query_kb.py "What is a Generator type?"

## Rebuild behavior

If the embedding model changes, the Chroma collection should be rebuilt.

The ingestion script should make rebuild behavior explicit instead of silently mixing embeddings from different models.

## Phase 1 completion standard

Phase 1 is not complete just because vectors were persisted.

It is complete only when:
- PDFs can be loaded
- chunks are created with metadata
- embeddings are stored
- Chroma can be reloaded
- a query returns relevant chunks with source metadata
