---
name: human-design-rag-ingestion-conventions
description: |
  Use when writing or reviewing PDF ingestion, chunking, embedding, vector store,
  or retrieval code for Human Design RAG Phase 1 maintenance only. This locks the
  Phase 1 pipeline: PyMuPDFReader, text-only extraction, SentenceSplitter, OpenAI
  embeddings, and Chroma. It does not apply to Phase 2 BodyGraph Vision work.
---

# Human Design RAG Ingestion Conventions

## Purpose

Use this skill only when writing or reviewing ingestion, chunking, embedding, vector store, or retrieval code for Human Design RAG Phase 1 maintenance.

This skill protects the completed Phase 1 PDF-to-Chroma retrieval baseline.

It does not apply to Phase 2 BodyGraph Vision extraction, including Vision API calls, chart-image parsing, deterministic BodyGraph interpretation, validation, evaluation, or Vision CLI work.

## Fixed Phase 1 Pipeline

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

## Required Implementation Choices

Use:

- LlamaIndex PyMuPDFReader for PDF loading
- LlamaIndex SentenceSplitter for chunking
- `chunk_size=800`
- `chunk_overlap=80`
- OpenAIEmbedding with model `text-embedding-3-small`
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

## Text-Only Extraction

Phase 1 extracts embedded PDF text only.

Embedded images, BodyGraph diagrams, sample chart visuals, scanned pages, and image-heavy content are out of scope.

If low-text pages are detected, record them in an extraction report if practical.

Do not add OCR or Vision parsing to the Phase 1 ingestion pipeline.

## Required Metadata on Each Chunk or Node

Each chunk or node should preserve metadata where available:

- `source_file`
- `page_label` or `page_number`
- `document_title`
- `chunk_size`
- `chunk_overlap`
- `embedding_model`
- `ingestion_version`

Metadata is important because retrieval smoke tests must show where each chunk came from.

## Ingestion Report

The ingestion flow should produce a clear report with:

- number of PDFs found
- number of documents or pages loaded
- number of chunks or nodes created
- number of chunks persisted
- collection name
- Chroma directory
- skipped or low-text pages, when tracked

## Retrieval Smoke-Test Output

The retrieval smoke-test script should return:

- text snippet
- similarity score, when available
- source file
- page metadata, when available

Example query:

    uv run python scripts/query_kb.py "What is a Generator type?"

## Rebuild Behavior

If the embedding model changes, the Chroma collection should be rebuilt.

The ingestion script should make rebuild behavior explicit instead of silently mixing embeddings from different models.

## Phase 1 Completion Standard

Phase 1 is not complete just because vectors were persisted.

It is complete only when:

- PDFs can be loaded
- chunks are created with metadata
- embeddings are stored
- Chroma can be reloaded
- a query returns relevant chunks with source metadata