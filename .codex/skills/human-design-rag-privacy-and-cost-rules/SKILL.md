---
name: human-design-rag-privacy-and-cost-rules
description: |
  Use when handling .env, OpenAI API keys, local PDFs, Chroma storage, logs,
  embeddings, tests, or retrieval output. Default tests must not call paid APIs.
---

# Human Design RAG privacy and cost rules

## Purpose

Use this skill when handling environment variables, OpenAI API keys, local PDFs, Chroma storage, logs, embeddings, tests, or retrieval output.

The goal is to protect private files, avoid leaking copyrighted text, and avoid accidental paid API usage.

## Never commit private or generated files

Never commit:
- .env
- data/pdfs/
- storage/
- Chroma DB files
- local embedding caches
- API keys
- raw extracted full-text dumps from copyrighted PDFs

## Required .gitignore

.gitignore must include:

    .venv/
    .env
    data/pdfs/
    storage/
    __pycache__/
    .pytest_cache/

## Required .env.example

.env.example should include:

    OPENAI_API_KEY=
    HD_RAG_PDF_DIR=data/pdfs
    HD_RAG_CHROMA_DIR=storage/chroma
    HD_RAG_COLLECTION=human_design
    HD_RAG_EMBED_MODEL=text-embedding-3-small

## API key handling

The OpenAI API key must come from environment variables or .env.

Never:
- hardcode API keys
- print API keys
- include API keys in errors
- commit .env
- log full request bodies

## Default tests must be free

Default tests must not call OpenAI API.

Use mocked or fake embeddings for tests where possible.

Real embedding calls must be manual or opt-in, for example:

    HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py

Normal test command must not spend money:

    uv run pytest

## Logging rules

Never log:
- OPENAI_API_KEY
- full PDF text
- full chunks
- full prompts
- full retrieved passages in debug logs

OK to log:
- filename
- page number
- chunk count
- collection name
- embedding model name
- vector store path
- number of retrieved chunks
- short preview limited to a safe length

## Retrieval output

A user-facing retrieval smoke test may show a short snippet and source metadata.

Do not dump entire chunks by default.

## Embedding model consistency

The same embedding model must be used for ingestion and query.

If the embedding model changes:
- do not query the old collection with the new model
- rebuild the Chroma collection
- update metadata and ingestion version

## Cost awareness

OpenAI embeddings are paid API calls.

Avoid repeated ingestion unless necessary.

Tests should verify pipeline logic without real embeddings by default.
