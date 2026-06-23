# Human Design RAG

This repository is currently in Phase 1: a local, text-only Human Design knowledge base built from local PDF books.

Phase 1 uses this fixed pipeline:

```text
Local PDFs
  -> LlamaIndex PyMuPDFReader
  -> embedded PDF text extraction only
  -> SentenceSplitter(chunk_size=800, chunk_overlap=80)
  -> OpenAIEmbedding(text-embedding-3-small)
  -> local persistent Chroma vector store
```

Phase 1 does not include image parsing, OCR, BodyGraph interpretation, API deployment, a web app, or final RAG answer generation. The retrieval script is a smoke test that returns retrieved chunks and source metadata only.

## Setup

Install the project dependencies with uv:

```sh
uv sync
```

Run the default verification commands:

```sh
uv run pytest
uv run ruff check .
```

## Environment

Copy `.env.example` to `.env` for local manual ingestion and retrieval work. Do not put real secret values in `.env.example`, README examples, commits, logs, or test fixtures.

Supported environment variables:

```env
OPENAI_API_KEY=
HD_RAG_PDF_DIR=data/pdfs
HD_RAG_CHROMA_DIR=storage/chroma
HD_RAG_COLLECTION=human_design
HD_RAG_EMBED_MODEL=text-embedding-3-small
HD_RAG_CHUNK_SIZE=800
HD_RAG_CHUNK_OVERLAP=80
HD_RAG_INGESTION_VERSION=v1
```

`OPENAI_API_KEY` is required only for manual real embedding ingestion and retrieval against an existing Chroma store. Default tests do not require it.

## Local Data

Local PDF source files live under:

```text
data/pdfs/
```

Local Chroma storage lives under:

```text
storage/chroma/
```

These are local development paths. Do not commit PDFs, `.env`, API keys, `storage/`, Chroma DB files, local embedding caches, or generated full-text dumps from copyrighted PDFs.

## Default Verification

Default tests are designed to be free and deterministic:

- They do not call OpenAI.
- They do not require `OPENAI_API_KEY`.
- They do not require real PDFs in `data/pdfs/`.
- They do not require an existing real Chroma collection.
- They should not create repo-level `storage/chroma`.

Use this verification flow when checking the offline test path:

```sh
rm -rf storage/chroma
env -u OPENAI_API_KEY uv run pytest
uv run ruff check .
test ! -d storage/chroma
```

## Manual Ingestion

Real embedding ingestion is manual and opt-in:

```sh
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py
```

This command may call OpenAI and create embedding cost. It requires local PDFs and `OPENAI_API_KEY`, and it writes vectors into the configured local Chroma directory.

Do not run real ingestion as part of default tests or routine offline verification.

## Retrieval Smoke Test

After manual ingestion has created an existing Chroma store, run:

```sh
uv run python scripts/query_kb.py "What is a Generator type?"
```

This may call OpenAI for the query embedding. It reloads the existing Chroma collection, retrieves top matching chunks with source metadata, and does not generate final LLM answers.
