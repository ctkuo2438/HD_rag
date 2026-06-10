# Phase 1 Implementation Plan: Local Human Design RAG Knowledge Base

> **For agentic workers:** Implement this plan task-by-task. Use TDD where practical: write each test first, verify it fails for the expected reason, implement the minimum code, then verify it passes. Use available planning/execution skills when helpful.

**Goal:** Build a local, text-only RAG knowledge base from Human Design PDF books using LlamaIndex, OpenAI embeddings, and Chroma.

**Architecture:** Phase 1 loads local PDFs, extracts embedded PDF text only, chunks the text with fixed SentenceSplitter settings, embeds chunks with `text-embedding-3-small`, and persists vectors in a local Chroma store. Phase 1 includes ingestion plus retrieval smoke testing only; it does not generate answers.

**Tech Stack:** Python 3.11+, uv, LlamaIndex, PyMuPDFReader, OpenAI embeddings, Chroma, pytest, ruff.

---

## Goal

Build a local, text-only RAG knowledge base from Human Design PDF books using LlamaIndex, OpenAI embeddings, and Chroma.

Phase 1 is knowledge base construction plus retrieval smoke testing. It is not a chatbot and does not perform RAG answer generation.

## Non-Goals

Explicitly out of scope for Phase 1:

- AWS
- S3
- Lambda
- API Gateway
- EC2
- SageMaker
- Pinecone
- OpenSearch
- LangChain
- SQL database
- user accounts
- Streamlit
- FastAPI
- web app
- Claude reading generation
- Vision extraction
- BodyGraph image interpretation
- OCR
- LlamaParse
- Tree Index
- SQL agent
- RAG answer generation

## Architecture

Use this fixed Phase 1 pipeline:

```text
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
```

Do not change this architecture unless the user explicitly changes Phase 1 scope.

## Tech Stack

- Python 3.11+: runtime for the local RAG pipeline.
- uv: dependency management, virtual environment, and command runner.
- `llama-index`: core indexing, nodes, storage context, and retrieval integration.
- `llama-index-readers-file`: provides `PyMuPDFReader` for PDF loading.
- `llama-index-embeddings-openai`: provides `OpenAIEmbedding`.
- `llama-index-vector-stores-chroma`: LlamaIndex integration for Chroma.
- `chromadb`: persistent local vector store backend.
- `pymupdf`: PDF text extraction dependency.
- `python-dotenv`: optional `.env` loading for local development.
- `pytest`: default test runner.
- `ruff`: linting.

Setup commands:

```sh
uv init
uv venv
source .venv/bin/activate
uv add llama-index llama-index-readers-file llama-index-embeddings-openai llama-index-vector-stores-chroma chromadb pymupdf python-dotenv
uv add --dev pytest ruff
```

## Target File Map

```text
<repo-root>/
├── data/
│   └── pdfs/
├── storage/
│   └── chroma/
├── scripts/
│   ├── ingest_pdfs.py
│   └── query_kb.py
├── src/
│   └── human_design/
│       ├── __init__.py
│       └── rag/
│           ├── __init__.py
│           ├── config.py
│           ├── models.py
│           ├── ingestion.py
│           ├── vector_store.py
│           └── retriever.py
├── tests/
│   ├── test_config.py
│   ├── test_ingestion.py
│   ├── test_vector_store.py
│   └── test_retriever.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Configuration

Configuration belongs in `src/human_design/rag/config.py` and should be represented with a typed dataclass such as `AppConfig`.

Config loading should:

- load environment values from `.env` when present
- allow tests to pass explicit environment mappings or monkeypatched values
- resolve path fields with `pathlib.Path`
- default `HD_RAG_PDF_DIR` to `data/pdfs`
- default `HD_RAG_CHROMA_DIR` to `storage/chroma`
- default `HD_RAG_COLLECTION` to `human_design`
- default `HD_RAG_EMBED_MODEL` to `text-embedding-3-small`
- store `chunk_size=800`
- store `chunk_overlap=80`
- store an `ingestion_version`

Required `.env.example`:

```env
OPENAI_API_KEY=
HD_RAG_PDF_DIR=data/pdfs
HD_RAG_CHROMA_DIR=storage/chroma
HD_RAG_COLLECTION=human_design
HD_RAG_EMBED_MODEL=text-embedding-3-small
```

## Data and Storage Rules

- Local PDFs live in `data/pdfs/`.
- Chroma data lives in `storage/chroma/`.
- Neither PDFs nor Chroma storage should be committed.
- `.env` should not be committed.
- `.env.example` is safe to commit.
- Do not commit raw extracted full-text dumps from copyrighted PDFs.
- Do not log full PDF text, full chunks, API keys, or full retrieved passages by default.

Required `.gitignore` entries:

```gitignore
.venv/
.env
data/pdfs/
storage/
__pycache__/
.pytest_cache/
```

## Required Metadata

Each chunk/node should preserve metadata where available:

- `source_file`
- `page_label` or `page_number` if available
- `document_title` if available
- `chunk_size`
- `chunk_overlap`
- `embedding_model`
- `ingestion_version`

Metadata matters because retrieval smoke tests need source information, future RAG answers will need citations, and debugging ingestion requires source traceability.

## Testing Strategy

- Default tests must not call the OpenAI API.
- Use fake or mocked embeddings in tests.
- Real embedding integration must be manual or opt-in.
- Use `uv run pytest` for tests.
- Use `uv run ruff check .` for linting.
- For every behavior change, write a failing test first, verify the expected failure, implement the minimum code, then verify it passes.

## Recommended Codex Skills

During planning, use:

- `human-design-rag-phase-discipline`
- `writing-plans`

During implementation, use:

- `human-design-rag-module-conventions`
- `human-design-rag-ingestion-conventions`

During tests, API key handling, logging, `.gitignore`, `.env.example`, and OpenAI embedding work, use:

- `human-design-rag-privacy-and-cost-rules`

Do not reuse SnapScript skills directly because they were for LLM-generated code execution.

Phase 1 does not need:

- `safety_checker`
- `sandbox_executor`
- AST safety rules
- Claude code-generation retry rules
- subprocess execution rules

## Implementation Tasks

### Task 1: Project skeleton and existing uv setup

Goal:
Confirm the existing uv project setup and create the package skeleton without adding RAG implementation behavior yet.

Files touched:
- `pyproject.toml` only if package configuration needs a minimal update
- `src/human_design/__init__.py`
- `src/human_design/rag/__init__.py`
- `scripts/.gitkeep`
- `tests/.gitkeep`
- `main.py` only if it is the default uv placeholder and should be removed or left untouched

Dependencies:
- Python 3.11+
- uv
- Existing `pyproject.toml`

Acceptance criteria:
- Existing `pyproject.toml` is preserved unless a minimal package-layout update is required.
- Project uses uv.
- Package layout supports imports from `src/human_design/rag/`.
- `src/human_design/__init__.py` exists.
- `src/human_design/rag/__init__.py` exists.
- `scripts/` and `tests/` directories exist, using `.gitkeep` if they would otherwise be empty.
- No PDF ingestion, OpenAI calls, or Chroma storage is created in this task.
- No `.env`, PDFs, or Chroma DB files are committed.

Verification:

```sh
uv run python -c "import human_design.rag"
uv run ruff check .
git status --short
```

### Task 2: Dependencies, `.gitignore`, and `.env.example`

Goal:
Add Phase 1 dependencies, protect private/generated files, and document required environment variables.

Files touched:
- `pyproject.toml`
- `uv.lock`
- `.gitignore`
- `.env.example`

Dependencies:
- uv
- Existing `pyproject.toml`

Acceptance criteria:
- Runtime dependencies are installed:
  - `llama-index`
  - `llama-index-readers-file`
  - `llama-index-embeddings-openai`
  - `llama-index-vector-stores-chroma`
  - `chromadb`
  - `pymupdf`
  - `python-dotenv`
- Dev dependencies are installed:
  - `pytest`
  - `ruff`
- `uv.lock` is created or updated and should be committed.
- `.gitignore` includes:
  - `.venv/`
  - `.env`
  - `data/pdfs/`
  - `storage/`
  - `__pycache__/`
  - `.pytest_cache/`
  - `.ruff_cache/`
- `.env.example` includes:
  - `OPENAI_API_KEY=`
  - `HD_RAG_PDF_DIR=data/pdfs`
  - `HD_RAG_CHROMA_DIR=storage/chroma`
  - `HD_RAG_COLLECTION=human_design`
  - `HD_RAG_EMBED_MODEL=text-embedding-3-small`
- `.env`, PDFs, and Chroma storage are not committed.
- No PDF ingestion code is implemented.
- No OpenAI API calls are made.
- No Chroma storage is created.

Verification:

```sh
uv run python -c "import llama_index; import chromadb; import fitz"
uv run pytest
uv run ruff check .
test -f .env.example
grep -q '^OPENAI_API_KEY=' .env.example
grep -q '^HD_RAG_COLLECTION=human_design$' .env.example
grep -q '^storage/$' .gitignore
grep -q '^.ruff_cache/$' .gitignore
git status --short
```

### Task 3: App config and environment loading

Goal:
Create typed configuration loading with defaults and path resolution.

Files touched:
- `src/human_design/rag/config.py`
- `tests/test_config.py`

Dependencies:
- `python-dotenv`
- `pytest`

Acceptance criteria:
- `AppConfig` is a dataclass.
- Config uses `pathlib.Path` for PDF and Chroma directories.
- PDF directory defaults to `data/pdfs`.
- Chroma directory defaults to `storage/chroma`.
- Chroma collection name defaults to `human_design`.
- Embedding model defaults to `text-embedding-3-small`.
- Chunk settings default to `chunk_size=800` and `chunk_overlap=80`.
- Config includes an `ingestion_version` default, for example `v1`.
- Config loading can read from environment variables.
- Config loading can optionally load `.env` when present.
- Tests can pass explicit environment mappings or monkeypatch environment variables.
- Tests do not require `.env`.
- Tests do not require `OPENAI_API_KEY`.
- Tests do not call OpenAI API.
- No PDF ingestion, Chroma storage, or vector DB files are created.

Verification:

```sh
uv run pytest tests/test_config.py
uv run ruff check .
git status --short
```

### Task 4: Core dataclasses in `models.py`

Goal:
Define typed result objects used by ingestion, vector storage, and retrieval.

Files touched:
- `src/human_design/rag/models.py`
- `tests/test_ingestion.py`
- `tests/test_vector_store.py`
- `tests/test_retriever.py`

Dependencies:
- `pytest`

Acceptance criteria:
- Dataclasses exist for ingestion result, PDF load result, chunking result, vector store result, and retrieval result.
- Public result types expose counts and source metadata without requiring loose dicts.
- Functions that return more than two related values use dataclasses.

Verification:

```sh
uv run pytest tests/test_ingestion.py tests/test_vector_store.py tests/test_retriever.py
uv run ruff check .
```

### Task 5: PDF discovery and PyMuPDFReader loading

Goal:
Discover local PDFs and load them through LlamaIndex `PyMuPDFReader`.

Files touched:
- `src/human_design/rag/ingestion.py`
- `tests/test_ingestion.py`

Dependencies:
- `llama-index-readers-file`
- `pymupdf`
- `pytest`

Acceptance criteria:
- PDF discovery reads all `*.pdf` files from `data/pdfs/` or configured PDF directory.
- `uv run python scripts/ingest_pdfs.py` will load all PDFs from `data/pdfs/` once the script exists.
- Ingestion fails with a clear message if no PDFs exist.
- Extracted documents preserve source metadata.
- Core ingestion functions raise exceptions or return typed result objects; scripts handle user-facing messages later.

Verification:

```sh
uv run pytest tests/test_ingestion.py
uv run ruff check .
```

### Task 6: Text-only extraction report

Goal:
Track embedded text extraction results without adding OCR, vision, or image parsing.

Files touched:
- `src/human_design/rag/ingestion.py`
- `src/human_design/rag/models.py`
- `tests/test_ingestion.py`

Dependencies:
- `llama-index-readers-file`
- `pytest`

Acceptance criteria:
- Extraction report includes number of PDFs found, documents/pages loaded, and skipped or low-text pages if tracked.
- Low-text pages are reported when practical.
- OCR, LlamaParse, BodyGraph image interpretation, and vision extraction are not added.
- No raw full-text dumps are written.

Verification:

```sh
uv run pytest tests/test_ingestion.py
uv run ruff check .
```

### Task 7: Chunking with SentenceSplitter(chunk_size=800, chunk_overlap=80)

Goal:
Chunk loaded documents with fixed Phase 1 settings and preserve metadata on each chunk/node.

Files touched:
- `src/human_design/rag/ingestion.py`
- `src/human_design/rag/models.py`
- `tests/test_ingestion.py`

Dependencies:
- `llama-index`
- `pytest`

Acceptance criteria:
- `SentenceSplitter` uses `chunk_size=800`.
- `SentenceSplitter` uses `chunk_overlap=80`.
- Chunks are created with `chunk_size=800` and `chunk_overlap=80` metadata.
- Each chunk/node preserves `source_file`, `page_label` or `page_number` when available, `document_title` when available, `embedding_model`, and `ingestion_version`.
- Tests use sample in-memory documents and do not read real PDFs by default.

Verification:

```sh
uv run pytest tests/test_ingestion.py
uv run ruff check .
```

### Task 8: Chroma persistent vector store setup

Goal:
Create local Chroma client and collection setup without embedding real PDFs in tests.

Files touched:
- `src/human_design/rag/vector_store.py`
- `tests/test_vector_store.py`

Dependencies:
- `chromadb`
- `llama-index-vector-stores-chroma`
- `pytest`

Acceptance criteria:
- Vectors persist under `storage/chroma` or configured Chroma directory.
- Chroma collection name defaults to `human_design`.
- Existing collections can be opened without deleting data.
- Tests use a temporary directory and do not create repo-level `storage/`.

Verification:

```sh
uv run pytest tests/test_vector_store.py
uv run ruff check .
test ! -d storage/chroma
```

### Task 9: OpenAI embedding integration

Goal:
Add embedding factory wiring for `OpenAIEmbedding(text-embedding-3-small)` while keeping tests free by default.

Files touched:
- `src/human_design/rag/vector_store.py`
- `src/human_design/rag/config.py`
- `tests/test_vector_store.py`

Dependencies:
- `llama-index-embeddings-openai`
- `pytest`

Acceptance criteria:
- Embedding model defaults to `text-embedding-3-small`.
- OpenAI API key comes from environment variables or `.env`.
- Default tests must not call OpenAI API.
- Real embedding calls are manual or opt-in.
- Embedding model consistency is checked before querying or appending to an existing collection.

Verification:

```sh
uv run pytest tests/test_vector_store.py
uv run ruff check .
```

### Task 10: End-to-end ingestion script

Goal:
Create a CLI script that runs the local ingestion pipeline from configured PDFs to Chroma.

Files touched:
- `scripts/ingest_pdfs.py`
- `src/human_design/rag/ingestion.py`
- `src/human_design/rag/vector_store.py`
- `tests/test_ingestion.py`
- `tests/test_vector_store.py`

Dependencies:
- Tasks 3 through 9

Acceptance criteria:
- `uv run python scripts/ingest_pdfs.py` loads all PDFs from `data/pdfs/`.
- Ingestion prints a clear report with PDFs found, documents/pages loaded, chunks/nodes created, chunks persisted, collection name, and Chroma directory.
- Ingestion fails with a clear message if no PDFs exist.
- Real OpenAI embedding calls are manual or opt-in.
- The script does not dump full PDF text or full chunks.

Verification:

```sh
uv run pytest tests/test_ingestion.py tests/test_vector_store.py
uv run ruff check .
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py
```

Manual note: run the final command only when local PDFs exist and real OpenAI embedding calls are intentionally allowed.

### Task 11: Reload existing Chroma index without re-ingesting

Goal:
Support loading an existing Chroma-backed index for retrieval.

Files touched:
- `src/human_design/rag/vector_store.py`
- `src/human_design/rag/retriever.py`
- `tests/test_vector_store.py`
- `tests/test_retriever.py`

Dependencies:
- Tasks 8 and 9

Acceptance criteria:
- Existing Chroma store can be reloaded without re-ingesting PDFs.
- Query path uses the same embedding model used during ingestion.
- If the embedding model changes, the collection is not queried silently; the user is told to rebuild.

Verification:

```sh
uv run pytest tests/test_vector_store.py tests/test_retriever.py
uv run ruff check .
```

### Task 12: Retrieval smoke-test script

Goal:
Create a query script that returns top-k chunks and source metadata from the existing Chroma store.

Files touched:
- `scripts/query_kb.py`
- `src/human_design/rag/retriever.py`
- `tests/test_retriever.py`

Dependencies:
- Tasks 8 through 11

Acceptance criteria:
- Query script can reload the existing Chroma store without re-ingesting.
- `uv run python scripts/query_kb.py "What is a Generator type?"` returns top-k chunks with source metadata.
- Output includes text snippet, similarity score if available, `source_file`, and page metadata if available.
- The script does not dump entire chunks by default.

Verification:

```sh
uv run pytest tests/test_retriever.py
uv run ruff check .
uv run python scripts/query_kb.py "What is a Generator type?"
```

Manual note: the final command requires an existing Chroma store created from local PDFs.
Manual note: the final command may call OpenAI for query embedding, so run it only when an existing Chroma store exists and real embedding calls are intentionally allowed.

### Task 13: Tests with mocked or fake embeddings

Goal:
Make default verification free, deterministic, and independent of real PDFs.

Files touched:
- `tests/test_config.py`
- `tests/test_ingestion.py`
- `tests/test_vector_store.py`
- `tests/test_retriever.py`

Dependencies:
- `pytest`

Acceptance criteria:
- Default tests must not call OpenAI API.
- Tests use fake or mocked embeddings where possible.
- Tests use temporary PDF/vector directories where needed.
- `uv run pytest` passes without `OPENAI_API_KEY`.
- No Chroma DB files are created under repo-level `storage/` during tests.

Verification:

```sh
env -u OPENAI_API_KEY uv run pytest
uv run ruff check .
test ! -d storage/chroma
```

### Task 14: README update

Goal:
Document Phase 1 setup, local-only scope, commands, and privacy rules.

Files touched:
- `README.md`

Dependencies:
- Tasks 1 through 13

Acceptance criteria:
- README explains Phase 1 is local-only and text-only.
- README documents setup with uv.
- README documents `.env.example`.
- README documents where PDFs and Chroma storage live.
- README states default tests do not call OpenAI.
- README includes ingestion and retrieval smoke-test commands with manual/opt-in warning for real embeddings.

Verification:

```sh
uv run pytest
uv run ruff check .
```

### Task 15: Final Phase 1 verification gate

Goal:
Verify Phase 1 end to end with tests, linting, ingestion, and retrieval smoke testing.

Files touched:
- No new source files expected; fix only issues found by verification.

Dependencies:
- Tasks 1 through 14
- Local PDFs in `data/pdfs/`
- OpenAI API key configured for manual real embedding run

Acceptance criteria:
- `uv run pytest` passes.
- `uv run ruff check .` passes.
- `uv run python scripts/ingest_pdfs.py` loads all PDFs from `data/pdfs/`.
- Vectors persist under `storage/chroma`.
- Chroma collection name defaults to `human_design`.
- Embedding model defaults to `text-embedding-3-small`.
- Query script reloads the existing Chroma store without re-ingesting.
- `uv run python scripts/query_kb.py "What is a Generator type?"` returns top-k chunks with source metadata.
- Real embedding calls are manual or opt-in.
- `.env`, PDFs, and Chroma storage are not committed.

Verification:

```sh
uv run pytest
uv run ruff check .
uv run python scripts/ingest_pdfs.py
uv run python scripts/query_kb.py "What is a Generator type?"
git status --short
```

Manual note: the ingestion and query commands require local PDFs and intentional real OpenAI embedding use.

## Final Verification Gate

Run:

```sh
uv run pytest
uv run ruff check .

# Manual only: requires local PDFs and intentional real OpenAI embedding use.
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py

# Manual only: requires an existing Chroma store and OpenAI embedding access for the query.
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/query_kb.py "What is a Generator type?"
```

Clarifications:

- `uv run python scripts/ingest_pdfs.py` requires local PDFs in `data/pdfs/`.
- Real OpenAI embedding calls should be manual or opt-in.
- Default tests must not require OpenAI API calls.
- `uv run python scripts/query_kb.py "What is a Generator type?"` requires an existing Chroma store.

## Future Phases

- Phase 2: Vision extraction for BodyGraph/chart images.
- Phase 3: RAG answer generation / reading generation.
- Phase 4: web app or API.
- Phase 5: cloud deployment if needed.

Do not include implementation tasks for future-phase features in Phase 1.
