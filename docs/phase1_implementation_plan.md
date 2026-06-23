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
- `tests/test_models.py`

Dependencies:
- `pytest`

Acceptance criteria:
- Dataclasses exist for:
  - `PdfLoadResult`
  - `ChunkingResult`
  - `IngestionResult`
  - `VectorStoreResult`
  - `RetrievalResult`
- Public result types expose counts and source metadata without requiring loose dicts.
- Dataclasses use type hints.
- Dataclasses use `pathlib.Path` for filesystem paths where appropriate.
- Retrieval result includes:
  - text snippet or text
  - score
  - source_file
  - page_label or page_number if available
  - metadata
- No PDF ingestion logic is implemented.
- No Chroma logic is implemented.
- No OpenAI API calls are made.
- No `data/pdfs/` or `storage/` directories are created.

Verification:

```sh
uv run pytest tests/test_models.py
uv run ruff check .
git status --short
```

### Task 5: PDF discovery and PyMuPDFReader loading

Goal:
Discover local PDF files and load them through LlamaIndex `PyMuPDFReader`.

Files touched:
- `src/human_design/rag/ingestion.py`
- `tests/test_ingestion.py`

Dependencies:
- `llama-index-readers-file`
- `pymupdf`
- `pytest`

Acceptance criteria:
- PDF discovery reads all `*.pdf` files from `data/pdfs/` or a configured PDF directory.
- PDF discovery returns deterministic results, such as sorted paths.
- Ingestion fails with a clear exception if no PDFs exist.
- PDFs are loaded with LlamaIndex `PyMuPDFReader`.
- Extracted documents preserve source metadata, including at least source path and file name.
- Page-level metadata is preserved if provided by `PyMuPDFReader`.
- Core ingestion functions raise exceptions or return typed result objects; scripts handle user-facing messages later.
- A future `scripts/ingest_pdfs.py` script can call these functions to load all PDFs from `data/pdfs/`.

Verification:

```sh
uv run pytest tests/test_ingestion.py
uv run ruff check .
```

### Task 6: Text-only extraction report

Goal:
Track embedded text extraction results from locally loaded PDFs without adding OCR, vision, LlamaParse, or image parsing.

Files touched:
- `src/human_design/rag/ingestion.py`
- `src/human_design/rag/models.py`
- `tests/test_ingestion.py`

Dependencies:
- `llama-index-readers-file`
- `pytest`

Acceptance criteria:
- Add a typed dataclass in `models.py` for the extraction report.
- The report includes:
  - number of PDFs discovered
  - number of documents/pages loaded
  - source PDF files
  - total extracted character count
  - low-text document/page count
  - low-text document/page references when practical
- Low-text pages are detected using a configurable character-count threshold.
- Low-text references include source metadata such as `file_name`, `source_path`, and page metadata if available.
- The report is generated from already-loaded LlamaIndex `Document` objects or from `load_pdfs()`.
- Core functions return typed result objects and do not print or call `sys.exit`.
- OCR, LlamaParse, BodyGraph image interpretation, and vision extraction are not added.
- No raw full-text dumps are written.
- No chunking, embeddings, Chroma, retrieval, or OpenAI calls are added.

Verification:

```sh
uv run pytest tests/test_ingestion.py
uv run ruff check .
```

### Task 7: Chunking with SentenceSplitter(chunk_size=800, chunk_overlap=80)

Goal:
Chunk already-loaded LlamaIndex `Document` objects with fixed Phase 1 settings and preserve useful source metadata on each chunk/node.

Files touched:
- `src/human_design/rag/chunking.py`
- `src/human_design/rag/models.py`
- `tests/test_chunking.py`

Dependencies:
- `llama-index`
- `pytest`

Acceptance criteria:
- Add a new `src/human_design/rag/chunking.py` module.
- Add a core function such as `chunk_documents(documents: list[Document])`.
- The core chunking function accepts already-loaded in-memory documents.
- The core chunking function does not read real PDFs.
- The core chunking function does not call `load_pdf()` or `load_pdfs()`.
- `SentenceSplitter` uses `chunk_size=800`.
- `SentenceSplitter` uses `chunk_overlap=80`.
- Chunks/nodes are created from the input documents.
- Each chunk/node includes metadata:
  - `chunk_size=800`
  - `chunk_overlap=80`
  - `source_file`, derived from `file_name` when available
  - `source_path` when available
  - `page_label` or `page_number` when available
  - `document_title` when available
  - `embedding_model` as a metadata string only
  - `ingestion_version`
- If `page_label` or `page_number` is missing but PyMuPDFReader metadata has `source`, use `source` as fallback when practical.
- Add or update `ChunkingResult` if needed to report:
  - `document_count`
  - `chunk_count`
  - `chunk_size`
  - `chunk_overlap`
- Tests use sample in-memory `Document` objects and do not read real PDFs by default.
- No OpenAI calls, embeddings, Chroma, vector store, retrieval, OCR, LlamaParse, or vision extraction are added.

Verification:

```sh
uv run pytest tests/test_chunking.py tests/test_ingestion.py
uv run ruff check .
```

### Task 8: Chroma persistent vector store setup

Goal:
Create the local persistent Chroma vector store setup for the Phase 1 RAG pipeline.

This task only creates and verifies the Chroma persistent client, collection, and LlamaIndex ChromaVectorStore wrapper. It does not embed real PDFs, does not call OpenAI, and does not persist Task 7 chunks yet.

Purpose:
The Chroma database created in this task will later store embeddings generated from Task 7 chunks. The actual embedding and node persistence step will be handled in a later task.

Files touched:
- `src/human_design/rag/vector_store.py`
- `tests/test_vector_store.py`

Dependencies:
- `chromadb`
- `llama-index-vector-stores-chroma`
- `pytest`

Acceptance criteria:
- A persistent Chroma client uses the default Chroma directory from config in production code, but tests must pass an explicit tmp_path or run from tmp_path so repo-level storage/chroma is not created.
- The default Chroma directory is `storage/chroma`.
- The default collection name is `human_design`.
- A configured Chroma directory and collection name can be passed in explicitly.
- Existing collections can be reopened without deleting existing data.
- A LlamaIndex `ChromaVectorStore` wrapper can be created from the Chroma collection.
- Tests use `tmp_path` and never create repo-level `storage/`.
- Tests may add small dummy vectors directly to Chroma to verify persistence.
- Tests must not use real PDFs, Task 7 chunks, OpenAI embeddings, or API keys.

Out of scope:
- Do not load PDFs.
- Do not call `chunk_documents()`.
- Do not call OpenAI.
- Do not create embeddings.
- Do not persist real Human Design chunks.
- Do not create `embeddings.py`.
- Do not create `retriever.py`.
- Do not delete existing Chroma collections.
- Do not create repo-level `storage/chroma` during tests.

Expected implementation:
- Add `src/human_design/rag/vector_store.py`.
- Provide a function to create a persistent Chroma client.
- Provide a function to get or create a Chroma collection.
- Provide a function to create a LlamaIndex `ChromaVectorStore`.
- Use config defaults for `DEFAULT_CHROMA_DIR` and `DEFAULT_COLLECTION_NAME`.

Suggested function names:
- `create_chroma_client(chroma_dir: Path = DEFAULT_CHROMA_DIR)`
- `get_or_create_chroma_collection(client, collection_name: str = DEFAULT_COLLECTION_NAME)`
- `create_chroma_vector_store(chroma_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str = DEFAULT_COLLECTION_NAME)`

Verification:

```sh
uv run pytest tests/test_vector_store.py
uv run ruff check .
test ! -d storage/chroma
git status --short
```

### Task 9: OpenAI embedding factory and embedding model consistency

Goal:
Add OpenAI embedding factory wiring for `OpenAIEmbedding(text-embedding-3-small)` while keeping default tests free and offline.

This task prepares the embedding layer used later to embed Task 7 chunks. It does not embed real PDFs, does not embed real Task 7 chunks, and does not persist nodes into Chroma yet.

Purpose:
Task 9 creates a safe embedding factory and adds embedding model metadata consistency checks for Chroma collections. A later task will use this embedding factory to embed Task 7 chunks and persist them into the Chroma vector store.

Files touched:
- `src/human_design/rag/embeddings.py`
- `src/human_design/rag/vector_store.py`
- `src/human_design/rag/config.py`
- `tests/test_embeddings.py`
- `tests/test_vector_store.py`

Possible dependency files if missing:
- `pyproject.toml`
- `uv.lock`

Dependencies:
- `llama-index-embeddings-openai`
- `pytest`

Acceptance criteria:
- Embedding model defaults to `text-embedding-3-small`.
- Embedding model can be overridden through config using `HD_RAG_EMBED_MODEL`.
- OpenAI API key can come from environment variables or `.env`.
- OpenAI API key is never printed, logged, or committed.
- Default tests must not call the OpenAI API.
- Default tests must not require a real `OPENAI_API_KEY`.
- Real embedding calls are manual or opt-in only.
- A function exists to create an `OpenAIEmbedding` instance from config values.
- Chroma collection metadata records the embedding model used for that collection.
- Existing Chroma collections are checked for embedding model consistency before appending or querying.
- If an existing non-empty collection has a different embedding model, raise a clear `ValueError`.
- Tests use fake or monkeypatched embedding classes by default.
- Tests do not embed real PDFs or Task 7 chunks.

Out of scope:
- Do not load PDFs.
- Do not call `load_pdfs()`.
- Do not call `chunk_documents()`.
- Do not embed real Task 7 chunks.
- Do not persist nodes into Chroma.
- Do not call OpenAI in default tests.
- Do not create retriever logic.
- Do not query the RAG system.
- Do not create repo-level `storage/chroma` during tests.

Expected implementation:

1. Add `src/human_design/rag/embeddings.py`.

Suggested functions:
- `create_openai_embedding_model(embedding_model: str, api_key: str | None = None)`
- or `create_openai_embedding_model_from_config(config: AppConfig)`

The function should return:
- `llama_index.embeddings.openai.OpenAIEmbedding`

using:
- model from config
- API key from config/env when provided

2. Update `src/human_design/rag/config.py`.

Add support for:
- `OPENAI_API_KEY`

Suggested config additions:
- `ENV_OPENAI_API_KEY = "OPENAI_API_KEY"`
- `openai_api_key: str | None` on `AppConfig`

Important:
- Do not print or log the API key.
- Do not add a default fake API key.
- Missing API key is acceptable for default tests.

3. Update `src/human_design/rag/vector_store.py`.

Current Task 8 vector store setup already creates:
- persistent Chroma client
- collection
- LlamaIndex `ChromaVectorStore`

Keep that behavior.

Add embedding model metadata consistency support.

Suggested constant:
- `EMBEDDING_MODEL_METADATA_KEY = "embedding_model"`

Suggested functions:
- `ensure_collection_embedding_model(collection, embedding_model: str) -> None`
- `set_collection_embedding_model_if_empty(collection, embedding_model: str) -> None`

Behavior:
- If collection metadata has the same embedding model, pass.
- If collection metadata has a different embedding model, raise `ValueError`.
- If collection metadata is missing and collection is empty, set the embedding model metadata.
- If collection metadata is missing and collection is non-empty, raise `ValueError` because the existing vectors' model is unknown.
- Never delete or reset an existing collection.

4. Tests.

Default tests must:
- monkeypatch/fake `OpenAIEmbedding`
- verify embedding model default
- verify embedding model override
- verify API key can be read from env or `.env`
- verify no OpenAI API call is made
- verify collection metadata is set for empty collections
- verify matching embedding model passes
- verify mismatched embedding model raises `ValueError`
- verify missing embedding metadata on a non-empty collection raises `ValueError`
- use `tmp_path` for Chroma
- not create repo-level `storage/chroma`

Manual / opt-in tests:
- If adding a real embedding test, mark it as skipped unless an explicit env var is set, for example:
  `RUN_OPENAI_EMBEDDING_TEST=1`

Verification:

```sh
uv run pytest tests/test_embeddings.py tests/test_vector_store.py
uv run ruff check .
test ! -d storage/chroma
git status --short
```

### Task 10: End-to-end ingestion script

Goal:
Create a CLI script that runs the local ingestion pipeline from configured PDFs to persistent Chroma.

This task connects the completed Phase 1 components:

- Task 3: Config loading
- Task 5: PDF discovery and loading
- Task 6: Text extraction report
- Task 7: Chunking
- Task 8: Chroma persistent vector store setup
- Task 9: OpenAI embedding factory and embedding model consistency check

Purpose:
When explicitly enabled, the script loads local PDFs, chunks them, creates OpenAI embeddings, and persists the resulting vectors into the configured Chroma collection.

Real OpenAI embedding calls must be manual / opt-in only.

---

Files touched

- `scripts/ingest_pdfs.py`
- `tests/test_ingest_pdfs.py`

Possible files only if needed:

- `src/human_design/rag/models.py`
- `src/human_design/rag/vector_store.py`
- `tests/test_vector_store.py`
- `tests/test_ingestion.py`

Do not modify unless necessary:

- `src/human_design/rag/ingestion.py`
- `src/human_design/rag/chunking.py`
- `src/human_design/rag/embeddings.py`
- `src/human_design/rag/config.py`

---

Dependencies

- Tasks 3 through 9
- Existing LlamaIndex dependencies

No new dependency should be added unless absolutely required.

---

Acceptance criteria

- `scripts/ingest_pdfs.py` exists.
- The script has a `main()` function.
- The script loads configuration using `load_config()`.
- The script uses configured:
  - PDF directory
  - Chroma directory
  - Chroma collection name
  - chunk size
  - chunk overlap
  - embedding model
  - ingestion version
  - OpenAI API key
- The script discovers and loads PDFs from the configured PDF directory.
- The script builds a text extraction report.
- The script chunks loaded documents using Task 7 `chunk_documents()`.
- The script creates an OpenAI embedding model using Task 9 embedding factory.
- The script creates a Chroma vector store using Task 8 `create_chroma_vector_store()`.
- The script enforces embedding model consistency before writing into Chroma.
- The script persists embedded nodes into Chroma.
- Ingestion prints a clear report with:
  - PDFs found
  - documents/pages loaded
  - chunks/nodes created
  - chunks persisted
  - collection name
  - Chroma directory
  - embedding model
  - ingestion version
- Ingestion fails with a clear message if no PDFs exist.
- Default tests must not call OpenAI.
- Default tests must not require real PDFs.
- Default tests must not create repo-level `storage/chroma`.
- The script does not dump full PDF text or full chunks.
- Real OpenAI embedding calls only happen when explicitly enabled with `HD_RAG_REAL_EMBEDDINGS=1`.

---

Opt-in behavior

Running without this environment variable must not call OpenAI:

```sh
uv run python scripts/ingest_pdfs.py
```

The script should exit clearly with a message like:

```text
Real embedding ingestion is disabled by default. Set HD_RAG_REAL_EMBEDDINGS=1 to run ingestion.
```

Running with this environment variable may call OpenAI and persist real vectors:

```sh
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py
```

---

Out of scope

- Do not create retriever logic.
- Do not query the RAG system.
- Do not build response synthesis.
- Do not add UI.
- Do not OCR PDFs.
- Do not parse BodyGraph images.
- Do not dump full extracted text or full chunks to disk.
- Do not delete existing Chroma collections.
- Do not run real OpenAI embedding calls in default tests.
- Do not persist real Human Design chunks in tests.

---

Suggested implementation

In `scripts/ingest_pdfs.py`, add a `main()` function.

Guard real embedding ingestion with:

```python
import os

if os.environ.get("HD_RAG_REAL_EMBEDDINGS") != "1":
    raise SystemExit(
        "Real embedding ingestion is disabled by default. "
        "Set HD_RAG_REAL_EMBEDDINGS=1 to run ingestion."
    )
```

Use the existing components:

```python
from llama_index.core import StorageContext, VectorStoreIndex

from human_design.rag.chunking import chunk_documents
from human_design.rag.config import load_config
from human_design.rag.embeddings import create_openai_embedding_model_from_config
from human_design.rag.ingestion import build_text_extraction_report, load_pdfs
from human_design.rag.vector_store import create_chroma_vector_store
```

Suggested flow:

```python
config = load_config()

documents = load_pdfs(config.pdf_dir)

extraction_report = build_text_extraction_report(config.pdf_dir)

nodes, chunk_result = chunk_documents(
    documents,
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    embedding_model=config.embedding_model,
    ingestion_version=config.ingestion_version,
)

embed_model = create_openai_embedding_model_from_config(config)

vector_store = create_chroma_vector_store(
    chroma_dir=config.chroma_dir,
    collection_name=config.collection_name,
    embedding_model=config.embedding_model,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)
```

The persisted count can be reported as:

```python
persisted_count = len(nodes)
```

Do not print full document text or full chunk text.

---

Testing requirements

Add tests in:

- `tests/test_ingest_pdfs.py`

Default tests must monkeypatch or fake:

- PDF loading
- text extraction report
- chunking
- embedding factory
- Chroma vector store creation
- LlamaIndex indexing call

Tests should verify:

- The script exits clearly when `HD_RAG_REAL_EMBEDDINGS` is missing.
- The script does not call OpenAI by default.
- The script does not require real PDFs.
- The script does not create repo-level `storage/chroma`.
- The script calls the pipeline components in the expected order when `HD_RAG_REAL_EMBEDDINGS=1`.
- The script prints a clear summary report.
- The script does not print full PDF text or full chunks.
- The script handles no-PDF errors with a clear message.

Tests must not:

- use real PDFs
- call OpenAI
- require `OPENAI_API_KEY`
- persist real Human Design chunks
- create retriever logic
- create repo-level `storage/chroma`

---

Verification

Default verification:

```sh
uv run pytest tests/test_ingest_pdfs.py tests/test_ingestion.py tests/test_vector_store.py
uv run ruff check .
test ! -d storage/chroma
git status --short
```

Manual verification only when local PDFs exist and real OpenAI embedding calls are intentionally allowed:

```sh
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py
```
Manual note: run the final command only when local PDFs exist and real OpenAI embedding calls are intentionally allowed.

### Task 11: Reload existing Chroma index without re-ingesting

Goal:
Support loading an existing Chroma-backed LlamaIndex index for retrieval without re-ingesting PDFs.

Purpose:
Task 10 creates and persists embeddings into Chroma. Task 11 should reload that existing Chroma collection and expose a retriever/query path without loading PDFs, chunking documents, embedding all chunks again, or writing new vectors.

This task is for the retrieval side of the local Phase 1 RAG pipeline.

---

Files touched

- `src/human_design/rag/retriever.py`
- `tests/test_retriever.py`

Possible files only if needed:

- `src/human_design/rag/vector_store.py`
- `tests/test_vector_store.py`

Do not modify unless necessary:

- `src/human_design/rag/ingestion.py`
- `src/human_design/rag/chunking.py`
- `src/human_design/rag/embeddings.py`
- `src/human_design/rag/config.py`
- `scripts/ingest_pdfs.py`

---

Dependencies

- Task 8: Chroma persistent vector store setup
- Task 9: OpenAI embedding factory and embedding model consistency check
- Task 10: End-to-end ingestion script
- Existing LlamaIndex dependencies

No new dependency should be added unless absolutely required.

---

Acceptance criteria

- Existing Chroma store can be reopened from the configured Chroma directory and collection name.
- Existing Chroma-backed LlamaIndex index can be created without re-ingesting PDFs.
- Retriever can be created from the existing Chroma-backed index.
- Query/retrieval path uses the same embedding model recorded during ingestion.
- If the configured embedding model differs from the collection metadata, raise a clear `ValueError`.
- If the collection is non-empty but has no embedding model metadata, raise a clear `ValueError`.
- The user should be told to rebuild the collection if the embedding model is incompatible or unknown.
- Default tests must not call OpenAI.
- Default tests must not require real PDFs.
- Default tests must not create repo-level `storage/chroma`.
- No PDFs are loaded.
- No documents are chunked.
- No new vectors are appended during reload.
- No existing Chroma collection is deleted or reset.

---

Out of scope

- Do not load PDFs.
- Do not call `load_pdfs()`.
- Do not call `load_pdf()`.
- Do not call `build_text_extraction_report()`.
- Do not call `chunk_documents()`.
- Do not embed Task 7 chunks.
- Do not persist nodes into Chroma.
- Do not create or modify `scripts/ingest_pdfs.py`.
- Do not create a query CLI script yet.
- Do not add response generation.
- Do not call an LLM for answer synthesis.
- Do not delete or rebuild Chroma collections.


Testing requirements

Add tests in:

- `tests/test_retriever.py`

Default tests must monkeypatch or fake:

- `create_chroma_vector_store`
- `create_openai_embedding_model_from_config`
- `VectorStoreIndex.from_vector_store`
- `index.as_retriever`

Tests should verify:

- `load_existing_chroma_index()` loads the vector store using config values.
- `load_existing_chroma_index()` creates the embedding model from config.
- `VectorStoreIndex.from_vector_store(...)` is called.
- `create_retriever_from_config()` calls `index.as_retriever(similarity_top_k=...)`.
- Embedding model mismatch from the vector store layer raises `ValueError`.
- No PDF loading is called.
- No chunking is called.
- No OpenAI API call is made in default tests.
- No repo-level `storage/chroma` is created.
- No new vectors are appended.
- No Chroma collection is deleted or reset.

Tests must not:

- use real PDFs
- call OpenAI
- require `OPENAI_API_KEY`
- call `load_pdfs()`
- call `chunk_documents()`
- persist real chunks
- create retriever CLI
- create repo-level `storage/chroma`

---

Verification

Default verification:

```sh
uv run pytest tests/test_vector_store.py tests/test_retriever.py
uv run ruff check .
test ! -d storage/chroma
git status --short
```

### Task 12: Retrieval smoke-test script

Goal:
Create a query script that reloads the existing Chroma store and returns top-k retrieved chunks with source metadata.

Purpose:
Task 12 verifies retrieval after Task 10 ingestion has already created a Chroma collection. It should reload the existing Chroma-backed retriever without re-ingesting PDFs, chunking documents, or writing new vectors.

Files touched:
- `scripts/query_kb.py`
- `src/human_design/rag/retriever.py`
- `tests/test_retriever.py`

Dependencies:
- Tasks 8 through 11

Acceptance criteria:
- `scripts/query_kb.py` accepts a query string from the command line.
- The script supports `--top-k` and `--snippet-chars`.
- The script reloads the existing Chroma store using Task 11 retriever logic.
- The script does not load PDFs, call `chunk_documents()`, or persist new vectors.
- Query path uses the configured embedding model and existing Chroma collection metadata check.
- Output includes top-k results with:
  - text snippet
  - similarity score if available
  - `source_file`
  - `source_path`
  - `page_label` or `page_number` if available
- The script does not dump entire chunks by default.
- If no results are found, print a clear message.
- If the Chroma collection is missing, empty, or has incompatible embedding metadata, print a clear message telling the user to run or rebuild ingestion.
- Default tests must not call OpenAI, require real PDFs, or create repo-level `storage/chroma`.

Out of scope:
- Do not generate final LLM answers.
- Do not call an LLM for synthesis.
- Do not create retriever CLI beyond this smoke-test script.
- Do not delete, reset, or rebuild Chroma collections.

Verification:
```sh
uv run pytest tests/test_retriever.py
uv run ruff check .
test ! -d storage/chroma
```

### Task 13: Tests with mocked or fake embeddings

Goal:
Make default verification free, deterministic, and independent of real PDFs, real embeddings, and repo-level Chroma storage.

Files touched:
- `tests/test_config.py`
- `tests/test_ingestion.py`
- `tests/test_chunking.py`
- `tests/test_embeddings.py`
- `tests/test_vector_store.py`
- `tests/test_retriever.py`
- `tests/test_ingest_pdfs.py`

Dependencies:
- `pytest`

Acceptance criteria:
- Default tests must not call OpenAI API.
- Tests must pass without OPENAI_API_KEY.
- Tests use fake or mocked embeddings where possible.
- Tests use temporary PDF directories and temporary Chroma/vector directories where needed.
- Tests do not require real local PDFs.
- Tests do not require an existing real Chroma collection.
- No Chroma DB files are created under repo-level storage/ during tests.
- Retrieval/query tests must use fake retrievers or monkeypatched retrieval paths.
- Ingestion script tests must keep real embedding ingestion opt-in only.
- Application code should not be modified unless a testability issue requires it.

Verification:

```sh
rm -rf storage/chroma
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
- README explains Phase 1 does not include image parsing, API deployment, or final RAG answer generation yet.
- README documents setup with `uv`.
- README documents `.env.example` and required environment variables.
- README documents where local PDFs live: `data/pdfs/`.
- README documents where local Chroma storage lives: `storage/chroma/`.
- README states default tests do not call OpenAI and do not require `OPENAI_API_KEY`.
- README includes default verification commands.
- README includes real ingestion command with clear manual/opt-in warning.
- README includes retrieval smoke-test command with clear note that it requires an existing Chroma store.
- README includes privacy rules: do not commit PDFs, `.env`, API keys, or local Chroma storage.

Out of scope:
- Do not modify application code.
- Do not run real ingestion.
- Do not call OpenAI.
- Do not create or delete Chroma collections.
- Do not add new dependencies.

Verification:

```sh
uv run pytest
uv run ruff check .
git status --short
```

### Task 15: Final Phase 1 verification gate

Goal:
Verify Phase 1 end to end with tests, linting, manual real ingestion, and retrieval smoke testing.

Files touched:
- No new source files expected.
- Fix only issues found by verification.

Dependencies:
- Tasks 1 through 14
- Local PDFs in `data/pdfs/`
- `.env` configured with `OPENAI_API_KEY`
- Real embedding calls intentionally allowed with `HD_RAG_REAL_EMBEDDINGS=1` for both ingestion and retrieval smoke testing

Acceptance criteria:
- `uv run pytest` passes.
- `uv run ruff check .` passes.
- Manual real ingestion loads all PDFs from `data/pdfs/`.
- Real embedding ingestion is opt-in only with `HD_RAG_REAL_EMBEDDINGS=1`.
- Retrieval smoke testing is also opt-in with `HD_RAG_REAL_EMBEDDINGS=1`, because query embedding may call OpenAI.
- Vectors persist under `storage/chroma`.
- Chroma collection name defaults to `human_design`.
- Embedding model defaults to `text-embedding-3-small`.
- Chroma collection count is greater than 0 after ingestion.
- Query script reloads the existing Chroma store without re-ingesting.
- `HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/query_kb.py "What is a Generator type?"` returns top-k chunks with source metadata.
- Query script returns snippets, score if available, and source/page metadata.
- `.env`, PDFs, and Chroma storage are not committed.

Final Verification Gate:

```sh
uv run pytest
uv run ruff check .

# Manual only: requires local PDFs and intentional real OpenAI embedding use.
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py

# Check persisted Chroma collection.
uv run python - <<'PY'
import chromadb
from human_design.rag.config import load_config

c = load_config()
client = chromadb.PersistentClient(path=str(c.chroma_dir))
collection = client.get_collection(name=c.collection_name)

print("chroma_dir:", c.chroma_dir)
print("collection_name:", c.collection_name)
print("embedding_model:", c.embedding_model)
print("count:", collection.count())
print("metadata:", collection.metadata)

assert c.collection_name == "human_design"
assert c.embedding_model == "text-embedding-3-small"
assert collection.count() > 0
PY

# Manual only: requires an existing Chroma store and OpenAI embedding access for the query.
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/query_kb.py "What is a Generator type?"

git status --short
```

Manual safety notes:
- The ingestion command may call OpenAI and create embedding cost.
- The retrieval smoke test may also call OpenAI for query embedding.
- Do not commit .env, PDFs, or storage/chroma.

### Future Phases

- Phase 2: Vision extraction for BodyGraph/chart images.
- Phase 3: RAG answer generation / reading generation.
- Phase 4: web app or API.
- Phase 5: cloud deployment.

Do not include implementation tasks for future-phase features in Phase 1.
