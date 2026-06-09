---
name: human-design-rag-module-conventions
description: |
  Use when creating or modifying Python modules under src/human_design/rag/.
  Core modules must stay interface-agnostic and local-first. Use this when deciding
  file layout, dataclasses, imports, error handling, and module boundaries.
---

# Human Design RAG module conventions

## Purpose

Use this skill when creating or modifying Python modules for the Human Design RAG project.

The goal is to keep core RAG logic clean, testable, local-first, and reusable by future interfaces.

## Expected layout

<repo-root>/
├── scripts/
│   ├── ingest_pdfs.py
│   └── query_kb.py
└── src/
    └── human_design/
        ├── __init__.py
        └── rag/
            ├── __init__.py
            ├── config.py
            ├── models.py
            ├── ingestion.py
            ├── vector_store.py
            └── retriever.py

## Core principle

Core modules under src/human_design/rag/ must be reusable and interface-agnostic.

Core modules must NOT import:
- streamlit
- argparse
- rich
- boto3
- AWS SDKs
- FastAPI
- LangChain
- UI code

Scripts may parse CLI arguments and call core functions.
Core modules should not parse CLI arguments or print user-facing output.

## Module responsibilities

### config.py

Responsible for:
- loading environment variables
- defining app configuration
- resolving local paths
- storing chunk size and chunk overlap
- storing Chroma collection name
- storing embedding model name
- storing ingestion version

Do not hardcode these settings throughout the codebase.

### models.py

Responsible for dataclasses and typed result objects.

Use dataclasses for structured results such as:
- ingestion result
- PDF load result
- chunking result
- vector store result
- retrieval result

If a function returns more than two related values, prefer a dataclass over a tuple or loose dict.

### ingestion.py

Responsible for:
- discovering local PDFs
- loading PDF documents with PyMuPDFReader
- text-only extraction behavior
- attaching metadata
- chunking with SentenceSplitter
- returning chunks or nodes for storage

Avoid putting Chroma-specific persistence details here.

### vector_store.py

Responsible for:
- Chroma persistent client setup
- Chroma collection setup
- LlamaIndex vector store integration
- storing nodes or chunks
- loading an existing Chroma-backed index

### retriever.py

Responsible for:
- loading the existing vector store
- running top-k retrieval
- returning typed retrieval results
- preserving source metadata in retrieval output

## Path handling

Use pathlib.Path for path arguments inside core modules.

Good example:

    def load_pdfs(pdf_dir: Path) -> list[Document]:
        ...

Avoid:

    def load_pdfs(pdf_dir: str) -> list:
        ...

## Type hints

Use type hints on all public functions.

Use Python 3.10+ union syntax.

Good example:

    def get_config(env_file: Path | None = None) -> AppConfig:
        ...

Avoid untyped public functions.

## Error handling boundary

Core functions should raise exceptions or return typed result objects.
Scripts should catch exceptions and format user-facing messages.

Core example:

    def load_documents(pdf_dir: Path) -> list[Document]:
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

Script example:

    try:
        result = run_ingestion(config)
    except FileNotFoundError as exc:
        print(f"Cannot ingest PDFs: {exc}")
        raise SystemExit(1)

## Public vs private helpers

Public API example:

    def ingest_pdfs(config: AppConfig) -> IngestionResult:
        ...

Private helper example:

    def _attach_metadata(...):
        ...

## File and function size

- Module greater than 300 lines: consider splitting it.
- Function greater than 50 lines: consider extracting helpers.

## Do not add future-phase dependencies

For Phase 1, do not add:
- boto3
- streamlit
- fastapi
- langchain
- pinecone
- sqlalchemy
- llama-parse
