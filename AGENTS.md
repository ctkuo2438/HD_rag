# AGENTS.md

## Project Overview

This is a Human Design RAG project. The current work is Phase 1: local text-only knowledge base construction from Human Design PDF books.

## Current Phase: Phase 1

Phase 1 is local-only and text-only.

In scope:
- Local PDFs
- LlamaIndex PyMuPDFReader
- Embedded PDF text extraction only
- SentenceSplitter(chunk_size=800, chunk_overlap=80)
- OpenAIEmbedding(text-embedding-3-small)
- Chroma persistent local vector store
- Ingestion script
- Retrieval smoke-test script
- Tests with mocked/fake embeddings where possible

Out of scope:
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
- User accounts
- Streamlit
- FastAPI
- Web app
- Claude reading generation
- Vision extraction
- BodyGraph image interpretation
- OCR
- LlamaParse
- Tree Index
- SQL agent
- RAG answer generation

## Architecture Decision

Use this fixed Phase 1 pipeline:

Local PDFs → LlamaIndex PyMuPDFReader → text-only extraction → SentenceSplitter(chunk_size=800, chunk_overlap=80) → OpenAIEmbedding(text-embedding-3-small) → Chroma persistent vector store.

Do not change this architecture unless the user explicitly asks.

## Implementation Rules

- Keep Phase 1 local-only.
- Do not add AWS services.
- Do not add Pinecone.
- Do not add LangChain.
- Do not add OCR or LlamaParse.
- Do not parse or interpret images inside PDFs.
- Do not implement Phase 2 features.
- Use LlamaIndex + Chroma only for Phase 1.
- Keep core modules interface-agnostic.
- Scripts may parse CLI args; core modules should not.
- Use pathlib.Path for file paths.
- Use type hints on public functions.
- Use dataclasses for structured results.

## Expected Future Layout

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
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Privacy and Cost Rules

Never commit:
- `.env`
- `data/pdfs/`
- `storage/`
- Chroma DB files
- local embedding caches
- API keys
- raw extracted full-text dumps from copyrighted PDFs

Default tests must not call OpenAI API.
Use mocked/fake embeddings for tests.
Real embedding calls must be manual or opt-in.

## Environment Variables

Expected `.env.example` values:

```env
OPENAI_API_KEY=
HD_RAG_PDF_DIR=data/pdfs
HD_RAG_CHROMA_DIR=storage/chroma
HD_RAG_COLLECTION=human_design
HD_RAG_EMBED_MODEL=text-embedding-3-small
```

## Testing and Verification

Use uv.

Expected setup commands later:
```sh
uv init
uv venv
source .venv/bin/activate
uv add llama-index llama-index-readers-file llama-index-embeddings-openai llama-index-vector-stores-chroma chromadb pymupdf python-dotenv
uv add --dev pytest ruff
```

Expected verification commands later:
```sh
uv run pytest
uv run ruff check .
```

Do not call paid APIs during default tests.

## Workflow Rules

- Plan before implementing.
- Implement incrementally.
- Verify each task before moving to the next.
- Do not broaden scope without explicit user approval.
- Prefer small, reviewable changes.
- Report changed files and verification commands after each task.
