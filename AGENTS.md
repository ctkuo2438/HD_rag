# AGENTS.md

## Project Overview

This is a Human Design RAG project.

Phase 1 is complete and should be treated as a stable baseline:

```text
Local PDFs -> PyMuPDFReader -> text extraction -> SentenceSplitter -> OpenAI embeddings -> persistent Chroma -> retrieval smoke test
```

The current active work is Phase 2: local BodyGraph Vision extraction.

```text
Local BodyGraph image -> Vision raw fact extraction -> strict JSON parser -> deterministic Python interpreter -> validation -> evaluation / local CLI
```

Phase 1 code should not be changed unless a Phase 2 task explicitly requires a small compatible integration.

## Current Phase: Phase 2

Phase 2 converts a local Human Design BodyGraph or chart image into structured chart data. It extracts raw visible facts, parses them, deterministically derives chart facts, validates consistency, evaluates against golden labels, and supports local CLI smoke testing.

Phase 2 does not generate Human Design readings.

In scope:
- Local BodyGraph or Human Design chart images
- Vision API extraction of raw visible facts only
- Strict JSON response parsing
- Planetary activation extraction from Personality and Design columns
- Canonical center, gate, and channel normalization
- Deterministic derivation of:
  - active gates
  - active channels
  - derived defined centers
  - profile
  - type
  - authority
  - definition
  - strategy
  - not-self theme
  - signature
- Structured validation warnings
- Golden-label evaluation
- Mocked/offline tests
- Local CLI smoke testing
- Real Vision API calls only through explicit opt-in

Out of scope:
- Human Design reading generation
- RAG answer generation
- Calling Phase 1 retrieval to generate readings
- Streamlit
- FastAPI
- Web app
- API deployment
- User accounts
- AWS
- S3
- Lambda
- EC2
- SageMaker
- Databases
- SQL database
- Cloud deployment
- OCR for arbitrary PDFs
- LlamaParse
- Training computer vision models
- YOLO or object-detection training
- Manual annotation UI
- Astrology calculation from birth date, time, or location
- Provider abstraction unless explicitly requested
- New external dependencies unless specifically required by the Phase 2 plan
- LangChain
- Pinecone
- OpenSearch

## Phase 2 Architecture Decisions

1. The Vision model extracts only raw visible facts.
2. The Vision model must not be trusted to directly determine:
   - type
   - authority
   - profile
   - definition
   - strategy
   - not-self theme
   - signature
3. Deterministic Python code derives all final chart concepts.
4. Personality and Design activation columns are the primary source of truth for active gates.
5. Active channels are derived only from canonical Human Design channels whose two gate endpoints are active.
6. Derived defined centers come from endpoints of deterministic active channels.
7. Visual centers, visible gates, and visible colored channels are supporting evidence for validation only.
8. Parser failures and validation warnings are distinct:
   - malformed JSON or missing required structural fields are parser errors
   - recoverable Vision uncertainty, invalid observed items, or visual versus derived disagreement become structured validation warnings
9. Phase 2 must not alter Phase 1 PDF ingestion, Chroma persistence, or retrieval architecture.

## Module Boundaries

Core modules under `src/human_design/` must stay local-first, typed, focused, and interface-agnostic.

Core modules should raise exceptions or return typed results. User-facing messages, CLI arguments, printing, and exit behavior belong in `scripts/`.

Use `pathlib.Path` for filesystem paths. Use dataclasses or typed schema objects for structured results.

### Phase 1 Baseline Modules

Existing `src/human_design/rag/` modules remain responsible for local PDF ingestion, chunking, embeddings, Chroma vector storage, and retrieval smoke testing. Do not refactor or broaden them during Phase 2 unless a task explicitly requires a small compatible integration.

### Phase 2 Module Responsibilities

- `vision/config.py`
  - Loads Phase 2 environment values only.
  - Must remain separate from Phase 1 RAG configuration unless a shared setting is genuinely needed.
- `vision/models.py`
  - Contains typed schemas for raw Vision facts, parsed activations, derived chart data, warnings, validation, evaluation results, and full extraction output.
  - Must not import Vision clients.
- `vision/constants.py`
  - Contains canonical centers, aliases, all 36 valid channels, channel-to-center mappings, and motor centers.
- `vision/prompt.py`
  - Loads or formats the raw fact extraction prompt.
  - Must not derive chart concepts.
- `vision/client.py`
  - Handles the one approved Vision API provider.
  - Must never be called by default tests.
  - Must never log keys or base64 image payloads.
- `vision/parser.py`
  - Parses strict JSON.
  - Normalizes channels, center aliases, and activation values.
  - Preserves recoverable extraction issues for validation.
- `vision/interpreter.py`
  - Performs only deterministic Python derivation.
  - Must not call a Vision API or LLM.
- `vision/validation.py`
  - Compares raw visual evidence against derived chart data.
  - Emits machine-readable warning codes.
- `vision/evaluation.py`
  - Compares predictions with golden labels using explicit metrics.
- `scripts/extract_bodygraph.py`
  - Handles CLI arguments and user-facing output.
  - Must not contain core derivation logic.
- `scripts/evaluate_bodygraph_extraction.py`
  - Handles local evaluation CLI behavior only.

## Expected Layout

This layout includes existing Phase 1 files and expected Phase 2 files. Not every Phase 2 file necessarily exists yet.

```text
<repo-root>/
├── data/
│   ├── pdfs/
│   └── bodygraph_samples/
│       ├── images/
│       ├── private/
│       └── golden_labels.example.json
├── prompts/
│   └── bodygraph_raw_extraction.txt
├── scripts/
│   ├── ingest_pdfs.py
│   ├── query_kb.py
│   ├── extract_bodygraph.py
│   └── evaluate_bodygraph_extraction.py
├── src/
│   └── human_design/
│       ├── rag/
│       │   ├── config.py
│       │   ├── models.py
│       │   ├── ingestion.py
│       │   ├── chunking.py
│       │   ├── embeddings.py
│       │   ├── vector_store.py
│       │   └── retriever.py
│       └── vision/
│           ├── __init__.py
│           ├── config.py
│           ├── models.py
│           ├── constants.py
│           ├── prompt.py
│           ├── client.py
│           ├── parser.py
│           ├── interpreter.py
│           ├── validation.py
│           └── evaluation.py
├── tests/
│   ├── fixtures/bodygraph/
│   ├── test_vision_config.py
│   ├── test_vision_models.py
│   ├── test_bodygraph_constants.py
│   ├── test_bodygraph_parser.py
│   ├── test_bodygraph_interpreter.py
│   ├── test_bodygraph_validation.py
│   ├── test_bodygraph_evaluation.py
│   └── test_extract_bodygraph.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Privacy and Cost Rules

Never commit:
- `.env`
- API keys
- `data/pdfs/`
- `storage/`
- Chroma DB files
- local embedding caches
- raw extracted full-text dumps from copyrighted PDFs
- `data/bodygraph_samples/images/`
- `data/bodygraph_samples/private/`
- private Human Design chart images
- raw private Vision responses
- generated private prediction JSON
- base64 image strings
- `*.vision_response.json` files
- `*.bodygraph_prediction.json` files

Default tests must:
- not call OpenAI APIs
- not call a Vision API
- not require `OPENAI_API_KEY`
- not require private BodyGraph images
- not require Phase 1 Chroma storage
- use mocked embeddings or mocked Vision responses where applicable
- use sanitized fixtures

Real embedding ingestion remains opt-in with:

```text
HD_RAG_REAL_EMBEDDINGS=1
```

Real Vision calls must require explicit opt-in with:

```text
HD_VISION_REAL_API=1
```

Never print or log:
- API keys
- full PDF text
- full chunks
- full prompts
- full retrieved passages
- full base64 images
- private raw Vision output
- private chart image content

Short structured results, warning codes, metadata, and safe snippets are allowed when appropriate.

## Environment Variables

Phase 1 variables:

```env
OPENAI_API_KEY=
HD_RAG_PDF_DIR=data/pdfs
HD_RAG_CHROMA_DIR=storage/chroma
HD_RAG_COLLECTION=human_design
HD_RAG_EMBED_MODEL=text-embedding-3-small
```

Phase 2 variables:

```env
HD_VISION_MODEL=
HD_VISION_REAL_API=0
HD_BODYGRAPH_SAMPLE_DIR=data/bodygraph_samples/images
HD_BODYGRAPH_GOLDEN_LABELS=data/bodygraph_samples/golden_labels.example.json
OPENAI_API_KEY=
```

`HD_VISION_REAL_API=0` is the safe default. Real Vision API use requires `HD_VISION_REAL_API=1`. Tests must run without an API key.

Do not modify `.env.example` unless a task explicitly requires it.

## Testing and Verification

Use uv.

Default commands:

```sh
uv run pytest
uv run ruff check .
git diff --check
```

Testing rules:
- Tests should be written before or alongside behavior changes.
- Parser, interpreter, validation, and evaluation should be independently unit-tested.
- Phase 2 tests must remain offline by default.
- Mock CLI extraction should work with a committed synthetic or non-private fixture image and mocked JSON response.
- Phase 1 tests must continue to pass.
- Real Vision tests are manual and opt-in only.
- Do not add executable commands that call a paid API by default.

## Workflow Rules

- Read `AGENTS.md` before implementation.
- Read the active phase implementation plan before changing code.
- Keep Phase 2 tasks narrow and incremental.
- Use TDD where practical.
- Run focused tests first, then the full suite and Ruff.
- Do not change Phase 1 behavior unless required and explicitly justified.
- Do not broaden scope without explicit approval.
- Prefer small, reviewable changes.
- Report changed files, tests, linting, and any opt-in API behavior after each task.
- Do not make commits unless explicitly requested.