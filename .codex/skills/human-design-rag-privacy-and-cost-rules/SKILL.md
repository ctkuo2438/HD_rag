---
name: human-design-rag-privacy-and-cost-rules
description: |
  Use when handling environment variables, API keys, local PDFs, Chroma storage,
  BodyGraph images, Vision requests or responses, logs, tests, CLI output, or
  generated predictions. Default tests must not call paid embedding or Vision APIs.
---

# Human Design RAG Privacy and Cost Rules

## Purpose

Use this skill when handling:

- `.env` files
- API keys
- local PDF books
- Chroma storage
- embeddings
- BodyGraph images
- Vision API requests and responses
- logs
- tests
- CLI output
- golden labels
- saved predictions

The goals are to:

- protect private user data
- avoid leaking copyrighted PDF content
- avoid leaking API keys
- avoid committing private BodyGraph images or raw Vision responses
- avoid accidental paid API usage
- keep default tests fully offline

## Phase Coverage

This skill applies to both project phases.

Phase 1:

    Local PDFs
    -> text extraction
    -> embeddings
    -> persistent Chroma
    -> retrieval

Phase 2:

    Local BodyGraph image
    -> Vision raw fact extraction
    -> parser
    -> deterministic interpreter
    -> validation
    -> evaluation and local CLI

## Never Commit Private or Generated Files

Never commit:

- `.env`
- API keys
- `data/pdfs/`
- `storage/`
- Chroma database files
- local embedding caches
- raw extracted full-text dumps from copyrighted PDFs
- private BodyGraph images
- private chart screenshots
- raw Vision API responses containing private chart data
- generated prediction files containing private chart data
- base64 image payloads
- full Vision request bodies
- temporary image conversions or image caches
- logs containing private chart data
- debug dumps containing full prompts, raw responses, or API payloads

Examples of Phase 2 files that must not be committed:

    data/bodygraph_samples/images/
    data/bodygraph_samples/private/
    *.vision_response.json
    *.bodygraph_prediction.json
    *.base64
    *.image_payload.json

The following may be committed only when non-private and sanitized:

- `data/bodygraph_samples/golden_labels.example.json`
- small synthetic or non-private fixture images under `tests/fixtures/bodygraph/`
- sanitized mock Vision JSON fixtures
- sanitized prediction fixtures
- prompt templates without secrets
- tests that use mocked API responses

## Required .gitignore

`.gitignore` must include at least:

    .venv/
    .env
    __pycache__/
    .pytest_cache/

    data/pdfs/
    storage/

    data/bodygraph_samples/images/
    data/bodygraph_samples/private/

    *.vision_response.json
    *.bodygraph_prediction.json
    *.base64
    *.image_payload.json

Do not ignore these paths unless they contain private data:

    data/bodygraph_samples/golden_labels.example.json
    tests/fixtures/bodygraph/

Private images must never be placed in committed test fixtures.

## Required .env.example

`.env.example` should include Phase 1 settings:

    OPENAI_API_KEY=

    HD_RAG_PDF_DIR=data/pdfs
    HD_RAG_CHROMA_DIR=storage/chroma
    HD_RAG_COLLECTION=human_design
    HD_RAG_EMBED_MODEL=text-embedding-3-small

It should also include Phase 2 settings:

    HD_VISION_MODEL=
    HD_VISION_REAL_API=0
    HD_BODYGRAPH_SAMPLE_DIR=data/bodygraph_samples/images
    HD_BODYGRAPH_GOLDEN_LABELS=data/bodygraph_samples/golden_labels.example.json

Only include this setting when the project explicitly supports provider configuration:

    HD_VISION_PROVIDER=openai

Do not include real secret values, private paths, or personal image filenames in `.env.example`.

## API Key Handling

API keys must come from environment variables or `.env`.

Never:

- hardcode API keys
- print API keys
- include API keys in exceptions
- include API keys in test fixtures
- commit `.env`
- log authorization headers
- log full request payloads
- log environment dictionaries containing secrets

Use generic error messages.

Good example:

    Vision API key is missing. Configure OPENAI_API_KEY before enabling real API mode.

Avoid including a real key value in error output, logs, fixtures, commits, screenshots, or documentation.

## Default Tests Must Be Free

The default test command must not call paid APIs:

    uv run pytest

Default tests must not require:

- API keys
- real embeddings
- real Vision API calls
- private PDFs
- private BodyGraph images
- Chroma storage
- network access

Use:

- mocked Vision client responses
- fake or mocked embeddings
- local JSON fixtures
- synthetic or non-private test images
- in-memory typed models
- temporary local files where needed

Real embedding calls must be explicit and manual:

    HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py

Real Vision API calls must be explicit and manual:

    HD_VISION_REAL_API=1 uv run python scripts/extract_bodygraph.py data/bodygraph_samples/images/example.png

A real Vision call must require both:

1. `HD_VISION_REAL_API=1`
2. a configured API key

Do not allow a real Vision API call merely because an API key exists.

## Vision Request and Response Safety

Vision requests may contain private chart images.

Never:

- print base64 image content
- log base64 image content
- store base64 images by default
- commit raw Vision request payloads
- commit private raw Vision responses
- write raw provider responses to disk by default
- send images to a real API during tests
- send images to a real API without explicit opt-in

The Vision client should return raw response content in memory to the parser.

Persistence of a raw Vision response must be explicit, local-only, and disabled by default.

When manually saving a response for debugging, use a private ignored path and remove it when no longer needed.

## Logging Rules

Never log:

- `OPENAI_API_KEY`
- authorization headers
- full PDF text
- full PDF chunks
- full retrieved passages
- full prompts
- full Vision request bodies
- base64 image data
- private image contents
- full private raw Vision responses
- full private chart outputs
- personal chart activations when logs may be committed or shared

It is acceptable to log:

- safe filename or basename
- file extension
- page number
- image dimensions
- chunk count
- collection name
- embedding model name
- Vision model name
- vector store path
- number of retrieved chunks
- number of parsed activations
- number of active channels
- number of warnings
- warning codes
- whether real API mode is enabled
- a short safe preview of non-private fixture data

Do not treat a local terminal as automatically safe for private content. Prefer counts, warning codes, and metadata over full chart payloads.

## CLI Output Rules

A local CLI may print structured extraction output for a user-supplied local image.

CLI output must never include:

- API keys
- authorization headers
- base64 image data
- full provider request payloads
- hidden environment variables
- private raw response dumps by default

CLI output should clearly distinguish:

- raw Vision extraction
- deterministic derived chart data
- validation warnings

Do not automatically save CLI output to repository-tracked paths.

When saving predictions, require an explicit output path and recommend an ignored local path for private chart data.

## Golden Labels and Fixtures

Golden labels may contain chart data and should be treated carefully.

Allowed committed golden labels must be:

- synthetic
- non-private
- manually reviewed
- limited to what is necessary for tests and evaluation

Do not use a real person's private chart as a committed golden-label fixture without explicit permission and sanitization.

Use:

    data/bodygraph_samples/golden_labels.example.json

for a non-private example file.

Use:

    tests/fixtures/bodygraph/

for small sanitized offline fixtures.

## Embedding Model Consistency

The same embedding model must be used for ingestion and query.

If the embedding model changes:

- do not query an old collection with the new model
- rebuild the Chroma collection
- update metadata
- update ingestion version

Do not silently mix embeddings from different models in one collection.

## Cost Awareness

OpenAI embeddings and Vision API calls may incur cost.

Avoid:

- repeated ingestion when no source PDF changed
- repeated real Vision extraction during development
- real Vision calls inside tests
- real Vision calls inside evaluation by default
- retry loops without limits
- multiple provider calls for one chart unless explicitly requested
- automatic fallback to another paid provider

Use mocked fixtures to develop:

- parser logic
- interpreter logic
- validation logic
- evaluation logic
- CLI mock mode
- error handling
- configuration behavior

Use real API calls only for intentional manual smoke tests after offline tests pass.

## Before Committing

Before committing, verify:

    git status --short
    git diff --check
    git diff -- .gitignore .env.example

Check specifically that no staged file contains:

- `.env`
- API keys
- private PDFs
- Chroma storage
- private BodyGraph images
- raw Vision responses
- generated private predictions
- base64 payloads
- copyrighted full-text dumps

When uncertain whether a file contains private chart data, do not commit it.