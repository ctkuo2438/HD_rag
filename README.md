# Human Design RAG

Phase 1 is the stable baseline: a local, text-only Human Design knowledge base built from local PDF books. Phase 2 adds a separate local BodyGraph Vision extraction pipeline documented below; it does not change Phase 1 ingestion or retrieval behavior.

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

Supported Phase 1 environment variables:

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
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/query_kb.py "What is a Generator type?"
```

This is manual and opt-in because it may call OpenAI for the query embedding. It reloads the existing Chroma collection, retrieves top matching chunks with source metadata, and does not generate final LLM answers.

## Phase 2: Local BodyGraph Vision Extraction

Phase 2 extracts structured chart data from one local Human Design BodyGraph image. It is separate from Phase 1 and does not change PDF ingestion, Chroma storage, or retrieval behavior.

Phase 2 does not generate a Human Design reading, call RAG retrieval, ingest PDFs, calculate a chart from birth data, provide a web app, or train a computer vision model.

```text
Local BodyGraph image
  -> Vision raw extraction
  -> strict JSON parser
  -> deterministic interpreter
  -> validation
  -> evaluation or CLI output
```

The Vision model extracts raw visible facts only: Personality and Design activations, visually defined centers, visual gates, visual channels, and `uncertain_items`. Each uncertain item carries its own numeric confidence. The model must not directly infer type, authority, profile, strategy, definition, not-self theme, or signature.

The deterministic interpreter is the source of final chart data:

- active gates come only from the 13 Personality and 13 Design planetary activations
- active channels are derived only when both canonical endpoint gates are active
- defined centers come only from derived active channel endpoints
- type, authority, profile, strategy, definition, not-self theme, and signature are derived with Python rules
- visual gates, channels, and centers are supporting evidence only
- visual disagreements become validation warnings instead of overriding derived data

### Phase 2 Environment

Phase 2 uses these settings:

```env
OPENAI_API_KEY=
HD_VISION_MODEL=gpt-5.5
HD_VISION_REASONING_EFFORT=high
HD_VISION_REAL_API=0
```

When run from the repository root, Phase 2 loads .env automatically. Inline environment variables such as `HD_VISION_REAL_API=1` override values from .env.

`HD_VISION_REAL_API=0` is the default and prevents real Vision API calls. Set `HD_VISION_REAL_API=1` only for an intentional manual API call. `OPENAI_API_KEY` is required only in real API mode. The default model is `gpt-5.5`, and the default reasoning effort is `high`.

### Offline Local Pipeline Test

This command runs the local parser, interpreter, and validation flow using a sanitized mock Vision response. It does not call OpenAI and does not require `OPENAI_API_KEY`.

```sh
uv run python scripts/extract_bodygraph.py \
  tests/fixtures/bodygraph/test1.png \
  --mock-response tests/fixtures/bodygraph/test1_raw_response.json
```

Use `--json` for machine-readable output:

```sh
uv run python scripts/extract_bodygraph.py \
  tests/fixtures/bodygraph/test1.png \
  --mock-response tests/fixtures/bodygraph/test1_raw_response.json \
  --json
```

`tests/fixtures/bodygraph/test1.png` is a small non-private synthetic fixture. The actual mock extraction comes from `test1_raw_response.json`.

### Manual Real Vision Test

Real Vision use is manual, opt-in, and may incur API cost. Put private chart images under `data/bodygraph_samples/images/`. If you want to save JSON output, create a private output directory first:

```sh
mkdir -p data/bodygraph_samples/private
```

Run real extraction with human-readable output:

```sh
HD_VISION_REAL_API=1 uv run python scripts/extract_bodygraph.py \
  data/bodygraph_samples/images/chart-image.png
```

Use `--json` to save machine-readable output:

```sh
HD_VISION_REAL_API=1 uv run python scripts/extract_bodygraph.py \
  data/bodygraph_samples/images/chart-image.png \
  --json > data/bodygraph_samples/private/chart-image.bodygraph_prediction.json
```

The inline `HD_VISION_REAL_API=1` overrides `HD_VISION_REAL_API=0` from `.env`. The CLI must not print API keys, image bytes, or base64 image data. Files under `data/bodygraph_samples/private/` may contain personal chart data and must not be committed.

### Offline Evaluation

Evaluation is optional and is mainly for development accuracy checks. It is not required for normal one-chart extraction.

The evaluator compares saved prediction JSON files against manually verified golden labels. The prediction and golden label must describe the same chart and use the same `case_id`. Do not compare a private real-chart prediction against `golden_labels.example.json`; that file is only a synthetic safe-to-commit example.

A normal one-chart extraction flow usually stops after saving a prediction:

```sh
mkdir -p data/bodygraph_samples/private

HD_VISION_REAL_API=1 uv run python scripts/extract_bodygraph.py \
  data/bodygraph_samples/images/chart-image.png \
  --json > data/bodygraph_samples/private/chart-image.raw_prediction.json
```

For evaluation, first create a manually verified golden-label file for the same image, for example:

```text
data/bodygraph_samples/private/golden_labels.local.json
```

Then wrap the saved single-image prediction in the canonical predictions file shape with the matching `case_id`:

```sh
python - <<'PY'
import json
from pathlib import Path

case_id = "chart_image_001"
src = Path("data/bodygraph_samples/private/chart-image.raw_prediction.json")
dst = Path("data/bodygraph_samples/private/predictions.local.json")

prediction = json.loads(src.read_text(encoding="utf-8"))
dst.write_text(
    json.dumps(
        {
            "schema_version": "phase2_predictions_v1",
            "predictions": [{"case_id": case_id, **prediction}],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
```

Golden-label files use `phase2_golden_labels_v2` and contain the exact top-level fields `schema_version`, `documentation`, `recommended_sample_coverage`, and `cases`. Prediction files use `phase2_predictions_v1` and contain only `schema_version` and `predictions`; each prediction has `case_id`, `raw_vision`, `derived_chart_data`, and `validation_result`. The evaluator intentionally rejects older aliases and unwrapped prediction forms.

Run evaluation:

```sh
uv run python scripts/evaluate_bodygraph_extraction.py \
  --golden-labels data/bodygraph_samples/private/golden_labels.local.json \
  --predictions data/bodygraph_samples/private/predictions.local.json
```

Evaluation reports activation exact-match rates, set precision/recall/F1 for gates, channels, and centers, exact matches for basic chart info, and warning-code metrics. Use `--json` for machine-readable metrics. `--threshold METRIC=VALUE` may be repeated to fail when an aggregate metric misses a required threshold.

### Offline Project Verification

Default tests are offline, deterministic, and independent of private images, OpenAI services, and Phase 1 Chroma storage.

```sh
uv run pytest
uv run ruff check .
```
