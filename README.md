# Human Design RAG

Phase 1 is the stable baseline: a local, text-only Human Design knowledge base built from local PDF books. Phase 2 adds a separate local BodyGraph Vision extraction pipeline documented below; it does not change Phase 1 ingestion or retrieval behavior.

Phase 3 v1 adds focused, cited Q&A using a separate hybrid index and an optional validated Phase 2 chart. Full-chart readings remain deferred to Phase 3.1.

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

Run the [offline verification commands](#default-verification).

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
- They use temporary test storage and leave any existing local Chroma store intact.

Use this verification flow when checking the offline test path. It removes inherited provider keys and explicitly disables real embedding, generation, reranking, and Vision API modes. Process-environment values intentionally override provider and real-mode opt-ins in `.env`, including Cohere selection. A local `.env` may still supply keys, but the disabled flags prevent paid calls. Keep any existing `storage/chroma` directory; verification does not require deleting, resetting, or removing it.

```sh
env \
  -u OPENAI_API_KEY \
  -u COHERE_API_KEY \
  HD_RAG_REAL_EMBEDDINGS=0 \
  HD_RAG_REAL_GENERATION=0 \
  HD_RAG_RERANK_PROVIDER=none \
  HD_RAG_REAL_RERANK_API=0 \
  HD_VISION_REAL_API=0 \
  uv run pytest
uv run ruff check .
git diff --check
git status --short
```

## Manual Ingestion

Real embedding ingestion is manual and opt-in:

```sh
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/ingest_pdfs.py
```

This command may call OpenAI and create embedding cost. It requires local PDFs and `OPENAI_API_KEY`, and it writes vectors into the configured local Chroma directory.

In this version, Phase 1 ingestion does not append to populated collections. It refuses before loading PDFs or constructing an embedding client to avoid duplicate content and repeated embedding cost. For an intentional rebuild, select a fresh/empty collection with `HD_RAG_COLLECTION`. Existing storage is not deleted or reset.

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

Golden-label files use `phase2_golden_labels_v2` and contain the exact top-level fields `schema_version`, `documentation`, `recommended_sample_coverage`, and `cases`. Prediction files use `phase2_predictions_v1` and contain only `schema_version` and `predictions`; each prediction requires `case_id` and may supply `raw_vision`, `derived_chart_data`, and `validation_result`. The evaluator intentionally rejects older aliases, unsupported entry fields, duplicate case IDs, and unwrapped prediction forms.

Missing prediction cases, output sections, or fields count as failures for applicable labeled metrics and remain in their denominators. Disabled metric families remain absent. Only an explicitly predicted `null` activation can match an expected null; a missing activation cannot. Explicit empty arrays use the normal set metrics (empty versus empty is perfect), while missing collections receive zero precision, recall, and F1. Null collections are invalid.

Malformed supplied prediction fields are evaluation-input errors, even when their metric family is disabled. Parent sections and activation columns must be objects when supplied. Activations accept null, Gate.Line strings, or integer gate/line objects (gate 1–64, line 1–6). Gate collections accept integers or numeric strings from 1–64; booleans and floating-point gates are rejected. Centers must be canonical strings; channels accept canonical or reversed valid forms. Basic-info values must be strings, `is_valid` must be a real boolean, and warning arrays retain the canonical warning-code metadata contract. Missing warnings score as failures; explicit empty warning arrays retain normal empty-set scoring. Valid prediction representations and scoring formulas are otherwise unchanged.

Run evaluation:

```sh
uv run python scripts/evaluate_bodygraph_extraction.py \
  --golden-labels data/bodygraph_samples/private/golden_labels.local.json \
  --predictions data/bodygraph_samples/private/predictions.local.json
```

Evaluation reports activation exact-match rates, set precision/recall/F1 for gates, channels, and centers, exact matches for basic chart info, and warning-code metrics. Use `--json` for machine-readable metrics. `--threshold METRIC=VALUE` may be repeated to fail when an aggregate metric misses a required threshold.

### Offline Project Verification

Default tests are offline, deterministic, and independent of private images, OpenAI services, and Phase 1 Chroma storage.

Use the [offline verification commands](#default-verification), which explicitly disable real provider modes and preserve existing local storage.

## Phase 3 v1: Focused Q&A

Ask one question about Authority, Profile, a Gate, a Channel, a Center, Type, Strategy, or Definition. Knowledge-only questions use the books without a chart. Chart-image questions add only the validated chart facts relevant to the question. Full-chart reading, multi-turn history, and sectioned reports are deferred to Phase 3.1.

The public Python service is `human_design.reading.ReadingPipeline`, with `answer_knowledge_question(KnowledgeQuestionRequest)`, `answer_chart_question(ChartQuestionRequest)`, and `answer_chart_image_question(ChartImageQuestionRequest)` methods. Construction is lazy: invalid questions and explicit full-reading requests terminate before configuration, storage, or provider work. Questions must be nonblank and at most 2,000 Python characters, measured before trimming.

```text
Question -> input validation -> focused-Q&A scope guard
  -> purpose-specific dense and sparse queries
  -> dense retrieval + BM25 retrieval -> RRF -> optional reranking -> final Top-K
  -> citation assignment and escaped prompt -> one structured Responses request
  -> local citation/chart-fact validation -> public AnswerResult
```

For image mode, the outer facade passes the image Path only to Phase 2. Phase 2 reuses its extraction, parser, interpreter, and validation flow and constructs the existing typed `BodyGraphExtractionResult`. The chart core receives `ChartQuestionRequest(query, bodygraph_result)`, never image paths, bytes, or raw Vision JSON. It projects safe deterministic facts, then selects facts relevant to the question before retrieval. Visual-only evidence, confidence, names, birth information, and unrelated metadata do not become prompt chart facts.

### One-time hybrid rebuild

A clean checkout has no production Phase 3 index. Before a real question, intentionally build a new hybrid index from your local PDFs. Phase 3 cannot use the legacy Phase 1 Chroma collection as its production dense index; it does not migrate, query, reset, or delete that collection.

The default Phase 3 root is `storage/hybrid_v1`, controlled by `HD_RAG_INDEX_DIR`:

```text
storage/hybrid_v1/
  manifest.json
  nodes.jsonl
  chroma/
  bm25/
```

The build reuses Phase 1 PDF extraction and chunking semantics. PDF-byte fingerprints, stable page keys, per-page chunk ordinals, and normalized text produce deterministic canonical chunk IDs. The same in-memory canonical nodes and IDs feed both Chroma and BM25. Canonical nodes persist in `nodes.jsonl`; the manifest and both indexes carry one deterministic ingestion identity. Reload verifies identities/counts, and querying requires both indexes. There is no dense-only or BM25-only production fallback.

For an intentional paid build, configure `OPENAI_API_KEY`, `HD_RAG_PDF_DIR`, and `HD_RAG_EMBED_MODEL`, then run:

```sh
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/build_hybrid_index.py
```

The target must be empty. A non-empty root fails without overwriting or deleting it. For a later intentional rebuild, choose a fresh empty `HD_RAG_INDEX_DIR` and use that same root when querying. Embedding-model changes require a matching rebuild. Existing `storage/chroma` remains independent.

### Queries, fusion, and optional reranking

Dense retrieval receives a natural semantic query with selected chart context. BM25 receives a separate normalized lexical query with the existing English/Chinese aliases and canonical G/Ego center vocabulary. RRF combines 1-based ranks with `1 / (rrf_k + rank)`; raw dense and BM25 scores are not combined. Candidates deduplicate by canonical chunk ID and have deterministic ordering. Fewer available candidates simply produce a shorter result.

`HD_RAG_RERANK_PROVIDER=none` preserves RRF order and selects final Top-K. The only optional reranker is Cohere. Install it with `uv sync --extra rerank`, then configure `COHERE_API_KEY`, `HD_RAG_RERANK_MODEL`, `HD_RAG_RERANK_PROVIDER=cohere`, and `HD_RAG_REAL_RERANK_API=1`. Cohere receives the semantic query and candidate text, not a raw chart dump. It is never imported or constructed for provider `none`.

Relevant settings are documented in `.env.example`: `HD_RAG_DENSE_TOP_K`, `HD_RAG_SPARSE_TOP_K`, `HD_RAG_FUSION_TOP_K`, `HD_RAG_FINAL_TOP_K`, and `HD_RAG_RRF_K`. Only final selected sources receive citation IDs S1, S2, and so on. Public citations include only sources actually bracket-cited in the validated answer; internal retrieval scores and passages are excluded.

### CLI modes

The application interface is exactly one positional query, optional `--bodygraph`, optional `--json`, and standard `--help`. There are no mock, provider/model override, prompt-dump, or chunk-dump flags.

After configuring the required opt-ins and building the hybrid index, knowledge-only mode is:

```sh
uv run python scripts/ask_human_design.py "What is Sacral Authority?"
```

Chart-image mode additionally requires the Vision opt-in:

```sh
uv run python scripts/ask_human_design.py \
  "How should I make decisions?" \
  --bodygraph data/bodygraph_samples/images/chart.png
```

For one public `AnswerResult` JSON object with string status values:

```sh
uv run python scripts/ask_human_design.py "What does Gate 42 mean?" --json
```

Human output contains the status, answer, cited source/page metadata, and safe warnings. `needs_focus`, `invalid_chart`, and `insufficient_evidence` are ordinary results with exit code 0. Invalid input/configuration/index/provider setup returns 2; unexpected operational failures return 1 with a generic safe message. A full-reading request asks the user to choose one topic and performs no provider or storage work. Invalid charts stop before retrieval. Missing evidence stops before prompt assembly or generation, including when structured chart facts are available.

### Paid-provider opt-ins and privacy

Default automated tests are offline, free, and credential-independent. They use injected fakes and temporary storage; they do not build a production index or call paid services. Use the [full offline verification command](#default-verification) so `.env` cannot enable a paid mode or select Cohere during verification.

Real operation is manual and may incur cost:

- Dense builds and queries require `HD_RAG_REAL_EMBEDDINGS=1` and `OPENAI_API_KEY`.
- Generation requires `HD_RAG_REAL_GENERATION=1`, `HD_RAG_GENERATION_MODEL`, and `OPENAI_API_KEY`.
- Image extraction requires `HD_VISION_REAL_API=1` and `OPENAI_API_KEY`, using the existing `HD_VISION_MODEL` settings.
- Cohere requires the optional extra and all four Cohere settings listed above.

For example, after setting a supported generation model and key in your process environment or private `.env`:

```sh
HD_RAG_REAL_EMBEDDINGS=1 HD_RAG_REAL_GENERATION=1 \
  uv run python scripts/ask_human_design.py "What is Sacral Authority?"
```

Add `HD_VISION_REAL_API=1` only when intentionally using chart-image mode. Cohere is separately opt-in. Disabled providers give configuration-variable guidance without printing secret values.

Embedding requests send PDF text or query text to OpenAI. Vision sends the chart image, which may contain private information. Generation sends the question, selected normalized chart facts, and final reference passages to OpenAI. Optional reranking sends its semantic query and candidate passages to Cohere. Consider those disclosures and costs before enabling each mode.

Generation uses exactly one OpenAI Responses request with structured output, `store=False`, and no tools. There is no repair/regeneration request. `store=False` does not make the request local: provider processing still occurs. Prompt sections escape untrusted user/source text, and instructions in passages cannot override template rules. Path-like syntax within natural text remains escaped data; structured local filesystem provenance is omitted from prompts and public citations.

Human Design is presented as general reflective/experimental information, not scientifically validated medical guidance. The prompt prohibits medical and mental-health diagnosis, legal/financial decision making, guaranteed futures, and deterministic death, illness, pregnancy, or compatibility claims. `REFUSED` is reserved; there is no automatic refusal detector, moderation service, or safety classifier.

Do not log full prompts, chunks, provider payloads, API keys, image/base64 content, or birth data. Do not commit `.env`, PDFs, generated storage/indexes, private chart images, raw Vision responses, or private outputs. Local files under ignored `storage/` and private data directories stay local unless an explicitly enabled provider receives the inputs described above.

### Evaluation and manual review

Retrieval evaluation remains separate from generation. The sanitized fixture at `tests/fixtures/rag/hybrid_queries.example.json` compares dense-only, BM25-only, and RRF hybrid modes against an existing index using Hit@5, MRR@20, exact entity hit rate, and expected source/page hit rate. Real dense evaluation requires `HD_RAG_REAL_EMBEDDINGS=1`. It does not build or mutate indexes. Exit codes distinguish passed aggregate thresholds (0), failed thresholds (1), and invalid setup (2); unlabeled metrics are excluded from thresholds.

After offline tests pass, an optional intentional paid smoke test can be reviewed manually:

- Relevance: does the answer address the single question and only relevant chart facts?
- Groundedness: are chart facts supplied by Phase 2 and book-derived explanations supported by passages?
- Citation usefulness: do bracket IDs resolve to the stated source/page and support the nearby claim?
- Clarity: is the answer concise, understandable, and explicit about missing evidence?
- Reflective framing: does it encourage experimentation and avoid diagnosis, guarantees, or deterministic predictions?

These checks require human review of an actual answer; automated tests validate structure, orchestration, citations, and privacy boundaries without an LLM judge. No paid smoke test is part of the final offline gate.
