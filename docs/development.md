# Development reference

For setup and CLI/Streamlit usage, start with the [README](../README.md). Docker usage and publishing are in the [Docker guide](../docker/README.md). This document describes the current system; completed implementation plans remain in Git history. Project working rules are in [AGENTS.md](../AGENTS.md).

## PDF ingestion and hybrid indexing

```text
Local PDFs -> PyMuPDFReader -> SentenceSplitter
  -> canonical nodes -> Chroma + BM25 -> verified manifest
```

- [rag/ingestion.py](../src/human_design/rag/ingestion.py) discovers PDFs in deterministic order and extracts embedded text with source/page metadata. There is no OCR.
- [rag/chunking.py](../src/human_design/rag/chunking.py) uses SentenceSplitter with defaults of 800 tokens and 80 overlap, preserving source/page metadata, embedding model and ingestion version.
- [rag/embeddings.py](../src/human_design/rag/embeddings.py) creates the configured OpenAI embedding client. Build/query callers enforce the opt-in and key before constructing it.
- [rag/hybrid_index.py](../src/human_design/rag/hybrid_index.py) coordinates the build. [rag/vector_store.py](../src/human_design/rag/vector_store.py) opens Chroma only at an explicit path.

### Hybrid index rebuilds

`scripts/build_hybrid_index.py` requires `HD_RAG_REAL_EMBEDDINGS=1` and `OPENAI_API_KEY` before loading PDFs or constructing providers. `HD_RAG_INDEX_DIR` defaults to `storage/hybrid_v1`, containing:

```text
manifest.json
nodes.jsonl
chroma/
bm25/
```

Chroma and BM25 receive the same canonical node objects from one build. A chunk ID is SHA-256 of compact deterministic JSON containing the PDF-byte SHA-256, stable page key, zero-based per-page chunk ordinal and normalized text. Text normalizes CRLF/CR to LF, applies Unicode NFC and trims only outer whitespace. Blank chunks are excluded before assigning ordinals. The LlamaIndex node ID equals the canonical chunk ID.

The deterministic `ingestion_id` excludes itself from its hash and is shared by the manifest and both indexes. Reload validates canonical IDs, metadata and counts. Keep all four artifacts together and preserve the v1 identity algorithms.

The target must be empty; builds never overwrite or delete existing files. When PDFs, chunk settings or the embedding model change, build into a fresh directory and use that root for subsequent questions. Queries reload the existing indexes without reading PDFs or rebuilding vectors. Both backends are required with matching ingestion identity; there is no single-index fallback.

## BodyGraph extraction

```text
Image -> Vision extraction -> strict parser
  -> deterministic interpreter -> validation -> BodyGraphExtractionResult
```

- [vision/client.py](../src/human_design/vision/client.py) prepares images and calls OpenAI. Real extraction requires `HD_VISION_REAL_API=1` and `OPENAI_API_KEY`. `HD_VISION_MODEL` and `HD_VISION_REASONING_EFFORT` are separate from answer-generation settings.
- [vision/parser.py](../src/human_design/vision/parser.py) parses provider JSON, normalizes accepted aliases and records recoverable issues.
- [vision/interpreter.py](../src/human_design/vision/interpreter.py) derives chart facts using [vision/constants.py](../src/human_design/vision/constants.py). [vision/validation.py](../src/human_design/vision/validation.py) combines parser issues and visual/deterministic disagreements into typed warnings and validity.
- [vision/pipeline.py](../src/human_design/vision/pipeline.py) returns a frozen `BodyGraphExtractionResult` containing raw extraction, derived chart data and validation.

The 13 Personality and 13 Design planetary activations determine active Gates. Active Channels require both canonical endpoint Gates; defined Centers come from those Channels. Python rules derive Type, Authority, Profile, Strategy, Definition, not-self theme and signature.

Visual-only Gates, Channels and Centers cannot override derived facts. Canonical Center vocabulary includes `G` and `Ego`. Unsupported or incomplete information is reported conservatively, never invented. Invalid charts stop before retrieval; the reading adapter exposes deterministic facts and minimal safe warnings.

## Focused Q&A

[`human_design.reading.ReadingPipeline`](../src/human_design/reading/pipeline.py) provides:

- `answer_knowledge_question(KnowledgeQuestionRequest)` for book-based questions.
- `answer_chart_question(ChartQuestionRequest)` for a typed `BodyGraphExtractionResult`.
- `answer_chart_image_question(ChartImageQuestionRequest)` as a thin facade over the extraction pipeline above.

The chart core receives no image Path, bytes or raw Vision JSON. Providers and indexes are initialized lazily, after validation and scope checks. Full-chart readings are unsupported.

```text
Validate question -> focused-Q&A guard -> optional chart validation/projection
  -> relevant-fact selection -> separate dense/BM25 queries
  -> both retrievals -> RRF -> configured reranking -> final Top-K
  -> citation assignment -> escaped prompt -> structured generation
  -> validated public AnswerResult
```

Queries must be nonblank and at most 2,000 Python characters before trimming. Broad full-reading requests return `needs_focus` before chart adaptation or providers. Invalid charts return `invalid_chart`. Empty final evidence returns `insufficient_evidence` without prompt rendering or generation, even when chart facts exist.

### Fact selection and retrieval

[reading/chart_context.py](../src/human_design/reading/chart_context.py) projects validated deterministic facts. [reading/query_builder.py](../src/human_design/reading/query_builder.py) selects existing facts by intent and never invents active Gates or Channels. Decision questions use Type, Authority and Strategy; Profile, Center, Gate, Channel and planetary questions use the matching focused facts and only useful context.

Dense retrieval receives the semantic query; BM25 receives a separate lexical query. Sparse normalization supports English/Chinese aliases, canonical `G`/`Ego` and Gate/Channel/Profile forms without rewriting the semantic query. Adapters preserve raw scores and 1-based ranks.

[rag/hybrid_retriever.py](../src/human_design/rag/hybrid_retriever.py) checks ingestion identity before either query, merges only by canonical chunk ID and rejects conflicting text/source metadata. RRF adds `1 / (rrf_k + rank)` for each present rank, never combining raw scores. Ordering is descending RRF, ascending best rank, then lexical chunk ID; truncation to `fusion_top_k` follows fusion.

`NoOpReranker` preserves RRF order. Optional Cohere receives the semantic query and fused candidate text, maps provider indexes back to canonical chunks and preserves provenance. It requires the optional dependency, selected provider, model, key and API opt-in. There is no silent reranking fallback. Up to `final_top_k` selected chunks proceed to generation.

### Prompt, citations and privacy

Only final sources receive `S1`, `S2`, etc.; only selected chart facts enter `PromptContext`. User questions and passages are escaped untrusted data, including legitimate path-like text. Structured filesystem provenance and retrieval scores are omitted. Names, birth details, images, confidence, uncertain items and raw provider payloads are never projected into the prompt as chart metadata.

Generation uses one OpenAI Responses request with structured output, `store=False` and no tools or repair call. It requires `HD_RAG_REAL_GENERATION=1`, a configured generation model and `OPENAI_API_KEY`.

Returned source IDs and chart-fact paths must belong to the supplied context. Every bracket citation must also appear in `used_source_ids`; unknown references fail. An OK answer always has at least one valid bracket citation. Public citations include only bracket-cited sources; valid IDs reported in `used_source_ids` but not cited add a safe warning instead of a public citation.

Errors use safe project-owned messages. Never log prompts, full passages, keys, image/base64 content, birth data or private provider payloads. Keep secrets, PDFs, private charts, generated indexes and predictions out of Git. Human Design is framed as reflective information without diagnosis or deterministic guarantees. `REFUSED` is reserved; there is no automatic refusal classifier.

### Streamlit session reuse

[scripts/streamlit_app.py](../scripts/streamlit_app.py) fingerprints uploaded image content and keeps validated typed charts only in that browser session. Invalid/failed extractions are not cached; valid extraction survives a later answer failure. Questions about a cached chart call the typed chart core and skip Vision, while each question still selects facts, retrieves and generates a fresh answer. Sessions are independent and cached chart data is not written to disk. Each chart-image CLI invocation performs a new extraction. Both interfaces preserve the same validation, scope, citation and provider gates.

## CLI source map

Each CLI script owns its argument parsing. Look for `parser.add_argument(...)` below, or run `uv run python scripts/<script>.py --help` with the [offline environment](../README.md#default-verification).

| Tool | Source | Parser location |
| --- | --- | --- |
| Focused Q&A | [ask_human_design.py](../scripts/ask_human_design.py) | `_build_parser()`; query, `--bodygraph`, `--json` |
| Hybrid index build | [build_hybrid_index.py](../scripts/build_hybrid_index.py) | `main()`; only `--help`, settings come from configuration |
| Chart extraction | [extract_bodygraph.py](../scripts/extract_bodygraph.py) | `_build_parser()` |
| Chart evaluation | [evaluate_bodygraph_extraction.py](../scripts/evaluate_bodygraph_extraction.py) | `_build_parser()` |
| Retrieval evaluation | [evaluate_hybrid_retrieval.py](../scripts/evaluate_hybrid_retrieval.py) | `main()` |

`uv run --extra rerank` is parsed by uv and selects the optional dependency in [pyproject.toml](../pyproject.toml); it does not enable Cohere calls. Streamlit owns its launch options (`uv run streamlit run --help`); the app defines its form in `main()` without a project argument parser.

Environment settings come from [rag/config.py](../src/human_design/rag/config.py) (`load_config()`) and [vision/config.py](../src/human_design/vision/config.py) (`load_vision_config()`). Process values override `.env`, including provider opt-ins. API keys alone never enable paid calls.

### Tests and diagnostics

Use the [explicit offline verification command](../README.md#default-verification) for routine checks and smoke tests. Tests use fake providers, sanitized fixtures and temporary storage, never production indexes. Add filenames after `uv run pytest` to run a subset:

- Index build/reload: `test_ingestion.py`, `test_chunking.py`, `test_embeddings.py`, `test_hybrid_index.py`, `test_bm25.py`, `test_retriever.py`.
- Charts: `test_bodygraph_parser.py`, `test_bodygraph_interpreter.py`, `test_bodygraph_validation.py`, `test_bodygraph_evaluation.py`, `test_extract_bodygraph.py`.
- Answers/interfaces: `test_reading_pipeline.py`, `test_generation.py`, `test_ask_human_design.py`, `test_streamlit_app.py`.

For an extraction smoke test, use the same offline environment and replace `uv run pytest` with:

```sh
uv run python scripts/extract_bodygraph.py \
  tests/fixtures/bodygraph/test1.png \
  --mock-response tests/fixtures/bodygraph/test1_raw_response.json --json
```

### Evaluation tools

`scripts/evaluate_bodygraph_extraction.py` compares saved predictions with manually verified labels for the same chart/case ID without calling Vision. Golden labels use `phase2_golden_labels_v2`; wrapped predictions use `phase2_predictions_v1`. [golden_labels.example.json](../data/bodygraph_samples/golden_labels.example.json) is synthetic, not ground truth for private charts. Keep private images, labels and predictions in the ignored `data/bodygraph_samples/` directories.

Chart evaluation reports activation exact matches, set precision/recall/F1, basic-info matches and warning metrics. Missing applicable predictions count as failures; malformed data fails validation. Optional aggregate thresholds can fail evaluation.

`scripts/evaluate_hybrid_retrieval.py` independently reports dense-only, BM25-only and RRF metrics: Hit@5, MRR@20, exact entity hits and expected source/page hits. It reloads indexes without rebuilding them. Aggregate thresholds determine exit status; unlabeled metrics are excluded. Real dense evaluation requires the embeddings opt-in and incurs cost.

Manual answer review checks relevance, groundedness, citation usefulness, clarity and reflective framing. Automated tests do not use an LLM judge.
