# Phase 3 Implementation Plan: Hybrid Retrieval and Grounded Human Design Q&A

> **For agentic workers:** Implement this plan one task at a time. Use TDD where practical: write each test first, verify it fails for the expected reason, implement the minimum code, then verify it passes. Do not expand Phase 3 beyond focused grounded Q&A unless the user explicitly changes scope.

**Goal:** Build a local focused-question answering pipeline that optionally uses a validated Phase 2 BodyGraph result, retrieves grounded knowledge from newly rebuilt dense and sparse indexes, and returns a cited answer.

**Architecture:** Phase 3 performs one deterministic rebuild that creates canonical nodes with shared chunk IDs, then builds Chroma and BM25 from exactly those nodes. At question time it selects only relevant chart facts, creates separate dense and sparse queries, fuses retrieval ranks with Reciprocal Rank Fusion (RRF), optionally reranks with Cohere, and makes one opt-in OpenAI Responses API generation call with validated citations.

**Tech Stack:** Python 3.11+, uv, pytest, ruff, LlamaIndex core, existing `PyMuPDFReader`, existing `SentenceSplitter`, existing OpenAI embeddings and Chroma integration, LlamaIndex BM25 retrieval, OpenAI Responses API structured output, optional Cohere reranking, and the existing Phase 2 `BodyGraphExtractionResult` model.

---

## Goal

Phase 3 builds a local focused-question answering pipeline that:

- accepts a Human Design knowledge question
- optionally accepts a local BodyGraph image at the outer CLI/application boundary
- uses Phase 2 to extract and validate chart information when an image is supplied
- selects only chart facts relevant to the current question
- creates separate purpose-specific dense and sparse retrieval queries
- retrieves from persisted Phase 3 Chroma and BM25 indexes
- combines results with RRF using shared deterministic chunk IDs
- optionally reranks fused candidates with Cohere
- sends only final selected sources and relevant chart facts to one OpenAI generation model
- returns a grounded answer with validated citations

Phase 3 v1 is focused Q&A only. It is not a full-chart-reading system.

## Fixed Architecture Decisions

These decisions are fixed for Phase 3 v1.

### One-Time Full Hybrid Ingestion Rebuild

Phase 3 performs one full ingestion rebuild:

```text
load PDFs
  -> extract embedded text using the existing Phase 1 approach
  -> chunk with the existing Phase 1 chunking settings
  -> create deterministic canonical nodes
  -> persist canonical nodes locally
  -> build a new Chroma index
  -> build a persisted BM25 index
  -> write shared ingestion metadata
```

Chroma and BM25 must be built from exactly the same canonical nodes. Phase 3 reuses the current Phase 1 `discover_pdfs()`, `load_pdfs()`, `build_text_extraction_report()`, and `chunk_documents()` behavior where practical; it does not change embedded-text extraction or the default `SentenceSplitter(chunk_size=800, chunk_overlap=80)` settings.

The legacy Phase 1 Chroma index under the existing `AppConfig.chroma_dir` default of `storage/chroma` is not the Phase 3 production dense index. It may remain locally for regression comparison, backup, and the legacy Phase 1 retrieval smoke test, but it must never be combined with the new Phase 3 BM25 corpus. Phase 3 uses one new index root, defaulting to `storage/hybrid_v1`.

### Shared Deterministic Chunk Identity

Every canonical chunk has one deterministic shared `chunk_id`. The same ID is used by:

- `nodes.jsonl`
- Phase 3 Chroma
- Phase 3 BM25
- RRF deduplication and fusion
- retrieval evaluation
- final citations

Do not use random LlamaIndex node IDs as the Phase 3 shared identity. The canonical ID input must include:

```text
source content fingerprint
+ page label or page number
+ deterministic chunk ordinal
+ normalized chunk text
```

Use a deterministic serialization rather than ambiguous string concatenation. The v1 algorithm should:

1. hash the source PDF bytes with SHA-256
2. normalize chunk text by converting CRLF/CR to LF, applying Unicode NFC, and trimming leading/trailing whitespace without collapsing meaningful internal whitespace
3. choose a stable page key such as `label:<page_label>`, otherwise `number:<page_number>`, otherwise `none`
4. use a zero-based chunk ordinal within that source-page key
5. serialize `[source_sha256, page_key, chunk_index, normalized_chunk_text]` as compact UTF-8 JSON
6. SHA-256 hash those serialized bytes

Conceptually:

```python
chunk_id = sha256(
    canonical_json(
        [source_sha256, page_key, chunk_index, normalized_chunk_text]
    ).encode("utf-8")
).hexdigest()
```

The separator, JSON options, normalization, page-key fallback, and zero-based ordinal are part of the manifest's `chunk_id_algorithm` and must be tested. A duplicate ID is an ingestion error.

### Separate Dense and Sparse Queries

Dense and BM25 retrieval receive different purpose-specific queries.

The dense query:

- preserves the semantic meaning of the user question
- includes only question-relevant chart context
- remains natural language suitable for embedding retrieval
- may remain Chinese for a Chinese question while including only the necessary canonical chart facts

The sparse query:

- emphasizes exact Human Design entities
- normalizes Gate, Channel, Profile, Authority, Type, Center, Line, and planetary terms
- translates only a small explicit set of Chinese Human Design aliases into canonical English terms
- is suitable for BM25 exact-term matching

Phase 3 v1 assumes that the BM25 reference corpus is primarily English, such as English PDFs in the style of *The Definitive Book*. Chinese user questions are supported by dense embeddings and by the small explicit sparse-query alias map, which emits canonical English Human Design terms. Chinese document-side segmentation and Chinese BM25 corpus optimization are out of scope for v1.

Do not send every active Gate and Channel into every query. Do not use one large `BodyGraphExtractionResult` or `ChartContext` dump as a retrieval query. Phase 3 v1 does not use an LLM for query rewriting, entity extraction, or chart-fact selection.

### RRF Fusion

Dense and BM25 results are combined using Reciprocal Rank Fusion. Shared `chunk_id` values provide deduplication. RRF uses 1-based ranks, not raw score addition:

```text
rrf_score(chunk) = dense contribution + sparse contribution

dense contribution = 1 / (rrf_k + dense_rank)
sparse contribution = 1 / (rrf_k + sparse_rank)
```

A missing rank contributes zero. Dense similarity scores and BM25 scores remain optional diagnostics because they are not directly comparable. Stable ties are resolved by best individual rank and then `chunk_id`.

### Minimum Evidence for Generation

Every answer that proceeds through the LLM generation path must have at least one final retrieved reference source. Structured Phase 2 chart facts may be stated without a PDF citation, but Human Design explanations or interpretations must be grounded in retrieved sources.

For example:

```text
Your Solar Plexus is undefined.

In Human Design, an undefined Solar Plexus is generally described as ... [S1]
```

The first sentence reports a structured Phase 2 fact and does not require `[S1]`. The explanatory statement requires a citation. If retrieval produces no final source, Phase 3 returns `insufficient_evidence` before generation. Phase 3 v1 does not add a separate direct fact-only response path or allow chart facts alone to trigger an uncited LLM answer.

### Focused Q&A Only

Supported Phase 3 v1 questions include:

```text
What is Sacral Authority?
How should I make decisions according to my chart?
What does Gate 42 mean in my chart?
What does Channel 42-53 mean?
What does my 4/6 Profile mean?
Is my Solar Plexus defined?
```

Broad requests such as these return a deterministic `needs_focus` result:

```text
Give me a complete reading of my chart.
Explain everything in my chart.
完整解讀我的人類圖。
```

The response asks the user to choose a narrower topic such as Authority, Profile, Gate, Channel, Center, Type, Strategy, or Definition. Sectioned full-chart retrieval, multi-section generation, and final synthesis belong to Phase 3.1.

### Minimal Reranker Support

Phase 3 v1 supports exactly:

```text
none
cohere
```

`none` preserves RRF order. `cohere` is optional, remote, and explicitly enabled. Phase 3 v1 does not add BGE, FlagEmbedding, sentence-transformers, local reranker downloads, GPU deployment, or multiple reranker benchmarks. BGE is future Phase 3.1 work only.

### No External Moderation Pipeline

Phase 3 v1 does not call the OpenAI Moderation API or any other moderation/classifier service for each input or output. It keeps lightweight baseline protections:

- a fixed maximum query length
- clearly delimited prompt sections
- retrieved chunks treated as data, never instructions
- chart facts treated as data
- no arbitrary tools
- no shell or file access exposed to generation
- no API keys in prompts
- no full prompt or full chunk logging
- no birth-data logging
- sanitized fixtures
- OpenAI generation requests use `store=False`

## Architecture

### Knowledge-Only Flow

```text
KnowledgeQuestionRequest
    ↓
Validate focused question
    ↓
Build dense query
Build sparse query
    ↓
Chroma dense Top-K
BM25 sparse Top-K
    ↓
RRF fusion
    ↓
Optional Cohere rerank
    ↓
Final Top-K sources
    ↓
Prompt assembly
    ↓
OpenAI generation
    ↓
Citation and chart-fact validation
    ↓
AnswerResult
```

### Query Plus BodyGraph Flow

```text
ChartImageQuestionRequest
    ↓
Phase 2 BodyGraph extraction
    ↓
BodyGraphExtractionResult
    ↓
Validation gate
    ↓
ChartQuestionRequest
    ↓
Safe normalized ChartContext
    ↓
Question-relevant ChartFact selection
    ↓
Build dense query
Build sparse query
    ↓
Same hybrid retrieval and generation pipeline
    ↓
AnswerResult
```

The BodyGraph image path is an outer application input. The Phase 3 core pipeline must not consume an image or `Path`; it receives an already parsed and validated `BodyGraphExtractionResult`.

## Request Boundaries

The current Phase 2 full result is the frozen dataclass:

```python
from human_design.vision.models import BodyGraphExtractionResult
```

It is also re-exported from `human_design.vision`. Phase 3 must use this actual type; there is no `BodyGraphPipelineResult` in the repository.

Preserve these request boundaries:

```python
class KnowledgeQuestionRequest:
    query: str


class ChartImageQuestionRequest:
    query: str
    bodygraph_image: Path


class ChartQuestionRequest:
    query: str
    bodygraph_result: BodyGraphExtractionResult
```

Responsibilities:

```text
KnowledgeQuestionRequest
  -> pure knowledge Q&A

ChartImageQuestionRequest
  -> public CLI/application image entry
  -> invokes the existing Phase 2 client/parser/interpreter/validation flow first

ChartQuestionRequest
  -> Phase 3 core personalized Q&A entry
  -> receives an already parsed and validated BodyGraphExtractionResult
```

Do not collapse the core into a function accepting `bodygraph_image: Path | None`. A thin `answer_chart_image_question()` facade may accept the image request and an injectable Phase 2 extraction callable, but it must construct `ChartQuestionRequest` before entering the chart Q&A core. Core chart-pipeline tests must inject a `BodyGraphExtractionResult` and must not read an image.

## Non-Goals

Explicitly out of scope for Phase 3 v1:

- AWS
- Lambda
- API Gateway
- EC2
- SageMaker
- cloud deployment
- FastAPI
- Streamlit
- frontend
- user accounts
- database persistence beyond local generated indexes
- full chart reading
- sectioned full-reading synthesis
- BGE reranker
- multiple reranker providers
- LangChain
- Pinecone
- OpenSearch
- a new vector database
- OCR
- LlamaParse
- PDF image extraction
- retraining embeddings
- fine-tuning an LLM
- LLM-based query rewriting
- LLM-based chart-fact selection
- LLM-as-a-judge
- external moderation API calls
- prompt-injection classifier services
- arbitrary tool execution
- medical diagnosis
- mental-health diagnosis
- legal or financial decision making
- guaranteed future predictions
- deterministic death, illness, pregnancy, or compatibility claims
- presenting Human Design as scientifically validated medical guidance

## Tech Stack

- Python 3.11+: current project runtime.
- uv: dependency and command runner.
- pytest: offline unit and integration tests.
- ruff: linting.
- LlamaIndex core: existing `Document`, node, index, storage, and retrieval integration.
- `PyMuPDFReader`: existing embedded-text PDF loading from `human_design.rag.ingestion`.
- `SentenceSplitter`: existing chunking through `human_design.rag.chunking.chunk_documents()`.
- OpenAI embeddings: existing `OpenAIEmbedding` factory and explicit `HD_RAG_REAL_EMBEDDINGS=1` gate.
- Chroma: existing local vector database and LlamaIndex integration, used in a new Phase 3 index root.
- LlamaIndex BM25 retriever integration: core Phase 3 sparse-retrieval dependency.
- OpenAI Responses API: the only Phase 3 v1 generation provider, with structured output and `store=False`.
- Cohere rerank integration: optional and lazily imported.
- Existing Phase 2 models and pipeline components, especially `human_design.vision.models.BodyGraphExtractionResult`.

Do not add a second generation provider, a second vector database, or a local reranker dependency.

## Minimal Target File Map

Adapt the implementation to existing modules rather than duplicating them. The minimal target shape is:

```text
<repo-root>/
├── docs/
│   └── phase3_implementation_plan.md
├── scripts/
│   ├── build_hybrid_index.py
│   ├── ask_human_design.py
│   └── evaluate_hybrid_retrieval.py
├── src/
│   └── human_design/
│       ├── rag/
│       │   ├── config.py              # extend existing AppConfig/load_config
│       │   ├── models.py              # extend existing frozen dataclasses
│       │   ├── ingestion.py           # preserve existing PDF behavior
│       │   ├── chunking.py            # preserve current SentenceSplitter behavior
│       │   ├── embeddings.py          # reuse existing OpenAIEmbedding factory
│       │   ├── vector_store.py        # reuse existing Chroma helpers where practical
│       │   ├── retriever.py           # preserve legacy Phase 1 entry points; add dense adapter
│       │   ├── hybrid_index.py        # canonical IDs, nodes.jsonl, manifest, build coordination
│       │   ├── bm25.py                # sparse normalization, BM25 persist/reload adapter
│       │   ├── hybrid_retriever.py    # separate-query retrieval and small RRF function
│       │   ├── reranker.py            # NoOp and optional Cohere adapter
│       │   └── evaluation.py          # small retrieval metrics only
│       ├── reading/
│       │   ├── __init__.py
│       │   ├── models.py              # requests, chart facts, prompt and answer models
│       │   ├── chart_context.py        # safe BodyGraphExtractionResult projection
│       │   ├── query_builder.py        # deterministic scope/entity/fact selection and queries
│       │   ├── prompts/
│       │   │   └── human_design_answer.txt
│       │   ├── prompt.py               # template loading and PromptContext rendering
│       │   ├── generator.py            # one OpenAI Responses API path
│       │   └── pipeline.py             # orchestration and thin image facade
│       └── vision/
│           └── existing Phase 2 modules, behavior unchanged
├── tests/
│   ├── fixtures/
│   │   └── rag/
│   │       ├── hybrid_queries.example.json
│   │       └── sanitized generation JSON fixtures only if a small fixture is clearer
│   ├── test_phase3_models.py
│   ├── test_hybrid_index.py
│   ├── test_bm25.py
│   ├── test_hybrid_retriever.py
│   ├── test_hybrid_evaluation.py
│   ├── test_reranker.py
│   ├── test_chart_context.py
│   ├── test_query_builder.py
│   ├── test_reading_prompt.py
│   ├── test_generation.py
│   ├── test_reading_pipeline.py
│   └── test_ask_human_design.py
└── storage/
    ├── chroma/                         # optional legacy Phase 1 local store
    └── hybrid_v1/                      # generated and ignored
        ├── nodes.jsonl
        ├── chroma/
        ├── bm25/
        └── manifest.json
```

Do not create both `dense_retriever.py` and `retriever.py`; the existing `retriever.py` can retain its Phase 1 functions and host the small Phase 3 dense adapter. Keep RRF in `hybrid_retriever.py`, not a separate `fusion.py`. Do not create a large `safety.py` framework or one module per model. All generated `storage/` content remains ignored by Git.

## Configuration Design

Extend the existing frozen `human_design.rag.config.AppConfig` and `load_config()` behavior rather than replacing them. Preserve its current precedence of explicit mapping/process environment over `.env` over Python defaults and its `Path`-typed filesystem settings.

Phase 3 adds these conceptual environment values while retaining current source, chunking, embedding, collection, and ingestion settings:

```env
OPENAI_API_KEY=
COHERE_API_KEY=

HD_RAG_INDEX_DIR=storage/hybrid_v1

HD_RAG_DENSE_TOP_K=20
HD_RAG_SPARSE_TOP_K=20
HD_RAG_FUSION_TOP_K=20
HD_RAG_FINAL_TOP_K=4
HD_RAG_RRF_K=60

HD_RAG_RERANK_PROVIDER=none
HD_RAG_RERANK_MODEL=
HD_RAG_REAL_RERANK_API=0

HD_RAG_GENERATION_MODEL=
HD_RAG_REAL_GENERATION=0

HD_RAG_REAL_EMBEDDINGS=0
```

Do not add a configurable Phase 3 Chroma collection name. Use one fixed v1 collection name, such as `human_design_hybrid_v1`, and store it in `manifest.json`. Existing `HD_RAG_COLLECTION` continues to describe the legacy Phase 1 collection only.

Configuration must validate:

```text
dense_top_k > 0
sparse_top_k > 0
fusion_top_k > 0
final_top_k > 0
rrf_k > 0
final_top_k <= fusion_top_k
```

Use the existing strict truthy/falsey parsing conventions where practical. Empty optional model/key values normalize to `None`. Secret dataclass fields use `repr=False`; exceptions name the missing variable but never include its value.

Provider rules:

```text
rerank_provider = none
  -> no rerank API call
  -> final results are the first FINAL_TOP_K items from RRF order

rerank_provider = cohere
  -> HD_RAG_REAL_RERANK_API must equal 1
  -> COHERE_API_KEY must exist
  -> HD_RAG_RERANK_MODEL must be configured
  -> fused candidates are reranked and truncated to FINAL_TOP_K
```

Any value other than `none` or `cohere`, including `bge`, is rejected.

Generation rules:

```text
HD_RAG_REAL_GENERATION = 0
  -> no real OpenAI generation call

HD_RAG_REAL_GENERATION = 1
  -> OPENAI_API_KEY must exist
  -> HD_RAG_GENERATION_MODEL must be configured
```

Embedding rules:

```text
HD_RAG_REAL_EMBEDDINGS = 0
  -> no real ingestion embedding call
  -> no real dense-query embedding call

HD_RAG_REAL_EMBEDDINGS = 1
  -> manual hybrid ingestion and dense query retrieval may call OpenAI embeddings
```

The BM25 LlamaIndex integration is a core dependency. Cohere support is an optional dependency group and is imported lazily. Do not add BGE dependencies. Default tests require no API keys, PDFs, production Chroma/BM25 storage, or network.

## Model Principles

- Reuse the repository's frozen dataclass convention and precise type hints.
- Keep retrieval-internal provenance separate from public `AnswerResult` output.
- Never expose absolute filesystem paths in prompts, citations, or answer JSON.
- Do not attach citation IDs to all dense/sparse/fused candidates.
- Assign `S1`, `S2`, and so on only after final source selection.
- Dense, sparse, and rerank scores are not directly comparable.
- Raw dense and BM25 scores are diagnostics only; RRF uses ranks.
- Dense and sparse ranks are 1-based.
- A chunk may appear in only one retriever, so its other rank and score remain `None`.
- Use optional score/rank fields for one-sided candidates rather than sentinel zeroes.
- Avoid a separate `RerankedChunk` model.
- Avoid a separate `RerankResult` model unless implementation evidence proves it necessary.
- Use safe default factories for tuples and mappings.
- Treat `source_path` in `ChartFact` as a Phase 2 field path, not a filesystem path.
- Keep `AnswerStatus.REFUSED` reserved for explicit pipeline-level refusals or future caller-directed refusal behavior. Phase 3 v1 does not require an automatic local refusal detector.

## Storage and Privacy

- PDFs remain local under the existing configured PDF directory.
- Persisted canonical nodes contain local copyrighted source chunks.
- `nodes.jsonl`, Phase 3 Chroma, BM25, manifests, and local evaluation output are generated local artifacts.
- All `storage/` content remains ignored by Git.
- No full chunks are written outside ignored local storage.
- No full chunks appear in normal logs or CLI output.
- BodyGraph images remain local unless the user explicitly runs Phase 2 real Vision extraction.
- Phase 3 generation receives selected structured chart facts, never the image or image path.
- Name, birth date, birth time, and birth place are excluded from `ChartContext`, prompts, generation requests, and logs.
- API keys and `.env` are never committed or printed.
- Cohere reranking sends the user query and fused candidate passage text to Cohere only when explicitly enabled.
- OpenAI generation sends the user question, selected chart facts, and final selected passages only when explicitly enabled.
- OpenAI generation uses `store=False`.
- Full prompts, full retrieved passages, provider payloads, birth information, and credentials are not logged.
- Sanitized mock fixtures may be committed; private BodyGraph responses and local prediction files may not.

## Testing Strategy

For each task, use this TDD loop where practical:

1. Write a focused failing test.
2. Run it and confirm RED for the intended missing behavior, not an unrelated import or environment failure.
3. Implement the minimum required code.
4. Rerun the focused test and confirm GREEN.
5. Run relevant existing Phase 1 and Phase 2 regression tests.
6. Run `uv run ruff check .`.

Global testing rules:

- Default tests are offline and free.
- Default tests need no API keys.
- Default tests need no real PDFs.
- Default tests need no private BodyGraph images.
- Tests use temporary storage and never create repo-level production indexes.
- Tests use fake embeddings where needed.
- Tests use fake dense retrievers.
- Tests use fake BM25 retrievers.
- Tests use fake Cohere clients.
- Tests use fake OpenAI generation clients.
- Tests may use the existing sanitized `tests/fixtures/bodygraph/test1_raw_response.json` to construct Phase 2 results.
- Tests must not modify the legacy Phase 1 Chroma index.
- Tests must not log full chunks or full prompts.
- Tests must not commit generated storage.
- Relevant Phase 1 ingestion/retrieval and Phase 2 parser/interpreter/validation tests must continue to pass.
- No test may silently fall back to a real provider because a developer has keys in `.env` or the shell.

## Implementation Tasks

### Task 26: Freeze Phase 3 scope and request boundaries

Goal:
Define the focused Phase 3 v1 boundary, create the minimal `human_design.reading` package, and add typed request models without implementing retrieval, generation, or image extraction.

Files touched:
- Create `src/human_design/reading/__init__.py`.
- Create `src/human_design/reading/models.py` with only the three request models in this task.
- Create `tests/test_phase3_models.py`.

Dependencies:
- Python frozen dataclasses, matching `human_design.rag.models` and `human_design.vision.models`.
- `pathlib.Path` for the outer image request.
- Existing `human_design.vision.models.BodyGraphExtractionResult`.
- Existing Phase 2 model exports from `human_design.vision`.

Acceptance criteria:
- Phase 3 v1 is explicitly represented as focused Q&A only; these models do not imply full-reading support.
- `KnowledgeQuestionRequest` contains exactly `query: str`.
- `ChartImageQuestionRequest` contains `query: str` and `bodygraph_image: Path`.
- `ChartQuestionRequest` contains `query: str` and `bodygraph_result: BodyGraphExtractionResult`.
- `ChartQuestionRequest` imports the actual Phase 2 result from `human_design.vision.models`; no obsolete `BodyGraphPipelineResult` is introduced.
- The Phase 3 chart core receives no image `Path`.
- `ChartImageQuestionRequest` is documented as an outer entry that invokes Phase 2 before constructing `ChartQuestionRequest`.
- Knowledge and chart requests are designed to share one later retrieval/generation core.
- Models are frozen dataclasses with public type hints, consistent with current model conventions.
- Models validate obvious wrong field types in `__post_init__` if current tests establish that convention; query content/length validation remains centralized in Task 39.
- `reading.models` imports no OpenAI, Cohere, Chroma, BM25, Vision client, or CLI module.
- `reading.__init__` exports the request types without broad re-exports of provider internals.

Out of scope:
- No retrieval.
- No ingestion.
- No generation.
- No Phase 2 image extraction implementation.
- No full reading.
- No CLI.
- No configuration or dependency changes.

Testing requirements:
- First add tests that import and instantiate all three request types; confirm RED because `human_design.reading` does not exist.
- Test that all three are frozen dataclasses.
- Test exact field names and type hints.
- Test `ChartQuestionRequest` accepts a real sanitized `BodyGraphExtractionResult` instance assembled from existing Phase 2 models.
- Test that an unrelated object is rejected for `bodygraph_result` if runtime model validation is implemented.
- Test that only `ChartImageQuestionRequest` contains a `Path` field.
- Tests must not call Phase 2 Vision, OpenAI, Cohere, Chroma, or BM25.

Verification:

```sh
uv run pytest tests/test_phase3_models.py
uv run pytest tests/test_vision_models.py
uv run ruff check .
git diff --check
```

### Task 27: Add Phase 3 configuration and dependencies

Goal:
Extend the existing `AppConfig` and `load_config()` with the minimum Phase 3 index, retrieval, reranking, generation, credential, and explicit real-API settings; add only BM25 core support and optional Cohere reranking dependencies.

Files touched:
- Modify `src/human_design/rag/config.py`.
- Modify `tests/test_config.py` rather than creating a competing configuration test module.
- Modify `.env.example`.
- Modify `pyproject.toml`.
- Modify `uv.lock` through uv dependency commands only.

Dependencies:
- Existing `AppConfig`, `_load_env_values()`, integer parsing, dotenv precedence, and chunk validation.
- Existing `python-dotenv` dependency.
- Add `llama-index-retrievers-bm25` as a core project dependency, using the version compatible with the locked LlamaIndex 0.14 series.
- Add `llama-index-postprocessor-cohere-rerank` through an optional `rerank` dependency extra if that package is compatible with the installed LlamaIndex version; do not install a local reranker.
- Existing transitive OpenAI SDK used by the current Vision client and embedding integration; do not add another generation SDK/provider.

Acceptance criteria:
- Existing `AppConfig` fields and Phase 1 defaults remain supported.
- `HD_RAG_INDEX_DIR` maps to `index_dir: Path` and defaults to `storage/hybrid_v1`.
- Dense Top-K defaults to 20.
- Sparse Top-K defaults to 20.
- Fusion Top-K defaults to 20.
- Final Top-K defaults to 4.
- RRF K defaults to 60.
- `rerank_provider` normalizes whitespace/case and supports only `none` and `cohere`.
- `bge` and all other providers raise a clear `ValueError` naming `HD_RAG_RERANK_PROVIDER`.
- `HD_RAG_RERANK_MODEL` and `HD_RAG_GENERATION_MODEL` normalize blank values to `None`.
- `HD_RAG_REAL_RERANK_API`, `HD_RAG_REAL_GENERATION`, and `HD_RAG_REAL_EMBEDDINGS` use one tested strict boolean parser.
- Existing Phase 1 `HD_RAG_REAL_EMBEDDINGS=1` opt-in remains enforced by scripts; Task 31 migrates the hybrid build script to the typed flag without weakening the legacy script.
- The config validates all Top-K values and `rrf_k` are positive.
- The config validates `final_top_k <= fusion_top_k`.
- `fusion_top_k` may exceed the sum of requested dense and sparse Top-K values; fusion simply returns at most the available unique candidates.
- `rerank_provider=none` does not require a model, key, or real-rerank flag.
- `rerank_provider=cohere` requires `HD_RAG_REAL_RERANK_API=1`, non-empty `COHERE_API_KEY`, and a configured rerank model before constructing the real adapter.
- Real generation enabled requires `OPENAI_API_KEY` and a configured generation model; disabled generation does not.
- Real embeddings disabled requires no OpenAI key; manual real hybrid ingestion and dense retrieval validate the key at their call boundaries.
- `openai_api_key` and new `cohere_api_key` fields use `repr=False` so `repr(config)` cannot reveal secrets.
- No secret value is interpolated into an exception.
- `.env.example` documents all Phase 3 variables with safe defaults and empty key/model placeholders.
- Existing `.env` loading precedence remains explicit mapping/process environment, then `.env`, then defaults.
- BM25 is a core dependency.
- Cohere reranking is optional and does not affect a normal `uv sync` or default test run unless project dependency conventions require installing all extras in development.
- No BGE, FlagEmbedding, sentence-transformers, GPU, web, cloud, or moderation dependency is added.

Out of scope:
- No retriever implementation.
- No index build implementation.
- No generation implementation.
- No BGE.
- No model downloads.
- No cloud settings.
- No configurable Phase 3 collection matrix.
- No separate storage path variable for Chroma, BM25, nodes, and manifest.

Testing requirements:
- Add failing default/config override tests before extending `AppConfig`.
- Test all numeric defaults and environment overrides.
- Parameterize invalid zero, negative, and non-integer numeric values.
- Test `final_top_k > fusion_top_k` is rejected.
- Test a larger `fusion_top_k` than the combined dense/sparse request sizes is accepted and does not introduce another constraint.
- Parameterize `none`, mixed-case/whitespace `cohere`, and rejected providers including `bge`.
- Test real rerank disabled/missing key/missing model errors without importing or calling Cohere.
- Test real generation disabled/missing key/missing model errors without importing or calling OpenAI.
- Test real embeddings defaults false and parses explicit true/false values.
- Test explicit mappings remain isolated from `.env` and process environment.
- Test config `repr` excludes two distinct fake keys.
- Verify imports for the chosen BM25 package and optional Cohere package without creating clients or network calls.
- Default tests require no credentials or storage.

Verification:

```sh
uv run pytest tests/test_config.py
uv run python -c "from llama_index.retrievers.bm25 import BM25Retriever; print(BM25Retriever.__name__)"
uv run --extra rerank python -c "from llama_index.postprocessor.cohere_rerank import CohereRerank; print(CohereRerank.__name__)"
uv run pytest tests/test_embeddings.py tests/test_ingest_pdfs.py tests/test_retriever.py
uv run ruff check .
git diff --check
git status --short
```

If the compatible installed package exposes a different documented Cohere import path, update the import-check command and Task 34 together; do not guess or add a second Cohere package.

### Task 28: Define Phase 3 core models and provenance

Goal:
Add the minimum frozen typed models for hybrid retrieval provenance, normalized chart context, prompt input, public citations, and public answers while keeping internal retrieval details out of `AnswerResult`.

Files touched:
- Modify `src/human_design/rag/models.py` to add retrieval-internal models.
- Modify `src/human_design/reading/models.py` to add chart, prompt, citation, status, and answer models alongside Task 26 requests.
- Modify `src/human_design/reading/__init__.py` only for intentional public exports.
- Modify `tests/test_phase3_models.py`.

Dependencies:
- Task 26 request models.
- Existing frozen dataclass conventions.
- `dataclasses.field`, `enum.StrEnum`, `collections.abc.Mapping`, and JSON scalar typing.
- Existing `BodyGraphExtractionResult` only for `ChartQuestionRequest`; retrieval models remain independent of Vision.

Acceptance criteria:
- `HybridSearchRequest` contains `original_query`, `dense_query`, and `sparse_query` strings.
- `RetrievedChunk` contains:
  - `chunk_id: str`
  - `text: str`
  - `source_file: str`
  - `source_relpath: str | None`
  - `document_title: str | None`
  - `page_label: str | None`
  - `page_number: int | None`
  - optional `dense_rank`, `dense_score`, `sparse_rank`, `sparse_score`, `rrf_score`, and `rerank_score`
  - a JSON-safe metadata mapping with a safe default factory
- Dense and sparse ranks, when present, are integers greater than or equal to 1; booleans and zero are rejected.
- Dense-only and sparse-only chunks are representable without sentinel values.
- An unfused candidate may have no RRF score; an unreranked candidate has no rerank score.
- `HybridRetrievalResult` contains the request, ordered candidate tuple, index version/identity information, configured Top-K values, RRF K, and optional rerank provider/model provenance.
- A `HybridRetrievalResult` accepts only fused candidates with non-`None` RRF scores.
- Candidate order is preserved as deterministic retrieval provenance.
- `ChartFact` contains `field: str`, a JSON scalar `value`, and `source_path: str` identifying a Phase 2 model-field path.
- List-like chart data is represented as individual facts where practical; `ChartFact.value` does not accept arbitrary nested dictionaries/lists.
- `ChartFact.source_path` is non-empty and stable. Repeated list facts use deterministic selectors such as `derived_chart_data.active_gates[gate=42]` so selected facts can be validated unambiguously.
- `ChartContext` contains `facts: tuple[ChartFact, ...]` and `validation_warnings: tuple[str, ...]`.
- `ChartContext` rejects duplicate `source_path` values; a generation-reported fact path must identify exactly one fact.
- `PromptSource` contains `citation_id` and one final `RetrievedChunk`.
- `PromptContext` contains only `user_question`, selected chart facts, and final prompt sources.
- `PromptContext` contains no BodyGraph image path and no absolute filesystem path.
- A safe internal `source_relpath` may remain on the `RetrievedChunk` nested inside a `PromptSource` for retrieval provenance, but it must never be rendered into the generation prompt or copied into a public `SourceCitation`.
- `PromptContext` has no image, raw provider payload, Top-20 list, key, provider config, or full Phase 2 result field.
- `SourceCitation` contains only `citation_id`, `chunk_id`, `source_file`, optional `document_title`, optional `page_label`, and optional `page_number`.
- `SourceCitation` cannot contain absolute path, chunk text, or retrieval/rerank scores.
- `AnswerStatus` has exactly `ok`, `insufficient_evidence`, `invalid_chart`, `needs_focus`, `refused`, and `error` values.
- Use uppercase enum member names such as `AnswerStatus.OK`, `AnswerStatus.INSUFFICIENT_EVIDENCE`, and `AnswerStatus.REFUSED` with the lowercase serialized values above, matching existing project `StrEnum` conventions; `AnswerStatus.OK` serializes as `"ok"`.
- `AnswerResult` contains `status`, `answer_markdown`, citations, chart facts used, and warnings.
- All tuple/mapping defaults are safe and independent.
- Do not add `RerankedChunk`, `RerankResult`, a second answer hierarchy, or one file per model.

Out of scope:
- No canonical node persistence.
- No retrieval or RRF implementation.
- No reranking implementation.
- No prompt rendering.
- No generation API.
- No full-reading models.

Testing requirements:
- Write model tests first and confirm RED for missing types.
- Test frozen dataclass behavior and public type hints.
- Test dense-only, sparse-only, and dual-retriever rank/score combinations.
- Parameterize invalid ranks including 0, negative, float, string, and boolean.
- Test a fused result rejects a candidate missing `rrf_score`.
- Test metadata values are JSON-safe and independent default dictionaries do not share state.
- Test independent empty tuple/mapping defaults.
- Test `ChartFact` accepts string, integer, float, boolean, and `None` JSON scalars and rejects nested containers.
- Test every answer status.
- Test `dataclasses.asdict()` plus `json.dumps()` works for representative public output, converting `StrEnum` to its string value through the existing `StrEnum` behavior or an explicit serializer.
- Test public citation/model fields cannot expose an absolute filesystem path.
- Tests use no storage or provider clients.

Verification:

```sh
uv run pytest tests/test_phase3_models.py tests/test_models.py
uv run ruff check .
git diff --check
```

### Task 29: Add canonical node persistence and stable hybrid index identity

Goal:
Create deterministic canonical chunk identity, persist canonical nodes as `nodes.jsonl`, and define one shared Phase 3 hybrid-ingestion manifest used by Chroma and BM25.

Files touched:
- Create `src/human_design/rag/hybrid_index.py`.
- Create `tests/test_hybrid_index.py`.
- Modify `.gitignore` only if `storage/` is no longer ignored; the current repository already ignores all `storage/`, so no change should be needed.

Dependencies:
- Existing LlamaIndex `BaseNode`/`TextNode` types.
- Existing `chunk_documents()` output and metadata.
- Standard library `hashlib`, `json`, `unicodedata`, `dataclasses`, and `pathlib.Path`.
- Existing `AppConfig` source, chunk, embedding, and ingestion settings.
- Task 27 `index_dir` configuration.

Acceptance criteria:
- `hybrid_index.py` defines focused typed structures or functions for source fingerprinting, canonical node creation, JSONL persistence/reload, manifest construction, and manifest read/write.
- Source fingerprints are SHA-256 hashes of file bytes, never filename-only hashes.
- PDF discovery/order is deterministic by reusing current sorted discovery.
- Before canonical chunk creation, duplicate source byte fingerprints are detected across the discovered PDFs.
- If two or more different source files have the same SHA-256 source fingerprint, ingestion fails with a clear file-level error naming only their safe relative source identities; the error never exposes absolute local paths.
- Duplicate source files fail at this fingerprint precheck rather than waiting for duplicate `chunk_id` detection. The existing duplicate canonical `chunk_id` failure remains a separate later safeguard.
- Existing embedded-text extraction and `chunk_documents()` semantics are unchanged.
- Every canonical node receives the deterministic shared `chunk_id` algorithm specified above.
- The canonical LlamaIndex node ID is explicitly set to that same `chunk_id` before either index is built.
- Chunk ordinals are deterministic and zero-based within a documented source-page key.
- Node text and useful current metadata are preserved, including source filename, document title/page metadata where available, chunk settings, embedding model, and ingestion version.
- Canonical metadata adds `chunk_id`, `source_sha256`, and a safe `source_relpath` suitable for later public provenance.
- Absolute `source_path` may remain only in ignored local canonical storage for ingestion diagnostics; retrieval adapters must never copy it into `SourceCitation` or prompts.
- `nodes.jsonl` has one compact UTF-8 JSON object per node with `chunk_id`, normalized text, and JSON-safe metadata.
- Loading `nodes.jsonl` reconstructs nodes without reading PDFs or embedding text.
- Duplicate IDs fail clearly before persistence/index construction.
- The shared root manifest includes at least:
  - `schema_version`
  - `ingestion_id`
  - `ingestion_version`
  - `corpus_fingerprint`
  - ordered source fingerprints with safe relative identities
  - `chunk_count`
  - `chunk_size`
  - `chunk_overlap`
  - `embedding_model`
  - fixed Phase 3 collection name
  - exact `chunk_id_algorithm`
- `corpus_fingerprint` is derived deterministically from the ordered source fingerprints and canonical node IDs.
- `ingestion_id` identifies the complete manifest/corpus contract without circular hashing. Compute it from a canonical identity payload that excludes the `ingestion_id` field itself:

  ```python
  identity_payload = {
      "schema_version": schema_version,
      "ingestion_version": ingestion_version,
      "corpus_fingerprint": corpus_fingerprint,
      "source_fingerprints": source_fingerprints,
      "chunk_count": chunk_count,
      "chunk_size": chunk_size,
      "chunk_overlap": chunk_overlap,
      "embedding_model": embedding_model,
      "collection_name": collection_name,
      "chunk_id_algorithm": chunk_id_algorithm,
  }

  ingestion_id = sha256(
      canonical_json(identity_payload).encode("utf-8")
  ).hexdigest()

  manifest = {
      **identity_payload,
      "ingestion_id": ingestion_id,
  }
  ```

- `canonical_json()` uses deterministic dictionary-key ordering, compact separators, UTF-8, and an explicitly ordered `source_fingerprints` list; no dictionary or set iteration order is left implicit.
- The same identity payload always produces the same `ingestion_id`.
- Changing any identity-payload field changes `ingestion_id`.
- The final manifest stores every identity-payload field plus the computed `ingestion_id`.
- Manifest reload recomputes the hash from the stored identity fields and rejects a mismatched `ingestion_id` deterministically.
- `ingestion_id` uses no UUID, timestamp, database ID, random build value, or signing key.
- Manifest chunk count must equal canonical node count.
- Writes create only the configured temporary/test index root; tests never touch repository `storage/`.
- Existing non-empty output fails clearly rather than silently overwriting canonical nodes.
- Production PDF text is never added to a committed fixture.

Out of scope:
- No Chroma build.
- No BM25 build or retrieval.
- No query retrieval.
- No RRF.
- No generation.
- No PDF extraction or chunk-size semantic changes.
- No overwrite/delete of the legacy Phase 1 index.
- No complex migration or atomic deployment system.

Testing requirements:
- Write deterministic-ID tests first and confirm the failure is the absent module/API.
- Use small synthetic `TextNode` objects and fake source bytes in `tmp_path`.
- Verify identical inputs produce identical IDs and byte-identical `nodes.jsonl`/manifest content.
- Verify changed normalized text, source bytes/fingerprint, page key, and chunk ordinal each change the ID.
- Verify duplicate PDF byte fingerprints fail before canonical chunk creation.
- Verify the duplicate-source error names the duplicate safe relative source identities.
- Verify the duplicate-source error does not expose absolute paths.
- Verify CRLF/CR normalization and Unicode NFC behavior explicitly.
- Verify delimiter-like characters in inputs cannot collide because canonical JSON serialization is used.
- Verify duplicate IDs fail with a message naming duplicate canonical IDs.
- Verify JSONL round-trip preserves IDs, text, and JSON-safe metadata.
- Verify malformed JSONL and manifest count mismatch fail clearly.
- Verify `ingestion_id` is absent from its own hash input and is added only after the identity hash is computed.
- Verify dictionary key ordering and preordered list serialization produce byte-identical canonical identity JSON across repeated calls.
- Verify the same identity payload produces the same `ingestion_id` and changing each identity field changes it.
- Verify manifest reload recomputes and accepts the correct identity and rejects a tampered identity field or stored `ingestion_id`.
- Verify reload performs no PDF, embedding, Chroma, or network calls.
- Verify no absolute path is present in the safe public metadata projection.

Verification:

```sh
uv run pytest tests/test_hybrid_index.py
uv run pytest tests/test_ingestion.py tests/test_chunking.py
uv run ruff check .
git diff --check
```

### Task 30: Build and reload the persisted BM25 index

Goal:
Build BM25 from the exact canonical nodes, persist it under the Phase 3 index root, reload it without PDFs or embeddings, convert sparse results into shared `RetrievedChunk` models, and deterministically normalize exact Human Design terms.

Files touched:
- Create `src/human_design/rag/bm25.py`.
- Create `tests/test_bm25.py`.

Dependencies:
- Task 27 BM25 dependency and configuration.
- Task 29 canonical node persistence and shared manifest identity.
- Installed `llama_index.retrievers.bm25.BM25Retriever` API; inspect the locked version before coding.
- Existing canonical center names from `human_design.vision.constants` only if importing them does not create a Vision client dependency.
- Standard library regex and small explicit alias tables.

Acceptance criteria:
- Implement the conceptual APIs:

  ```python
  build_and_persist_bm25(nodes, persist_dir, ingestion_id)
  load_bm25_retriever(
      persist_dir,
      expected_ingestion_id,
  ) -> BM25RetrieverAdapter
  normalize_sparse_query(query)


  class BM25RetrieverAdapter:
      ingestion_id: str

      def retrieve(
          self,
          query: str,
          top_k: int,
      ) -> list[RetrievedChunk]:
          ...
  ```

- The exact constructor details may follow the installed LlamaIndex API, but Task 30 owns this one small sparse adapter. Do not defer result conversion to Task 32 or create a retriever hierarchy.
- BM25 is built from the exact in-memory canonical nodes loaded/persisted by Task 29.
- The loaded adapter exposes its validated `ingestion_id` as a simple read-only value.
- Each BM25 result is converted into a `RetrievedChunk` with the canonical node ID preserved as `chunk_id`.
- `sparse_rank` is 1-based and reflects BM25 result order.
- `sparse_score` preserves the raw BM25 score without normalization.
- `dense_rank` and `dense_score` are `None`.
- `rrf_score` and `rerank_score` are `None`.
- Full chunk text is preserved internally in `RetrievedChunk.text` for later fusion, reranking, and prompt selection.
- Safe source filename, relative identity, title, page label, page number, and JSON-safe metadata are preserved where available.
- Absolute source paths and private local path metadata are excluded from the returned `RetrievedChunk` fields and metadata projection.
- BM25 uses the documented persistence API of the installed integration; its internal persisted filenames are treated as opaque.
- A small project-owned identity sidecar inside the BM25 directory, or an equivalent documented integration metadata mechanism, records `ingestion_id` and chunk count without relying on library-internal filenames.
- Loading compares the supplied expected ingestion ID and fails clearly on mismatch.
- Reading the root `manifest.json` is the caller's responsibility in the Task 31 build/verification flow and the Task 40 pipeline.
- Task 30 loading does not read the root manifest directly. It validates the supplied `expected_ingestion_id` against the BM25 directory's own identity sidecar and validates the sidecar's `ingestion_id` and chunk count against the loaded BM25 corpus where supported by the installed integration.
- Keep `load_bm25_retriever(persist_dir, expected_ingestion_id)` unchanged; do not add a root-manifest parameter or make the loader discover `manifest.json` itself.
- Reload reads neither PDFs nor `nodes.jsonl` unless the installed BM25 API explicitly requires nodes; if nodes are required, reload uses canonical `nodes.jsonl`, never PDFs, and documents that behavior.
- Reload and sparse retrieval do not read PDFs, create embeddings, or call a network provider.
- Sparse query normalization is whitespace/case tolerant and preserves the original useful words while appending canonical exact terms deterministically without duplicate tokens/phrases.
- Sparse normalization translates only a small explicit set of Chinese Human Design aliases into canonical English terms because the Phase 3 v1 BM25 corpus is assumed to be primarily English.
- Regex normalization supports Gate expressions such as `Gate 42`, `42nd Gate`, `Hexagram 42`, `42號閘門`, and `42閘門`.
- Regex normalization supports Channel expressions such as `Channel 42-53`, `Channel 53-42`, `42 - 53`, and `42號與53號通道`; canonical output includes both endpoint orderings where useful.
- Regex normalization supports Profile expressions such as `Profile 4/6`, `4-6 Profile`, and `4/6人生角色`.
- A small explicit alias map covers canonical Human Design Types, Authorities, Centers, planetary terms, and the required Chinese aliases such as `薦骨權威`, `情緒權威`, and `脾臟權威`.
- `Emotional Authority` normalizes with `Solar Plexus Authority`; `Splenic Authority` and `Sacral Authority` remain distinct.
- Natural canonical phrases are emitted, for example `Gate 42 Hexagram 42` and `Profile 4/6 Profile 4 6`.
- No synthetic document/query tokens such as `gate_42`, `channel_42_53`, or `profile_4_6` are added in v1.
- Unknown ordinary words remain available to BM25 rather than being discarded.

Out of scope:
- No dense retrieval.
- No RRF.
- No reranking.
- No generation.
- No document-side lexical enrichment.
- No Chinese tokenizer, Chinese document segmentation, or complete Chinese translation system.
- No production dense-only fallback.

Testing requirements:
- Write normalization and persistence/reload tests first and confirm RED for the missing module.
- Parameterize required English punctuation, reverse-channel, spacing, and Chinese examples.
- Verify output ordering is deterministic and repeated normalization is idempotent enough for stable query construction.
- Build BM25 from a few synthetic canonical nodes under `tmp_path`, persist, reload, and retrieve an exact term.
- Assert the Task 30 adapter exposes the validated `ingestion_id`.
- Assert returned `RetrievedChunk` objects preserve canonical IDs, full text, safe source metadata, raw BM25 scores, and 1-based sparse ranks.
- Assert dense rank/score, RRF score, and rerank score remain `None`.
- Assert absolute source paths are removed from returned fields and metadata.
- Verify wrong ingestion ID fails before retrieval.
- Verify loading uses only the supplied expected ingestion ID and the BM25 directory's identity sidecar and does not read the root `manifest.json` directly.
- Monkeypatch Phase 1 PDF loaders, the OpenAI embedding factory, and any provider boundary to raise if called during reload/retrieval.
- Tests require no key, network, real PDF, or repository storage.

Verification:

```sh
uv run pytest tests/test_bm25.py tests/test_hybrid_index.py
uv run pytest tests/test_chunking.py tests/test_bodygraph_constants.py
uv run ruff check .
git diff --check
```

### Task 31: Build the new Phase 3 Chroma index and add a dense retriever adapter

Goal:
Build the new Phase 3 Chroma index from the same canonical nodes as BM25, complete the one-time hybrid build entry point, and expose dense query retrieval as shared `RetrievedChunk` models without touching the legacy Phase 1 collection.

Files touched:
- Modify `src/human_design/rag/hybrid_index.py` for coordinated build/identity checks.
- Modify `src/human_design/rag/retriever.py` to add a Phase 3 dense adapter while preserving all existing Phase 1 functions.
- Modify `src/human_design/rag/vector_store.py` only for a small reusable metadata/strict-open helper if the current helpers cannot enforce Phase 3 ingestion identity.
- Create `scripts/build_hybrid_index.py`.
- Modify `tests/test_hybrid_index.py`.
- Modify `tests/test_retriever.py` for the dense adapter and Phase 1 regression behavior.
- Create a focused script test in `tests/test_hybrid_index.py` rather than a new test module unless that file becomes unclear.

Dependencies:
- Tasks 27, 29, and 30.
- Existing `load_config()`, `load_pdfs()`, `build_text_extraction_report()`, and `chunk_documents()`.
- Existing OpenAI embedding factory.
- Existing `create_chroma_vector_store()`/`open_existing_chroma_vector_store()` behavior where compatible.
- Existing LlamaIndex `StorageContext` and `VectorStoreIndex` pattern from `scripts/ingest_pdfs.py`.
- Task 28 `RetrievedChunk`.

Acceptance criteria:
- The script performs this exact high-level sequence:

  ```text
  load existing Phase 1 PDF configuration
    -> discover/load PDFs in deterministic order
    -> reuse embedded-text extraction reporting
    -> reuse current chunk settings
    -> compute source byte fingerprints
    -> fail on duplicate source byte fingerprints
    -> create deterministic canonical nodes
    -> persist nodes.jsonl
    -> build new Chroma from those nodes
    -> build persisted BM25 from those same in-memory nodes
    -> write/validate the shared manifest
    -> verify dense and sparse ingestion identity and chunk count
  ```

- The Phase 3 Chroma directory is `<HD_RAG_INDEX_DIR>/chroma`.
- The Phase 3 collection name is fixed for v1 and stored in the root manifest.
- The script never uses `AppConfig.chroma_dir`/legacy `HD_RAG_CHROMA_DIR` as the Phase 3 target.
- The legacy Phase 1 Chroma index is neither queried, appended, deleted, reset, nor migrated.
- Chroma and BM25 receive the same canonical node objects/IDs from one build run.
- Chroma collection metadata records at least embedding model, ingestion ID, schema/index version, corpus fingerprint, and chunk count.
- The root manifest is written with matching collection name and identities.
- A post-build identity check compares root manifest, Chroma metadata/count, BM25 identity/count, and `nodes.jsonl` count.
- A non-empty `HD_RAG_INDEX_DIR` causes a clear error instructing the developer to remove it intentionally; the script does not delete it.
- Running without `HD_RAG_REAL_EMBEDDINGS=1` exits before loading PDFs or creating storage and makes no OpenAI call.
- Running with real embeddings enabled requires `OPENAI_API_KEY` and warns through documentation/CLI text that embedding cost may occur, without printing the key.
- The script exposes no incremental append mode in v1.
- The dense adapter conceptually provides:

  ```python
  class DenseRetrieverAdapter:
      ingestion_id: str

      def retrieve(
          self,
          query: str,
          top_k: int,
      ) -> list[RetrievedChunk]:
          ...
  ```

  Keep this as one small adapter around an already-open retriever so retrieval behavior can be injected in tests; do not create a dense retriever hierarchy.
- The dense adapter exposes its validated `ingestion_id` as the same simple read-only value used by the Task 30 sparse adapter; do not add a separate identity model or copy the full manifest into either adapter.
- Dense retrieval opens only the Phase 3 Chroma root/collection described by `manifest.json`.
- Dense loading enforces the configured embedding model, manifest ingestion ID, corpus fingerprint, and collection metadata consistency.
- Dense querying never appends vectors or silently rebuilds.
- Query embeddings remain explicitly gated by `HD_RAG_REAL_EMBEDDINGS=1` for real CLI use.
- Dense results preserve canonical chunk ID, safe source metadata, raw dense score, and 1-based dense rank.
- Dense results leave sparse rank/score, RRF score, and rerank score as `None`.
- Absolute source paths are excluded from the returned `RetrievedChunk` public fields and metadata projection.
- Empty/missing/mismatched Phase 3 Chroma produces a clear index error; there is no legacy fallback.

Out of scope:
- No hybrid fusion.
- No reranking.
- No generation.
- No silent build during retrieval.
- No incremental update or migration.
- No overwrite of non-empty Phase 3 or legacy Phase 1 indexes.

Testing requirements:
- Write failing coordinated-build and dense-adapter tests before implementation.
- Use fake PDF documents/nodes, fake embeddings, fake Chroma/vector objects, and a real temporary BM25 only where it remains fast/offline.
- Verify the same canonical node IDs are passed to both Chroma and BM25 builders.
- Verify a different list/order passed to either builder is caught by the identity check.
- Verify disabled real embeddings exits before PDF loading, directory creation, or embedding construction.
- Verify enabled mode without a key fails before creating provider clients.
- Verify non-empty target root fails without deleting contents.
- Verify returned dense ranks start at 1 and preserve scores/source metadata.
- Verify the dense adapter exposes the manifest-validated `ingestion_id`.
- Verify the adapter rejects a Chroma/manifest ingestion mismatch.
- Verify no append/delete API is called during dense retrieval.
- Preserve all existing `tests/test_retriever.py` Phase 1 expectations.
- No test accesses repository PDFs, `storage/chroma`, `storage/hybrid_v1`, `.env`, or network.

Verification:

```sh
uv run pytest tests/test_hybrid_index.py tests/test_retriever.py
uv run pytest tests/test_ingestion.py tests/test_chunking.py tests/test_vector_store.py tests/test_embeddings.py tests/test_ingest_pdfs.py
uv run ruff check .
git diff --check
```

Do not run the real build command during default verification.

### Task 32: Implement separate-query hybrid retrieval and RRF fusion

Goal:
Execute dense and sparse retrieval with different query strings, merge candidates by canonical `chunk_id`, and return deterministic fusion Top-K candidates ranked by RRF.

Files touched:
- Create `src/human_design/rag/hybrid_retriever.py`.
- Create `tests/test_hybrid_retriever.py`.

Dependencies:
- Task 28 `HybridSearchRequest`, `RetrievedChunk`, and `HybridRetrievalResult`.
- Task 30 BM25 adapter.
- Task 31 dense adapter.
- Task 27 Top-K/RRF configuration.
- Standard library protocols or callables for injectable retrievers.

Acceptance criteria:
- The public hybrid adapter accepts one `HybridSearchRequest` and configured dense/sparse/fusion Top-K plus RRF K.
- It sends `request.dense_query` only to the dense retriever.
- It sends `request.sparse_query` only to the BM25 retriever.
- It never substitutes `original_query` for both queries.
- It requests dense Top-K and sparse Top-K independently.
- It compares `dense_adapter.ingestion_id` and `sparse_adapter.ingestion_id` before issuing either retrieval call and fails clearly when they differ.
- It preserves the existing index-version checks without introducing a `RetrieverIdentity`, `IndexIdentity`, provider hierarchy, or full-manifest copy in each adapter.
- Candidates are deduplicated only by shared canonical `chunk_id`.
- A dual-hit candidate merges ranks, raw diagnostic scores, text, and safe metadata into one `RetrievedChunk`.
- Conflicting text or public source metadata for the same `chunk_id` raises an index consistency error rather than silently choosing one.
- A dense-only or sparse-only candidate remains eligible.
- Ranks are 1-based.
- Missing rank contributes zero.
- RRF uses exactly `1 / (rrf_k + rank)` per present retriever.
- Dense and BM25 raw scores are never normalized, multiplied, or added to the fusion score.
- Every fused candidate has an `rrf_score`.
- Results sort by descending RRF score, then ascending best available individual rank, then lexical `chunk_id`.
- Fusion output is capped at `fusion_top_k`.
- If fewer unique candidates are available than `fusion_top_k`, return all available candidates without treating the short result as a configuration error.
- `HybridRetrievalResult` records request, candidates, identity/version, Top-K settings, RRF K, and no reranker provenance yet.
- Missing BM25, missing Chroma, or identity mismatch is a clear production error.
- Production never silently falls back to dense-only or BM25-only.
- Dense-only and BM25-only modes exist only as explicit evaluation paths in Task 33, not as production recovery behavior.
- No LlamaIndex fusion helper may rewrite or expand the query.

Out of scope:
- No reranking.
- No generation.
- No chart-fact selection.
- No LLM query expansion.
- No score normalization or weighted score fusion.
- No production one-retriever fallback.

Testing requirements:
- Write fake-retriever tests first and confirm RED for the missing hybrid module.
- Assert exact query strings and Top-K values received by each fake retriever.
- Verify exact RRF values for dual-hit, dense-only, and sparse-only candidates.
- Verify deduplication produces one dual-provenance candidate.
- Verify one-sided candidates retain `None` for missing rank/score fields.
- Verify raw scores do not affect order when ranks are fixed.
- Verify stable ties use best rank then chunk ID.
- Verify result truncation occurs after fusion.
- Verify a `fusion_top_k` larger than the available unique candidates returns the available candidates without error.
- Verify text/metadata conflict and ingestion mismatch fail before returning output.
- Verify an ingestion-ID mismatch is detected before either fake retriever receives a query.
- Verify a missing/raising BM25 adapter propagates a clear error and does not return dense-only output.
- Tests make no Chroma, BM25 persistence, embedding, or network call.

Verification:

```sh
uv run pytest tests/test_hybrid_retriever.py
uv run pytest tests/test_bm25.py tests/test_retriever.py
uv run ruff check .
git diff --check
```

### Task 33: Add a small hybrid retrieval regression evaluation

Goal:
Create a small manually labeled English/Chinese evaluation set and deterministic offline metrics that compare dense-only, BM25-only, and RRF hybrid retrieval.

Files touched:
- Create `src/human_design/rag/evaluation.py`.
- Create `scripts/evaluate_hybrid_retrieval.py`.
- Create `tests/fixtures/rag/hybrid_queries.example.json`.
- Create `tests/test_hybrid_evaluation.py`.

Dependencies:
- Tasks 30 through 32.
- Task 28 retrieval models.
- Standard library JSON/argparse/path handling.
- Existing evaluation-script convention in `scripts/evaluate_bodygraph_extraction.py` for pure metric code versus CLI formatting.

Acceptance criteria:
- The committed fixture contains approximately 10 to 15 focused, sanitized English and Chinese queries, including Generator, Sacral Authority, Gate 42, Channel 42-53, Profile 4/6, Split Definition, Solar Plexus Center, `情緒權威`, `42號閘門`, and `4/6人生角色`.
- Labels contain no full book chunks, private chart facts, absolute paths, keys, or provider responses.
- Labels identify expected source/page metadata and/or expected canonical entity literals; fixed chunk IDs are used only when tied to a reproducibly fixed corpus.
- Metric functions implement only:
  - Hit@5
  - MRR@20
  - exact entity hit rate
  - expected page/source hit rate
- Ranks for MRR are 1-based; no result contributes zero reciprocal rank.
- Metric aggregation is deterministic and clearly separates per-query from aggregate values.
- Evaluation reports three modes separately: dense-only, BM25-only, and RRF hybrid.
- Dense-only calls only the dense adapter with the dense query.
- BM25-only calls only the sparse adapter with the sparse query.
- Hybrid calls Task 32 RRF with both purpose-specific queries.
- The script reads an existing Phase 3 index and evaluation labels; it never builds or mutates indexes.
- Real dense evaluation is opt-in through `HD_RAG_REAL_EMBEDDINGS=1` because query embeddings may incur cost.
- Offline unit tests supply synthetic ranked outputs and call no provider/storage.
- The evaluation's purpose is regression detection and query-normalization feedback, not architecture selection or generation quality scoring.
- Output includes query/mode metrics and mode-level aggregates in human-readable form; a minimal `--json` flag may be added only if it matches existing CLI conventions and has tests.
- CLI arguments remain minimal and explicit, for example `--queries`, with `--index-dir` omitted in favor of `AppConfig.index_dir` unless a temporary local override is needed for tests.
- The implemented manual command is:

  ```sh
  HD_RAG_REAL_EMBEDDINGS=1 \
  uv run python scripts/evaluate_hybrid_retrieval.py \
    --queries tests/fixtures/rag/hybrid_queries.example.json
  ```

  It evaluates dense-only, BM25-only, and RRF hybrid modes in one run. If `--json` is implemented, it only changes output formatting.

Out of scope:
- No large benchmark.
- No nDCG.
- No graded relevance UI.
- No automatic label generation.
- No LLM judge.
- No generation evaluation.
- No reranker benchmark framework.

Testing requirements:
- Write metric tests first using tiny in-memory ranked lists.
- Verify Hit@5 boundary behavior at ranks 5 and 6.
- Verify MRR@20 boundary behavior and missing hits.
- Verify entity matching uses normalized exact entity labels rather than accidental substring collisions such as Gate 4 matching Gate 42.
- Verify expected source/page matches use safe public metadata.
- Verify all three modes are reported independently.
- Verify fixture schema errors name the case/query.
- Verify default CLI tests inject fake retrievers and do not open Chroma/BM25 or embed queries.
- Verify no full chunk text is printed by normal evaluation output.

Verification:

```sh
uv run pytest tests/test_hybrid_evaluation.py
uv run pytest tests/test_bm25.py tests/test_hybrid_retriever.py
uv run python scripts/evaluate_hybrid_retrieval.py --help
uv run ruff check .
git diff --check
```

### Task 34: Add minimal optional Cohere reranking

Goal:
Add one small reranker boundary with `NoOp` and optional Cohere implementations, preserving RRF provenance and selecting by rerank rank rather than a score threshold.

Files touched:
- Create `src/human_design/rag/reranker.py`.
- Create `tests/test_reranker.py`.
- Modify `src/human_design/rag/config.py` or `tests/test_config.py` only if Task 27 validation proves incomplete.

Dependencies:
- Task 27 optional Cohere integration and config gates.
- Task 28 `RetrievedChunk`.
- Task 32 fused candidates.
- `typing.Protocol`/`collections.abc.Sequence`.

Acceptance criteria:
- Define the minimal protocol:

  ```python
  class Reranker(Protocol):
      def rerank(
          self,
          query: str,
          chunks: Sequence[RetrievedChunk],
          top_n: int,
      ) -> list[RetrievedChunk]: ...
  ```

- `NoOpReranker` preserves RRF order and truncates to `top_n`.
- Cohere reranking receives the semantic dense query, not the sparse query or a chart dump.
- Cohere receives only fused candidate text needed for the rerank call.
- Provider results map by returned candidate index back to existing canonical chunks/IDs.
- Cohere adds optional `rerank_score` while retaining dense rank/score, sparse rank/score, and RRF score.
- No `RerankedChunk` or `RerankResult` model is created.
- Selection uses provider rank/order and `top_n`; there is no fixed score threshold.
- Real Cohere construction/call requires provider `cohere`, `HD_RAG_REAL_RERANK_API=1`, configured model, and `COHERE_API_KEY`.
- Provider `none` never imports or constructs a Cohere client.
- The Cohere SDK/integration import is lazy and produces an actionable optional-extra message when unavailable.
- Unknown, duplicate, or out-of-range provider result indexes fail clearly.
- Empty input returns empty output without a provider call.
- The output count is at most `HD_RAG_FINAL_TOP_K`.
- Candidate text, keys, and full provider responses are not logged.
- Manual quality/latency comparison reuses Task 33 labels/results; no benchmark framework is added.

Out of scope:
- No BGE.
- No local model download.
- No GPU inference.
- No additional remote provider.
- No threshold filtering.
- No complex retries or fallbacks.

Testing requirements:
- Write NoOp and fake-Cohere tests first.
- Verify NoOp makes no provider import/call and preserves exact RRF order.
- Use a fake client to return reordered indexes/scores and verify correct chunk mapping/provenance.
- Verify dense query is sent and sparse query is not.
- Verify only candidate text and required request parameters are passed.
- Test empty candidates, top_n truncation, invalid indexes, and duplicate indexes.
- Test disabled flag, missing model, missing key, and missing optional package errors without network.
- Confirm fake key and full chunk payload do not appear in exception text or captured logging.

Verification:

```sh
uv run pytest tests/test_reranker.py tests/test_config.py
uv run pytest tests/test_hybrid_retriever.py tests/test_hybrid_evaluation.py
uv run ruff check .
git diff --check
```

### Task 35: Build the safe Phase 2 chart adapter and validation gate

Goal:
Convert the actual Phase 2 `BodyGraphExtractionResult` into a safe normalized `ChartContext`, block invalid charts before retrieval, and preserve field-level provenance for allowed chart facts.

Files touched:
- Create `src/human_design/reading/chart_context.py`.
- Create `tests/test_chart_context.py`.
- Modify `src/human_design/reading/__init__.py` only if the adapter is intentionally public.

Dependencies:
- Task 28 `ChartFact` and `ChartContext`.
- Existing `human_design.vision.models.BodyGraphExtractionResult`, `Activation`, and validation types.
- Existing canonical `CANONICAL_CENTERS` ordering from `human_design.vision.constants`.
- Existing sanitized Phase 2 raw JSON fixture/parser/interpreter/validation helpers for tests.

Acceptance criteria:
- Implement:

  ```python
  build_chart_context(
      bodygraph_result: BodyGraphExtractionResult,
  ) -> ChartContext
  ```

- `bodygraph_result.validation_result.is_valid is False` produces one focused typed `InvalidChartError` (or an equally small typed outcome) before query construction/retrieval; it does not return a partially trusted personalized context.
- Valid results include individual `ChartFact` objects for:
  - `derived_chart_data.basic_info.type`
  - `authority`
  - `profile`
  - `strategy`
  - `definition`
  - `not_self_theme`
  - `signature`
  - each active Gate
  - each active Channel
  - each defined Center
  - each deterministically computed undefined Center when useful
  - each non-`None` normalized Personality activation
  - each non-`None` normalized Design activation
- Active Gates are ordered numerically; Channels follow canonical Phase 2 channel order; Centers follow `CANONICAL_CENTERS`; activations follow the Phase 2 canonical planetary field order.
- Activation values are represented in a stable readable form with exact source paths, for example `Design Earth 42.6` and `raw_vision.design.earth`.
- Basic-info facts use exact scalar paths. List-member provenance uses stable selectors such as `derived_chart_data.active_gates[gate=42]`, `derived_chart_data.active_channels[channel=42-53]`, and `derived_chart_data.defined_centers[center=Sacral]`.
- Computed undefined-center facts use a path rooted in the deterministic source field, such as `derived_chart_data.defined_centers[complement_center=Head]`, making the complement operation explicit without implying visual evidence.
- Undefined Centers are the ordered canonical set difference from deterministic `derived_chart_data.defined_centers`, never the visually undefined list.
- Validation warning strings are minimal and deterministic, using code and a safe short message where relevant.
- Non-fatal visual disagreement warnings may be carried as warnings, but visual-only gates/channels/centers never become `ChartFact` chart truth.
- `ChartContext` contains no image bytes, image path, base64, raw response string, provider reasoning/payload, confidence object, visual-only fact lists, person name, birth date, birth time, birth place, or unrelated metadata.
- The adapter reads Phase 2 models only; it does not reparse raw JSON or reimplement interpreter rules.
- ChartContext serialization does not contain `visually_active_gates`, `visible_colored_channels`, `visually_defined_centers`, confidence fields, or uncertain observations.

Out of scope:
- No question-specific fact selection.
- No retrieval.
- No generation.
- No Phase 2 parser/interpreter changes.
- No chart calculation from birth information.
- No reading generation.

Testing requirements:
- Write adapter tests first and confirm RED because `chart_context.py` is absent.
- Build a full `BodyGraphExtractionResult` from the existing sanitized fixture using the real parser, interpreter, and validation functions; do not call Vision.
- Verify valid charts produce deterministic facts and exact source paths.
- Verify `Design Earth 42.6` is representable as a fact when present in the sanitized fixture/result.
- Verify Personality and Design facts remain distinguishable.
- Verify invalid validation blocks before any fake retrieval callback can be invoked.
- Verify undefined Centers use canonical constants and deterministic derived centers only.
- Serialize context and assert excluded raw/visual/confidence/PII field names and a fake absolute path are absent.
- Tests require no image, key, network, Chroma, or BM25.

Verification:

```sh
uv run pytest tests/test_chart_context.py
uv run pytest tests/test_bodygraph_parser.py tests/test_bodygraph_interpreter.py tests/test_bodygraph_validation.py tests/test_vision_models.py
uv run ruff check .
git diff --check
```

### Task 36: Build deterministic chart-aware dense and sparse queries

Goal:
Build separate dense and sparse retrieval queries and select only the current question's relevant chart facts using deterministic entity extraction and small explicit intent rules.

Files touched:
- Create `src/human_design/reading/query_builder.py`.
- Create `tests/test_query_builder.py`.

Dependencies:
- Task 28 `HybridSearchRequest`, `ChartContext`, and `ChartFact`.
- Task 30 sparse normalization patterns/alias tables; reuse a small shared normalization function rather than duplicating incompatible regexes.
- Task 35 safe chart adapter.
- Existing canonical Gates/channels/centers and Phase 2 activation source paths.
- Standard library regex only; no NLP/LLM dependency.

Acceptance criteria:
- Implement the conceptual function:

  ```python
  build_retrieval_queries(
      user_query: str,
      chart_context: ChartContext | None,
  ) -> tuple[HybridSearchRequest, tuple[ChartFact, ...]]
  ```

- A small typed result may replace the tuple only if it reduces ambiguity without introducing a hierarchy.
- Knowledge-only mode with `chart_context=None` works and selects no chart facts.
- Original user text is preserved as `original_query` after safe outer whitespace trimming.
- Dense query remains natural language and preserves question meaning.
- Sparse query uses Task 30 canonical exact-term normalization.
- Supported deterministic intent/entity categories are:
  - exact Gate
  - exact Channel
  - exact Profile
  - exact Type
  - exact Authority
  - exact Center
  - decision-making/Authority
  - Strategy/Type
  - Profile
  - Definition
  - Channel/Center
  - specific planetary activation
  - vague but focused personalized question
- Decision questions select only Type, Authority, and Strategy facts when available.
- Exact Authority questions select Authority and, when useful to explain decision mechanics, Type/Strategy; they do not select unrelated Profile/Gate/Channel facts.
- Exact Type or Strategy questions select Type and Strategy only.
- Exact Profile questions select the Profile fact only unless the query names another explicit entity.
- Exact Definition questions select the Definition fact only unless the query names a Center/Channel.
- Exact Center questions select only that Center's deterministic defined/undefined fact and directly named Type/Authority context when the question explicitly asks for it.
- A generic exact Channel question selects the matching active-Channel fact when present and the two endpoint active-Gate facts where available. It does not automatically select planetary activation facts.
- A Channel question selects matching endpoint Personality/Design activation facts only when the query explicitly asks about a Gate, Line, planetary activation, Personality, Design, or a named planet.
- Merely naming a Channel by its endpoint numbers, such as `Channel 42-53`, does not count as asking for planetary activation details.
- An inactive Channel is never described as active, and no unrelated Channel, Gate, or activation fact is selected.
- Specific planetary questions select only the matching Personality/Design activation facts and the directly named Gate/Line fact context.
- For a Manifesting Generator/Sacral/To Respond example, dense text is natural language and sparse text contains those exact terms plus decision-making language.
- Gate questions detect `Gate 42`, `Hexagram 42`, common punctuation/word-order forms, and required Chinese forms.
- For a generic Gate question such as `What does Gate 42 mean in my chart?`, select only the matching active-Gate fact when active plus matching Personality and Design activation facts. Do not automatically select a related active Channel.
- A Gate question may select a matching active Channel containing that Gate only when the user explicitly asks about a Channel, connection, pairing, what the Gate connects to, whether it completes an active Channel, or how it works within a named Channel.
- Connection-oriented examples include `Is Gate 42 part of an active Channel?`, `What does Gate 42 connect to?`, `How does Gate 42 work in Channel 42-53?`, and `Does Gate 42 complete a Channel in my chart?`.
- If Gate 42 is not active, do not create or select a fact claiming it is active; keep the knowledge question intact and, if personalization wording is added, state only that the selected chart facts do not show it as active.
- For `Design Earth 42.6`, select the exact activation fact and include natural sparse terms `Gate 42`, `Hexagram 42`, `Line 6`, `42.6`, and `Design Earth`.
- Channel questions normalize endpoints so `42-53` and `53-42` identify the same canonical channel and sparse text may contain both orderings plus Gate 42/Gate 53.
- Profile questions preserve `4/6`, normalize `4-6 Profile`, and support `4/6人生角色` without treating it as a channel.
- Type, Authority, Center, Definition, and planetary names use small explicit canonical alias maps, including required Chinese aliases.
- Vague focused personalized questions select only core anchors: Type, Authority, and Profile.
- The builder never automatically maps career/relationship questions to selected Gates or Channels.
- Every selected `ChartFact` is an existing object/value from `ChartContext`; no invented chart fact or visual-only fact is created.
- Selected facts are deduplicated and deterministically ordered.
- No unselected chart fact is appended to either query.
- Full-reading phrase detection is implemented in Task 41 or a small shared predicate; Task 36 may expose entity/intent helpers but must not implement full-reading retrieval.
- Output is deterministic and makes no provider call.

Out of scope:
- No LLM intent classification.
- No LLM query rewriting.
- No LLM chart-fact selection.
- No career or relationship interpretation engine.
- No full reading.
- No retrieval execution.
- No generation.

Testing requirements:
- Write table-driven tests first and confirm RED for the missing builder.
- Test knowledge-only and chart-context modes.
- Test exact Gate, reverse Channel, Profile punctuation, Type, Authority, Center, Definition, decision, Strategy, and planetary activation intents.
- Test English and Chinese exact-entity examples.
- Assert dense and sparse queries differ for exact-entity/chart cases and each serves its documented purpose.
- Assert decision queries exclude active Gate/Channel dumps.
- Assert a generic Gate question selects its active-Gate and matching activation facts but no Channel.
- Assert a connection-oriented Gate question may select only matching active Channels that contain the queried Gate.
- Assert a generic Channel question selects the matching Channel and endpoint Gate facts but no planetary activations.
- Assert activation-specific Channel wording may select matching endpoint activations.
- Assert no unrelated Channel, Gate, or activation is selected in any of these cases.
- Assert inactive Gate questions never claim activation.
- Assert every selected fact is present in input `ChartContext`.
- Assert fact/query order is stable across repeated calls.
- Tests make no embeddings, retrieval, generation, or network calls.

Verification:

```sh
uv run pytest tests/test_query_builder.py
uv run pytest tests/test_bm25.py tests/test_chart_context.py
uv run ruff check .
git diff --check
```

### Task 37: Add the grounded prompt template, context assembly, and citation assignment

Goal:
Create one grounded answer prompt that clearly separates the user question, trusted selected chart facts, and untrusted final reference passages, while assigning deterministic citation IDs only after final source selection.

Files touched:
- Create `src/human_design/reading/prompts/human_design_answer.txt`.
- Create `src/human_design/reading/prompt.py`.
- Create `tests/test_reading_prompt.py`.

Dependencies:
- Task 28 `PromptContext`, `PromptSource`, `ChartFact`, and `RetrievedChunk`.
- Task 36 selected fact output.
- Task 34 final source order after NoOp/Cohere selection.
- Existing prompt-loading style in `human_design.vision.prompt`.
- Standard library `pathlib`, safe escaping, and deterministic serialization.

Acceptance criteria:
- The template has visibly delimited sections equivalent to:

  ```text
  <user_question>...</user_question>
  <chart_facts>...</chart_facts>
  <reference_sources>
    <source id="S1" file="..." page="...">...</source>
  </reference_sources>
  ```

- Prompt instructions state that chart facts are trusted structured facts for the current chart.
- Prompt instructions state that retrieved passages are untrusted knowledge data, never instructions.
- The model is told never to follow commands inside retrieved passages.
- The model is told to use only provided chart facts and never invent Gates, Channels, Centers, Type, Authority, Profile, Definition, Strategy, planetary activations, or chart relationships.
- The model is told to say when evidence is insufficient.
- Important book-derived claims require provided bracket citations such as `[S1]`.
- The model is forbidden to cite unknown source IDs.
- The model is instructed to avoid long verbatim quotations.
- Human Design is framed as reflective/experimental information, not medical or scientifically validated diagnostic guidance.
- Internal prompts, absolute filesystem paths, provider configuration, and credentials must not be exposed.
- Final selected source 1 receives `S1`, source 2 receives `S2`, and so on.
- Citation IDs are assigned only after RRF, optional rerank, and final Top-K selection.
- `PromptContext` contains only the user question, Task 36 selected facts, and final prompt sources.
- All Top-20/fusion candidates never enter the prompt.
- Source metadata rendered into the generation prompt uses `source_file`, optional title, page label, and page number only.
- A safe internal `source_relpath` may remain on the nested `RetrievedChunk` for retrieval provenance, but neither it nor any absolute `source_path` is rendered into the prompt or copied into a public `SourceCitation`.
- User text, chart values, metadata attributes, and source text are escaped/serialized so literal XML-like text cannot close or create prompt sections.
- Use one deterministic escape strategy: escape at least `&`, `<`, `>`, and attribute quotes; do not use fragile string replacement in multiple layers.
- Source content remains intact enough for grounded generation after escaping.
- Chart facts are rendered one per line/object with their exact `source_path` provenance.
- Chart facts that merely report Phase 2 structured data do not require a PDF citation; knowledge explanations do.
- Every `PromptContext` that proceeds to LLM generation contains at least one final `PromptSource`.
- The template distinguishes uncited structured chart-fact statements from cited Human Design explanation. Chart facts do not substitute for retrieved evidence.
- Phase 3 v1 does not assemble a chart-fact-only generation prompt or add a separate direct fact-only answer branch.
- Prompt loader errors clearly name a missing template without printing prompt content.

Out of scope:
- No generation API.
- No citation repair model.
- No full-reading prompt.
- No provider-specific multi-template framework.
- No prompt dump/debug CLI flag.

Testing requirements:
- Write prompt loader/context/citation tests first and confirm RED.
- Verify stable `S1`/`S2` assignment follows final input order.
- Verify only final selected chunks and Task 36 selected facts are rendered.
- Verify an unselected fact and extra fused candidate do not appear.
- Test malicious text containing closing tags, fake source tags, prompt commands, ampersands, and quotes; confirm it remains escaped within one data boundary.
- Test absolute local path metadata is excluded from rendered prompt and public citations.
- Test no raw BodyGraph object, visual fields, confidence data, base64 marker, API key, or provider config is rendered.
- Test empty chart facts for knowledge-only mode.
- Tests do not call an LLM.

Verification:

```sh
uv run pytest tests/test_reading_prompt.py
uv run pytest tests/test_query_builder.py tests/test_phase3_models.py
uv run ruff check .
git diff --check
```

### Task 38: Add one OpenAI Responses API structured generation path

Goal:
Generate one grounded `AnswerResult` from `PromptContext` using a single opt-in OpenAI Responses API call with structured output, then validate every cited source and chart-fact path locally.

Files touched:
- Create `src/human_design/reading/generator.py`.
- Create `tests/test_generation.py`.
- Modify `src/human_design/reading/__init__.py` only for an intentional public generator API.

Dependencies:
- Task 27 generation configuration and OpenAI key redaction.
- Task 28 answer/prompt/citation models.
- Task 37 prompt rendering.
- Existing OpenAI SDK already used lazily by `human_design.vision.client`; inspect the installed Responses API before implementation.
- Standard library JSON/schema validation if the installed SDK structured-output helper would require an unnecessary new dependency.

Acceptance criteria:
- Implement the conceptual API:

  ```python
  generate_answer(
      prompt_context: PromptContext,
      config: AppConfig,
      *,
      client: object | None = None,
  ) -> AnswerResult
  ```

- A small explicit client protocol is preferable to `object` in production annotations; optional injection keeps tests offline.
- The OpenAI Responses API is the only generation path/provider.
- Structured output requires at least:

  ```json
  {
    "answer_markdown": "...",
    "used_source_ids": ["S1", "S3"],
    "used_chart_fact_paths": [
      "derived_chart_data.basic_info.authority"
    ],
    "limitations": []
  }
  ```

- Use the installed SDK's documented Responses structured-output/JSON-schema contract; do not duplicate a Chat Completions implementation.
- The request sends the configured generation model, rendered prompt/input, structured schema, and `store=False`.
- `HD_RAG_REAL_GENERATION=0` prevents construction/call of a real OpenAI client.
- Real generation enabled requires the configured model and `OPENAI_API_KEY` before client construction.
- Tests may inject a fake client even while real generation remains disabled only through a clearly test-only/internal dependency path; production cannot bypass opt-in by passing a client from CLI input.
- No final sources returns `AnswerResult(status=AnswerStatus.INSUFFICIENT_EVIDENCE)` before rendering or calling generation.
- Generation is not called when final sources are empty, even when selected structured chart facts are available.
- Every answer that uses the LLM generation path has at least one final retrieved reference source.
- Structured chart facts may be stated without a PDF citation, but explanations and interpretations require citations to final retrieved sources.
- Chart facts alone do not trigger an uncited LLM answer.
- Phase 3 v1 has no separate direct fact-only response path; a future direct fact-only path may be considered later but is out of scope.
- Every `used_source_id` must exist in supplied `PromptSource` objects.
- Every `used_chart_fact_path` must identify a supplied selected `ChartFact`.
- Duplicate used IDs/paths are deduplicated in prompt order or rejected consistently and tested.
- Only source IDs actually cited in `answer_markdown` map to public `SourceCitation` objects containing no chunk text or scores.
- Valid chart fact paths map back to the supplied typed `ChartFact` objects.
- Unknown source IDs raise a focused generation validation error.
- Unknown chart fact paths raise a focused generation validation error.
- Bracket citations found in `answer_markdown` are parsed and must reference known supplied `PromptSource` IDs. Unknown bracket citations such as `[S99]` fail.
- Any cited source ID that is not listed in `used_source_ids` fails as a structured-output consistency error.
- A `used_source_id` that names an unknown or unsupplied source still fails under the separate supplied-`PromptSource` validation rule. The listed-but-unused relaxation applies only to valid supplied source IDs.
- Valid supplied source IDs listed in `used_source_ids` but not cited in `answer_markdown` do not fail. They are omitted from public citations, and one safe warning is appended to `AnswerResult.warnings`; the result remains `AnswerStatus.OK` when at least one valid bracket citation remains.
- An `AnswerResult` with `AnswerStatus.OK` must contain at least one bracket citation in `answer_markdown`. If removing listed-but-unused IDs leaves zero cited sources, structured-output validation fails rather than returning an uncited successful answer.
- Malformed JSON/structured output, wrong field types, and empty answer text fail clearly without a second LLM call.
- Valid output produces:

  ```python
  AnswerResult(status=AnswerStatus.OK)
  ```

  The Python enum member is `AnswerStatus.OK`, while its serialized value remains `"ok"`.
- `limitations` become safe answer warnings.
- Provider request/response payloads, full prompt, API key, and full chunks are never logged.
- Exactly one generation request is made; there is no fallback, citation repair, regeneration, or automatic JSON repair call.

Out of scope:
- No Chat Completions duplicate.
- No Anthropic or local model.
- No tools/agents.
- No streaming UI.
- No full reading.
- No direct chart-fact-only answer branch.
- No retry orchestration beyond minimal SDK-safe transport behavior.

Testing requirements:
- Write disabled/structured-output validation tests first and confirm RED.
- Use a fake Responses client that captures request kwargs and returns controlled structured output.
- Assert configured model and `store=False` are passed.
- Assert exactly one client call.
- Test valid cited-source and chart-fact mappings.
- Test that a valid supplied source listed in `used_source_ids` but unused in `answer_markdown` produces a safe warning, is omitted from public citations, keeps `AnswerStatus.OK`, and is not a hard failure when another valid bracket citation remains.
- Keep hard-fail tests for an unknown bracket citation, a cited-but-unlisted source ID, and a `used_source_id` naming an unknown or unsupplied source.
- Test that structured output with zero bracket citations after listed-but-unused IDs are removed fails consistency validation and cannot produce an uncited `AnswerStatus.OK` result.
- Test unknown chart-fact path, malformed output, wrong types, and blank answer.
- Test no-evidence short-circuit makes zero client calls.
- Test chart facts with zero final sources still return `insufficient_evidence` and make zero client calls.
- Test disabled real generation and missing key/model errors occur before OpenAI import/client call.
- Assert fake key, full prompt, and source text are absent from error strings/log output.
- No default test calls OpenAI or requires a key.

Verification:

```sh
uv run pytest tests/test_generation.py
uv run pytest tests/test_reading_prompt.py tests/test_config.py
uv run ruff check .
git diff --check
```

### Task 39: Add lightweight safety, prompt-injection, and privacy rules

Goal:
Add the minimum local input, prompt-boundary, high-stakes-framing, and privacy protections without adding a moderation service, classifier, or large policy framework.

Files touched:
- Modify `src/human_design/reading/query_builder.py` to expose one focused query validator if that is the smallest shared location.
- Modify `src/human_design/reading/prompt.py` and `src/human_design/reading/prompts/human_design_answer.txt` for safety instructions and escaping guarantees.
- Modify `src/human_design/reading/generator.py` only for safe validation/error behavior.
- Modify `tests/test_query_builder.py`, `tests/test_reading_prompt.py`, and `tests/test_generation.py`.
- Do not create `src/human_design/reading/safety.py` unless the three existing modules cannot hold these small rules cleanly.

Dependencies:
- Tasks 36 through 38.
- Task 28 `AnswerStatus` for safe pipeline outcomes.
- Existing privacy/cost rules and sanitized fixture conventions.
- No moderation dependency.

Acceptance criteria:
- Empty or whitespace-only queries fail through one typed `InvalidQuestionError` or equivalent focused exception.
- Query length has one fixed documented maximum measured in Python characters; use a modest v1 value such as 2,000 characters and test the exact boundary.
- Query validation occurs before embedding, BM25, reranking, prompt rendering, or generation.
- There is no keyword blacklist for ordinary Human Design words.
- Prompt sections continue to identify chart facts and retrieved passages as data.
- System/template instructions explicitly override any instructions found inside retrieved text.
- Escaped source text cannot break out of its source element even when it contains prompt-injection/XML-like strings.
- Generation receives no arbitrary tools, shell, file access, URL-fetching capability, or hidden prompt from source data.
- No API key or environment mapping enters `PromptContext`, rendered prompt, or provider input.
- The prompt prohibits medical and mental-health diagnosis, legal or financial decisions, guaranteed future events, deterministic death/illness/pregnancy/compatibility claims, and claims of scientifically proven medical guidance.
- The prompt permits safe reframing as general reflective information.
- `AnswerStatus.REFUSED` is reserved for explicit pipeline-level refusals or future caller-directed refusal behavior.
- Phase 3 v1 does not automatically decide when safe reframing is inappropriate and does not implement a local refusal detector, keyword refusal engine, prompt-injection classifier, or safety classifier.
- Name, birth date, birth time, birth place, image/path, raw Vision payload, confidence data, and unrelated BodyGraph metadata remain excluded by the Task 35/37 data model boundary.
- Normal logs/errors contain no full prompts, full retrieved chunks, birth information, keys, raw provider payloads, or arbitrary user-provided file paths.
- OpenAI generation continues to use `store=False`.
- Fixtures remain sanitized.
- No external moderation, output moderation, image moderation, injection classifier, policy database, or keyword-deny service is added.

Out of scope:
- No moderation API.
- No prompt-injection classifier service.
- No large safety framework or policy database.
- No arbitrary tool execution.
- No medical/legal/financial answer engine.
- No logging/observability platform.

Testing requirements:
- Add failing validation/privacy tests before modifying behavior.
- Test empty input and exact maximum-length boundary plus one character over.
- Inject fakes for dense, sparse, reranker, and generator boundaries and prove invalid input calls none of them.
- Test source injection text remains escaped inside a data delimiter.
- Test fake name/birth/path metadata supplied outside allowed models cannot reach prompt serialization.
- Test absolute paths are excluded from prompt and public citations.
- Test fake API keys are absent from prompt, errors, stdout/stderr capture, and logs.
- Test the prompt includes each high-stakes prohibition and reflective framing without asserting provider behavior.
- Reassert `store=False` through the fake client.
- No test calls a moderation, OpenAI, Cohere, Vision, embedding, or network service.

Verification:

```sh
uv run pytest tests/test_query_builder.py tests/test_reading_prompt.py tests/test_generation.py
uv run pytest tests/test_chart_context.py tests/test_phase3_models.py
uv run ruff check .
git diff --check
```

### Task 40: Build the end-to-end Phase 3 pipeline

Goal:
Connect validated request handling, safe chart adaptation, deterministic query construction, hybrid retrieval, optional reranking, prompt assembly, structured generation, and citation validation in one independently testable service with a thin Phase 2 image facade.

Files touched:
- Create `src/human_design/reading/pipeline.py`.
- Create `tests/test_reading_pipeline.py`.
- Modify `src/human_design/reading/__init__.py` for the intended public entry points.
- Modify existing reading/RAG modules only if an integration test exposes a narrow interface mismatch.

Dependencies:
- Task 26 request boundaries.
- Tasks 28 and 35 through 39 reading models/adapters.
- Tasks 31, 32, and 34 dense, sparse, hybrid, and rerank boundaries.
- Existing Phase 2 client/parser/interpreter/validation modules for the injectable outer image facade only.
- Existing `BodyGraphExtractionResult` constructor to assemble a full Phase 2 result after parse/interpret/validate.

Acceptance criteria:
- Provide conceptual public entry points:

  ```python
  answer_knowledge_question(
      request: KnowledgeQuestionRequest,
  ) -> AnswerResult

  answer_chart_question(
      request: ChartQuestionRequest,
  ) -> AnswerResult

  answer_chart_image_question(
      request: ChartImageQuestionRequest,
  ) -> AnswerResult
  ```

- A small `ReadingPipeline` dataclass/service with injected dense retriever, BM25 retriever, reranker, generator, index identity, and config may back these functions; do not create parallel knowledge/chart/image service hierarchies.
- The image function remains a thin outer facade that receives or constructs an injectable `Callable[[Path], BodyGraphExtractionResult]`.
- Production image extraction composes the existing `extract_bodygraph_raw_json()`, `parse_bodygraph_raw_extraction_json()`, `interpret_bodygraph()`, and `validate_bodygraph_extraction()` behavior, then explicitly constructs `BodyGraphExtractionResult`.
- The image facade constructs `ChartQuestionRequest` and immediately delegates to the same chart core.
- The core `answer_chart_question()` never receives or reads a `Path`, image bytes, or raw Vision JSON.
- Core chart tests inject a `BodyGraphExtractionResult`; they do not invoke Phase 2.
- Knowledge flow is:

  ```text
  validate query
    -> build dense/sparse queries
    -> dense + BM25 retrieval
    -> RRF
    -> optional rerank
    -> final Top-K selection
    -> citation assignment/prompt
    -> generation
    -> citation/chart-fact validation
    -> AnswerResult
  ```

- Task 40 deliberately leaves one insertion point after input validation and before chart adaptation/query building; Task 41 adds the focused full-reading scope guard there.
- Chart flow inserts validation gate, safe ChartContext, and selected relevant facts before the shared retrieval flow.
- Invalid Phase 2 validation returns `AnswerResult(status=AnswerStatus.INVALID_CHART)` with no citations/facts used and no retrieval/rerank/generation calls.
- Empty retrieval/final evidence returns `AnswerStatus.INSUFFICIENT_EVIDENCE` before generation.
- Selected chart facts with no final retrieved source also return `AnswerStatus.INSUFFICIENT_EVIDENCE`; Task 40 does not create a direct fact-only branch or call generation without reference evidence.
- Missing/mismatched manifest, dense index, or BM25 index raises a focused local index error and never silently falls back.
- `rerank_provider=none` uses the first `final_top_k` RRF candidates.
- `rerank_provider=cohere` is used only through the Task 34 opt-in adapter.
- Only final selected chunks enter `PromptContext`.
- Only Task 36 selected chart facts enter `PromptContext`.
- Final answer exposes validated citations/chart facts only; internal retrieval ranks/scores/candidates are not in default output.
- Dependencies are injectable so default tests use fakes and no storage/network.
- Real provider gates remain enforced at the individual client/embedding/rerank/generation boundaries; the pipeline does not bypass them.
- Expected operational errors become typed core exceptions or `AnswerResult` statuses according to their semantics; core code does not print.

Out of scope:
- No web/API server.
- No streaming.
- No chat/session history or multi-turn state.
- No full chart reading.
- No cloud deployment.
- No moderation API.
- No production mock flags.

Testing requirements:
- Write pipeline orchestration tests first with call-recording fakes.
- Verify exact knowledge-flow call order and that separate dense/sparse query strings reach their retrievers.
- Verify chart flow selects facts before retrieval and sends only selected facts to the fake generator.
- Verify invalid-chart and insufficient-evidence paths make zero downstream calls as required.
- Verify selected chart facts without final sources return `insufficient_evidence` and do not call the fake generator.
- Do not require `needs_focus` behavior in Task 40 tests; Task 41 is the first task that adds and tests that outcome.
- Verify missing BM25 or identity mismatch never returns dense-only output.
- Verify NoOp versus fake Cohere branches and final Top-K behavior.
- Verify only final candidates receive `S` citation IDs.
- Verify internal retrieval scores do not appear in `AnswerResult` serialization.
- Verify core chart tests contain no image access.
- Add one thin image-facade test with a fake extractor returning a sanitized `BodyGraphExtractionResult`; assert the image path is passed only to the fake extractor.
- No test calls Vision, embeddings, Chroma, BM25 storage, Cohere, OpenAI, or network.

Verification:

```sh
uv run pytest tests/test_reading_pipeline.py
uv run pytest tests/test_hybrid_retriever.py tests/test_reranker.py tests/test_chart_context.py tests/test_query_builder.py tests/test_reading_prompt.py tests/test_generation.py
uv run pytest tests/test_bodygraph_parser.py tests/test_bodygraph_interpreter.py tests/test_bodygraph_validation.py
uv run ruff check .
git diff --check
```

### Task 41: Add the focused-Q&A scope guard

Goal:
Detect broad complete-reading requests deterministically and return a helpful `needs_focus` result before any retrieval, reranking, or generation.

Files touched:
- Modify `src/human_design/reading/query_builder.py` for one small normalized broad-request predicate, or keep the predicate private in `pipeline.py` if it has no query-building reuse.
- Modify `src/human_design/reading/pipeline.py` to apply the guard after input validation and before query construction/retrieval.
- Modify `tests/test_query_builder.py` only if the predicate is public/tested there.
- Modify `tests/test_reading_pipeline.py`.

Dependencies:
- Task 28 `AnswerStatus.NEEDS_FOCUS` and `AnswerResult`.
- Task 39 input validation order.
- Task 40 pipeline injection/call-recording fakes.
- Standard library string normalization/regex only.

Acceptance criteria:
- English examples such as `Give me a complete reading`, `Read my entire chart`, `Explain everything in my chart`, and `Tell me everything about my Human Design` trigger the guard.
- Chinese examples such as `完整解讀我的人類圖`, `幫我看完整張圖`, and `全部都解釋給我` trigger the guard.
- Matching is case-insensitive for English and tolerant of surrounding whitespace/common terminal punctuation.
- The guard is a small explicit phrase/pattern set; it does not use an LLM, embeddings, classifier, or broad keyword blacklist.
- Task 41 modifies the completed Task 40 pipeline and is the first task that implements `needs_focus` behavior.
- The updated pipeline sequence is `input validation -> full-reading scope guard -> chart adaptation/query building/retrieval`.
- The guard runs after empty/oversized input validation but before chart adaptation, query construction, dense retrieval, or sparse retrieval.
- The deterministic result is equivalent to:

  ```json
  {
    "status": "needs_focus",
    "answer_markdown": "Please choose a specific topic, such as Authority, Profile, a Gate, a Channel, a Center, Type, Strategy, or Definition.",
    "citations": [],
    "chart_facts_used": [],
    "warnings": []
  }
  ```

- Wording may improve slightly but remains deterministic and useful.
- A broad request makes no dense/embedding/Chroma retrieval call, no BM25 retrieval call, no reranker call, and no generation call.
- Focused questions about one Gate, Channel, Profile, Authority, Center, Type, Strategy, or Definition continue normally.
- A phrase containing `complete` or `everything` in an unrelated focused context does not automatically trigger; patterns require reading/chart/explain-all intent.
- Full reading remains deferred to Phase 3.1.

Out of scope:
- No sectioned retrieval.
- No multiple generation calls.
- No synthesis.
- No full chart report or long-form templates.
- No multilingual classifier beyond the explicit English/Chinese patterns.

Testing requirements:
- Add English/Chinese parameterized failing tests first.
- Use pipeline fakes that raise if query building, dense/BM25 retrieval, reranking, or generation is called.
- Assert broad requests return `needs_focus` before chart adaptation or either retriever is invoked.
- Test case, whitespace, and punctuation normalization.
- Parameterize focused counterexamples for Gate 42, Channel 42-53, Profile 4/6, Sacral Authority, Solar Plexus Center, Generator Type, Strategy, and Split Definition.
- Test at least one false-positive-resistant sentence using `complete` in a focused question.
- Assert exact empty citations/chart facts/warnings and stable message.
- No provider or storage calls.

Verification:

```sh
uv run pytest tests/test_reading_pipeline.py tests/test_query_builder.py
uv run ruff check .
git diff --check
```

### Task 42: Add the CLI, README documentation, and final verification gate

Goal:
Expose focused Phase 3 knowledge/chart-image Q&A through one minimal local CLI, document the one-time hybrid build and opt-in provider behavior, and run the final offline regression/privacy gate across all phases.

Files touched:
- Create `scripts/ask_human_design.py`.
- Create `tests/test_ask_human_design.py`.
- Modify `README.md`.
- Modify `.env.example` only if Task 27 did not already document the final implemented variables accurately.
- Fix only narrow Phase 3 issues found by final verification; do not refactor Phase 1/2.

Dependencies:
- Tasks 26 through 41.
- Existing script conventions: `argparse`, `main(argv) -> int` where useful, core exceptions converted to safe stderr/exit behavior, and `dataclasses.asdict()`/JSON-safe enum conversion.
- Existing Phase 2 configuration and client for manual chart-image mode.
- Existing Phase 3 `AppConfig` provider gates.

Acceptance criteria:
- Knowledge command:

  ```sh
  uv run python scripts/ask_human_design.py \
    "What is Sacral Authority?"
  ```

- Chart-image command:

  ```sh
  uv run python scripts/ask_human_design.py \
    "How should I make decisions?" \
    --bodygraph data/bodygraph_samples/images/chart.png
  ```

- JSON command:

  ```sh
  uv run python scripts/ask_human_design.py \
    "What does Gate 42 mean?" \
    --json
  ```

- CLI supports exactly the positional query plus `--bodygraph` and `--json` application flags (in addition to standard `--help`).
- CLI does not expose `--mock-retrieval`, `--mock-generation`, `--provider-debug`, `--dump-prompt`, or `--dump-chunks`.
- Knowledge mode constructs `KnowledgeQuestionRequest`.
- Chart-image mode constructs `ChartImageQuestionRequest`, runs Phase 2 through the thin Task 40 facade, and enters Phase 3 with `BodyGraphExtractionResult`.
- Default output is concise human-readable status/answer/citation metadata and safe warnings.
- `--json` emits one serializable `AnswerResult` object with string enum values.
- `invalid_chart`, `needs_focus`, and `insufficient_evidence` are clear user-facing results rather than tracebacks.
- Disabled real embeddings, generation, Vision, or Cohere behavior produces an actionable message naming only the required opt-in/variable, never its value.
- No API key, full prompt, full retrieved chunk, provider payload, base64, or birth information is printed.
- CLI tests import/inject/monkeypatch pipeline dependencies; production mock flags are not added.
- Subprocess tests remove OpenAI/Cohere keys, explicitly set the rerank provider to `none`, set every Phase 2/Phase 3 real-mode flag to `0`, and run in a temporary cwd without the developer's `.env`.
- README documents:
  - Phase 3 focused-Q&A scope and non-goal of full reading
  - knowledge-only and chart-image modes
  - the Phase 2 image facade to `BodyGraphExtractionResult` boundary
  - one-time full hybrid ingestion rebuild
  - persisted canonical nodes and deterministic shared chunk IDs
  - new `storage/hybrid_v1` index root
  - legacy Phase 1 Chroma is not used by Phase 3 production retrieval
  - separate dense and BM25 queries
  - RRF rank fusion
  - optional Cohere reranking
  - offline/free default tests
  - the robust offline verification command that unsets inherited OpenAI/Cohere keys and explicitly sets real embedding mode to `0`, real generation mode to `0`, rerank provider to `none`, real rerank API mode to `0`, and real Vision API mode to `0`
  - process-environment values intentionally override provider and real-mode values in the developer's `.env`
  - real embeddings/generation/Cohere/Vision opt-ins and costs
  - privacy, provider data-sharing, and logging rules
  - full reading deferred to Phase 3.1
  - PDFs and generated storage must not be committed
- README does not claim a production index exists in a clean checkout.
- README does not include private filenames, absolute user paths, keys, full chunks, prompts, or provider payloads.
- Generation verification remains deterministic: valid/unknown citation IDs, valid/unknown chart paths, invalid chart, needs-focus, insufficient evidence, unselected facts, and PII exclusion are covered by tests; no LLM judge is added.
- Include a manual review checklist for answer relevance, groundedness, citation usefulness, clarity, and reflective framing.
- Do not create a second generation evaluation framework; retrieval evaluation remains Task 33.

Out of scope:
- No web/API/frontend.
- No production mock flags.
- No full reading.
- No LLM-as-a-judge.
- No deployment.
- No real provider call during automated verification.
- No commit of local indexes or private chart/PDF data.

Testing requirements:
- Write CLI tests first and confirm RED because the script does not exist.
- Unit-test `main(argv)` with an injected/fake pipeline for knowledge, chart-image, human-readable, and JSON modes.
- Add safe subprocess coverage from a temporary cwd with absolute script/fixture paths where useful.
- Add a regression test proving explicit process-environment values prevent `.env` values from enabling embeddings, generation, reranking, or Vision during offline CLI/test execution.
- In particular, when `.env` contains `HD_RAG_RERANK_PROVIDER=cohere` and `HD_RAG_REAL_RERANK_API=1`, explicit process values `HD_RAG_RERANK_PROVIDER=none` and `HD_RAG_REAL_RERANK_API=0` keep execution offline and avoid Cohere configuration validation.
- Verify no fake key, base64 marker, full source text, or fake birth data appears in stdout/stderr.
- Verify only the three intended flags are accepted/documented.
- Verify broad request returns before all fake provider calls.
- Verify chart mode passes its path only to the Phase 2 facade.
- Run the full suite offline with provider keys removed, `HD_RAG_RERANK_PROVIDER=none`, and `HD_RAG_REAL_EMBEDDINGS`, `HD_RAG_REAL_GENERATION`, `HD_RAG_REAL_RERANK_API`, and the existing Phase 2 `HD_VISION_REAL_API` flag explicitly set to `0`.
- Do not delete or assert absence of `storage/chroma`; the legacy Phase 1 store may legitimately exist locally.

Verification:

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

The explicit process-environment values intentionally override `.env`: provider keys are unset, real embedding/generation/rerank/Vision modes are forced to `0`, and the rerank provider is forced to `none`. A developer's local opt-in settings therefore cannot turn the final offline verification into a paid provider run or trigger Cohere configuration validation.

Manual hybrid build, optional and real-cost only:

```sh
HD_RAG_REAL_EMBEDDINGS=1 \
uv run python scripts/build_hybrid_index.py
```

Manual knowledge question, optional and real-cost only:

```sh
HD_RAG_REAL_EMBEDDINGS=1 \
HD_RAG_REAL_GENERATION=1 \
HD_RAG_GENERATION_MODEL=<configured-model> \
uv run python scripts/ask_human_design.py \
  "What is Sacral Authority?"
```

Manual Cohere reranking, optional and real-cost only:

```sh
HD_RAG_REAL_EMBEDDINGS=1 \
HD_RAG_REAL_GENERATION=1 \
HD_RAG_GENERATION_MODEL=<configured-model> \
HD_RAG_RERANK_PROVIDER=cohere \
HD_RAG_RERANK_MODEL=<configured-cohere-model> \
HD_RAG_REAL_RERANK_API=1 \
uv run python scripts/ask_human_design.py \
  "What does Gate 42 mean?"
```

Manual chart-image question, optional and real-cost only:

```sh
HD_VISION_REAL_API=1 \
HD_RAG_REAL_EMBEDDINGS=1 \
HD_RAG_REAL_GENERATION=1 \
HD_RAG_GENERATION_MODEL=<configured-model> \
uv run python scripts/ask_human_design.py \
  "How should I make decisions?" \
  --bodygraph data/bodygraph_samples/images/chart.png
```

The manual commands require the relevant keys/models in the process environment or local `.env`, may send data to external providers, and may incur cost. They are never part of the default verification gate.


## Plan Completion Checklist

Before declaring Phase 3 implementation complete, verify:

1. Tasks are implemented in order from 26 through 42.
2. No task implements BGE or adds a local reranker dependency.
3. No task implements a complete chart reading.
4. No task adds an external moderation API.
5. Dense and BM25 receive different purpose-specific queries.
6. Chroma and BM25 are built from the same canonical nodes.
7. Shared chunk IDs are deterministic and used end to end.
8. Phase 3 production uses a newly rebuilt hybrid index.
9. Legacy Phase 1 Chroma is not the Phase 3 production dense index.
10. Canonical nodes are persisted locally in ignored storage.
11. RRF uses 1-based ranks rather than raw score addition.
12. Image paths remain outside the Phase 3 chart core.
13. The actual `human_design.vision.models.BodyGraphExtractionResult` type is used.
14. Only question-relevant chart facts enter queries and prompts.
15. Normalized planetary activations remain available for specific activation questions.
16. Only `none` and `cohere` reranking are supported.
17. Every real API/embedding feature remains explicit opt-in.
18. Default tests are offline, free, and credential-independent.
19. Production has no silent dense-only fallback.
20. Test mocks are dependency-injected and not exposed as CLI flags.
21. No unnecessary model hierarchy, provider abstraction, safety framework, or module is introduced.
22. Every task has Goal, Files touched, Dependencies, Acceptance criteria, Out of scope, Testing requirements, and Verification.
23. Commands, modules, and filenames match the repository at implementation time.
24. No secrets, private chart data, copyrighted full chunks, or generated storage are committed.
25. `ingestion_id` is computed from canonical identity fields that exclude `ingestion_id` itself and is revalidated on reload.
26. Dense and sparse adapters expose only a simple validated `ingestion_id`, and Task 32 compares those values before retrieval.
27. Task 30 converts BM25 output into sparse-ranked `RetrievedChunk` models; Task 32 does not own sparse adaptation.
28. Generic Gate questions do not select Channels, and Channel/activation expansion follows only the explicit deterministic Task 36 triggers.
29. Every LLM-generated answer has at least one final retrieved source; structured chart facts alone never trigger generation.
30. Structured chart facts may be stated without PDF citations, while Human Design explanations require validated source citations.
31. The final offline test command unsets provider keys, forces `HD_RAG_RERANK_PROVIDER=none`, and sets every real-mode flag to `0`, overriding `.env` opt-ins.
32. `AnswerStatus.REFUSED` remains reserved and does not introduce an automatic refusal detector or safety classifier.
33. `fusion_top_k` may exceed available candidate counts; the pipeline returns the available unique candidates without a sum constraint.
