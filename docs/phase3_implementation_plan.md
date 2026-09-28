# Focused Q&A and reading pipeline

This replaces the completed Tasks 26-42 plan with the current service contract. Setup and usage are in the [README](../README.md); full-chart readings remain deferred to Phase 3.1.

## Public boundaries

`human_design.reading.ReadingPipeline` provides:

- `answer_knowledge_question(KnowledgeQuestionRequest)` for book-based questions.
- `answer_chart_question(ChartQuestionRequest)` for an existing typed `BodyGraphExtractionResult`.
- `answer_chart_image_question(ChartImageQuestionRequest)` as the thin image facade over Phase 2 extraction.

The chart core receives no image Path, bytes or raw Vision JSON. Providers and indexes are constructed lazily, after applicable validation and scope checks.

## Request flow

```text
Validate question -> focused-Q&A guard -> optional chart validation/projection
  -> relevant-fact selection -> separate dense/BM25 queries
  -> both retrievals -> RRF -> configured reranking -> final Top-K
  -> citation assignment -> escaped prompt -> structured generation
  -> validated public AnswerResult
```

Queries must be nonblank and at most 2,000 Python characters before trimming. Broad full-reading requests return `needs_focus` before chart adaptation or providers. Invalid charts return `invalid_chart`. Empty final evidence returns `insufficient_evidence` without prompt rendering or generation, even when chart facts exist.

## Fact selection and retrieval

`reading/chart_context.py` projects only validated deterministic facts. `reading/query_builder.py` selects existing facts by intent; it never invents active Gates or Channels. Decision questions use Type, Authority and Strategy. Profile, Center, Gate, Channel and planetary questions use the corresponding focused facts and only explicitly useful context.

Dense and BM25 receive purpose-specific queries. Sparse normalization supports English/Chinese aliases, canonical `G`/`Ego` terms and Gate/Channel/Profile forms. It does not rewrite the semantic query.

`rag/hybrid_retriever.py` verifies matching ingestion identities before either query. It merges only by canonical chunk ID and rejects conflicting text/source metadata. RRF adds `1 / (rrf_k + rank)` for each present 1-based rank, never raw score arithmetic. Ordering is descending RRF, ascending best rank, then lexical chunk ID; truncation follows fusion.

`NoOpReranker` preserves RRF order. Optional Cohere receives the semantic query and fused candidate text, preserves provenance, and maps provider indexes back to canonical chunks. Cohere requires its optional dependency, selected provider, model, key and explicit API opt-in. There is no silent retrieval or reranking fallback.

## Prompt and answer contracts

Only final selected sources receive `S1`, `S2`, etc. Only selected chart facts enter `PromptContext`. User questions and passages remain escaped untrusted data; structured local paths, retrieval scores and private chart metadata are omitted.

Generation uses one OpenAI Responses request with structured output, `store=False` and no tools. It requires `HD_RAG_REAL_GENERATION=1`, a configured generation model and `OPENAI_API_KEY`. There is no repair call.

Returned source IDs and chart-fact paths must belong to the supplied context. Every bracket citation must also be listed in `used_source_ids`. Unknown references fail. An OK result always has at least one real bracket citation; public citations include only bracket-cited sources. Supplied-but-uncited IDs add a safe warning instead of a public citation.

Errors expose safe project-owned messages, not prompts, passages, credentials or provider payloads. Human Design is framed as reflective information without diagnosis or deterministic guarantees. `REFUSED` remains reserved; there is no automatic refusal classifier.

## Interfaces and session reuse

- `scripts/ask_human_design.py` accepts a query, optional `--bodygraph` and optional `--json`.
- `scripts/streamlit_app.py` uses the same service. A submitted image is fingerprinted by content; its validated typed result is reused only within that browser session.
- Invalid/failed extractions are not cached. Valid extraction survives a later answer failure. Each new question still selects facts and retrieves/generates a fresh answer.
- Sessions are independent; cached chart data is not written to disk. Cached image requests call the typed chart core and skip Vision.
- Input validation, scope checks, citation validation and provider gates apply to both interfaces.

## Verification

Use the [full offline gate](../README.md#default-verification). Pipeline, CLI and Streamlit tests inject fake boundaries; retrieval tests use temporary indexes. Existing production storage must not be used by tests.

`scripts/evaluate_hybrid_retrieval.py` independently reports dense-only, BM25-only and RRF results: Hit@5, MRR@20, exact entity hits and expected source/page hits. It reloads existing indexes without building them. Aggregate thresholds determine exit status; unlabeled metrics are excluded. Real dense evaluation requires an explicit embeddings opt-in.

Manual answer review checks relevance, groundedness, citation usefulness, clarity and reflective framing. Automated tests do not use an LLM judge.
