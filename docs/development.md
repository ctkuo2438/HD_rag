# Development reference

For installation, your own API keys, index creation, CLI and Streamlit usage, start with the [README](../README.md).

## Current components

- [PDF ingestion and hybrid indexing](phase1_implementation_plan.md): shared PDF loading/chunking, canonical nodes, Chroma and BM25.
- [BodyGraph extraction](phase2_implementation_plan.md): Vision, parsing, deterministic interpretation and validation.
- [Focused Q&A](phase3_implementation_plan.md): query selection, retrieval, generation, citations and web-session reuse.

These are short descriptions of the implemented system. The original step-by-step implementation plans remain in Git history.

## CLI source map

Each CLI script owns its argument parsing. Look for `parser.add_argument(...)` in the locations below, or run `uv run python scripts/<script>.py --help` to list that script's options.

| Tool | Source | Parser location |
| --- | --- | --- |
| Focused Q&A | [ask_human_design.py](../scripts/ask_human_design.py) | `_build_parser()` |
| Hybrid index build | [build_hybrid_index.py](../scripts/build_hybrid_index.py) | `main()`; only `--help`, build settings come from configuration |
| Chart extraction | [extract_bodygraph.py](../scripts/extract_bodygraph.py) | `_build_parser()` |
| Chart evaluation | [evaluate_bodygraph_extraction.py](../scripts/evaluate_bodygraph_extraction.py) | `_build_parser()` |
| Retrieval evaluation | [evaluate_hybrid_retrieval.py](../scripts/evaluate_hybrid_retrieval.py) | `main()` |

`uv run --extra rerank` is parsed by uv. The `rerank` extra is declared under `[project.optional-dependencies]` in [pyproject.toml](../pyproject.toml); see `uv run --help` for uv options. It does not itself enable Cohere API calls.

[streamlit_app.py](../scripts/streamlit_app.py) defines the web form in `main()`, with no project argument parser. Streamlit owns its launch options: `uv run streamlit run --help`.

Environment settings come from [rag/config.py](../src/human_design/rag/config.py) (`load_config()`) and [vision/config.py](../src/human_design/vision/config.py) (`load_vision_config()`), with process values taking precedence over `.env`. They are separate from CLI arguments.

## Hybrid index rebuilds

Run `scripts/build_hybrid_index.py` only when intentionally building from your local PDFs with paid embeddings enabled. `HD_RAG_INDEX_DIR` defaults to `storage/hybrid_v1`.

The target must be empty. Existing indexes are never overwritten or deleted. When PDFs, chunk settings or the embedding model change, build into a fresh directory and point subsequent queries at the same root. Normal questions reuse the index.

The manifest, canonical nodes, Chroma and BM25 must agree on IDs and ingestion identity. Keep all four artifacts together. Both retrieval backends are required.

## Tests and diagnostics

Use the [explicit offline verification command](../README.md#default-verification) for all routine checks. It overrides any paid flags in a developer's private `.env`. Tests use temporary storage and fake providers.

Run a subset by adding test filenames after `uv run pytest` in that command. Useful groups:

- Index build/reload: `test_ingestion.py`, `test_chunking.py`, `test_hybrid_index.py`, `test_bm25.py` and `test_retriever.py`.
- Chart parsing and validation: `test_bodygraph_parser.py`, `test_bodygraph_interpreter.py`, `test_bodygraph_validation.py` and `test_extract_bodygraph.py`.
- Answers and interfaces: `test_reading_pipeline.py`, `test_generation.py`, `test_ask_human_design.py` and `test_streamlit_app.py`.

For a deterministic extraction smoke test, use the same offline environment with:

```sh
uv run python scripts/extract_bodygraph.py \
  tests/fixtures/bodygraph/test1.png \
  --mock-response tests/fixtures/bodygraph/test1_raw_response.json --json
```

`scripts/evaluate_bodygraph_extraction.py` evaluates saved predictions against matching manually labeled charts. `scripts/evaluate_hybrid_retrieval.py` evaluates an existing index with query labels; its real dense mode incurs embedding cost. Both provide `--help`. Examples under `tests/fixtures/` are sanitized, not labels for private charts.

## Change discipline

Preserve the v1 index format and canonical IDs, process-environment precedence, provider opt-ins, chart-validation rules and citation checks. Do not log private provider inputs or outputs. Keep generation to one structured request with `store=False`. See [AGENTS.md](../AGENTS.md) for project instructions.
