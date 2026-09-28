# Human Design RAG

Ask focused Human Design questions through the CLI or a local Streamlit web app. Answers use your PDF library, include source citations, and can use relevant facts from an uploaded BodyGraph image.

Ask about one topic: Authority, Profile, Gate, Channel, Center, Type, Strategy, or Definition. Full-chart readings remain deferred to Phase 3.1. Human Design is presented as reflective/experimental information, not scientifically validated medical guidance.

## 1. Install

Requires Python 3.11+ and uv. Run all commands from the repository root:

```sh
uv sync
cp -n .env.example .env
```

The copy command preserves an existing `.env`. A clean checkout includes neither API keys nor a production hybrid index.

## 2. Configure your own keys and models

**Each user must fill in their own API keys in the local `.env` file.** Do not edit `.env.example` to contain real keys or commit `.env`.

The example file disables paid providers by default. For intentional real use, fill in your keys and models, then either update the switches below in your `.env` or use [temporary launch overrides](#optional-cohere-reranking). Enabling these flags allows paid API calls when you submit questions or build the index:

```dotenv
OPENAI_API_KEY=replace-with-your-openai-key
HD_RAG_GENERATION_MODEL=replace-with-your-responses-model

HD_RAG_REAL_EMBEDDINGS=1
HD_RAG_REAL_GENERATION=1
HD_VISION_REAL_API=1

HD_RAG_RERANK_PROVIDER=none
HD_RAG_REAL_RERANK_API=0
```

- `HD_VISION_MODEL` controls image parsing; `HD_RAG_GENERATION_MODEL` controls the final answer. Choose models available to your account; generation requires Responses API structured output.
- For knowledge-only questions, `HD_VISION_REAL_API` can remain `0`.
- Cohere is optional. Leave it disabled unless you configure [reranking](#optional-cohere-reranking).
- Other paths, model settings, and retrieval limits are listed in [.env.example](.env.example).
- Both interfaces load `.env`. Process-environment values override `.env`, including provider choices and opt-ins. Restart Streamlit after changing settings.

## 3. Build the hybrid index once

Put your local PDF books in `data/pdfs/`, then run:

```sh
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/build_hybrid_index.py
```

This sends PDF text to OpenAI for embeddings and incurs cost. It requires `OPENAI_API_KEY` and `HD_RAG_REAL_EMBEDDINGS=1`.

The build writes `nodes.jsonl`, `manifest.json`, `chroma/`, and `bm25/` under `storage/hybrid_v1`. Both indexes receive the same canonical nodes with deterministic shared chunk IDs and ingestion identity.

Reuse the existing index for later questions. See [rebuild details](docs/development.md#hybrid-index-rebuilds) when your PDFs or embedding model change.

## Use the CLI

Question arguments are defined in [`scripts/ask_human_design.py`](scripts/ask_human_design.py), in `_build_parser()`:

| Argument | Purpose |
| --- | --- |
| `"Your question"` | Required positional question; nonblank, at most 2,000 characters. |
| `--bodygraph PATH` | Optional chart image; omit for knowledge-only questions. |
| `--json` | Print one structured public answer instead of human-readable output. |
| `-h`, `--help` | List supported arguments and exit without provider calls. |

`--extra rerank` belongs to **`uv run`**, so place it before `python` or `streamlit`. It includes the optional `rerank` dependencies declared in [`pyproject.toml`](pyproject.toml); provider opt-ins are still required. Settings such as `HD_RAG_RERANK_PROVIDER=cohere` are environment variables, read by the configuration loaders.

Check the options directly:

```sh
uv run python scripts/ask_human_design.py --help
uv run --help
```

Knowledge-only question:

```sh
uv run python scripts/ask_human_design.py "What is Sacral Authority?"
```

Question about your chart, using your own local image and JSON output:

```sh
uv run python scripts/ask_human_design.py \
  "How should I make decisions?" \
  --bodygraph data/bodygraph_samples/images/chart.png \
  --json
```

A chart-image CLI command performs a new extraction each time; use Streamlit for repeated questions about the same image. For the other tools and their argument parsers, see the [CLI source map](docs/development.md#cli-source-map).

## Use Streamlit

```sh
uv run streamlit run scripts/streamlit_app.py
```

Open http://127.0.0.1:8501, enter a focused question, optionally upload an image, then select **送出問題**. The app shows the answer, cited books/pages, chart facts used, and limitations. Without an upload, it answers general knowledge questions. PNG, JPEG, WebP, and GIF uploads are limited to 20 MB.

Only submission starts the pipeline; opening the page or editing inputs does not call providers. The first image extraction may take several minutes. Within the current browser session, the same image content reuses its validated chart result, even after renaming the file. Failed or invalid extractions are not cached. A later answer failure does not discard an already validated chart.

Each follow-up still performs relevant-fact selection, retrieval, configured reranking, and generation. The cache is session-local memory, not shared between users or written to disk. Reloading the tab or restarting the server starts fresh. Temporary image copies are deleted after processing; the attached upload remains in Streamlit session memory. Stop the server with Ctrl+C. The app binds to localhost and disables Streamlit usage telemetry.

## Optional Cohere reranking

The default `none` preserves RRF order. To enable Cohere, set all of the following in your private `.env`:

```dotenv
COHERE_API_KEY=replace-with-your-cohere-key
HD_RAG_RERANK_MODEL=replace-with-your-rerank-model
HD_RAG_RERANK_PROVIDER=cohere
HD_RAG_REAL_RERANK_API=1
```

Use `--extra rerank` to install/include the optional dependency for either interface:

```sh
uv run --extra rerank python scripts/ask_human_design.py "What is Sacral Authority?"
uv run --extra rerank streamlit run scripts/streamlit_app.py
```

A Cohere key alone does not enable reranking. Selecting `cohere` while its real-API flag is `0` is a configuration error. Reranking has its own provider cost.

Alternatively, keep all real-API flags at `0` and `HD_RAG_RERANK_PROVIDER=none` in `.env`, then enable them for one Streamlit run:

```sh
HD_RAG_RERANK_PROVIDER=cohere \
HD_RAG_REAL_RERANK_API=1 \
HD_VISION_REAL_API=1 \
HD_RAG_REAL_EMBEDDINGS=1 \
HD_RAG_REAL_GENERATION=1 \
  uv run --extra rerank streamlit run scripts/streamlit_app.py
```

Your `.env` must still contain `OPENAI_API_KEY`, `COHERE_API_KEY`, `HD_RAG_GENERATION_MODEL` and `HD_RAG_RERANK_MODEL`, and the hybrid index must already exist. The overrides apply to all submissions during this Streamlit process and do not modify `.env`. For CLI use, keep the same prefix and replace the command with `uv run --extra rerank python scripts/ask_human_design.py "Your question"`.

## How answers are grounded

Dense retrieval gets a semantic query; BM25 gets a separate normalized lexical query. RRF fuses their ranks, optional Cohere reranks the candidates, and only final selected passages receive citation IDs such as `[S1]`.

The image facade reuses Phase 2 extraction, parsing, deterministic interpretation, and validation to produce `BodyGraphExtractionResult`. The chart core receives that typed result, never image paths, bytes, or raw Vision JSON. Only relevant validated chart facts reach generation.

Full-reading requests ask for one topic (`needs_focus`). Invalid charts stop before retrieval (`invalid_chart`); missing reference evidence stops before generation (`insufficient_evidence`). Structured chart facts alone do not trigger an answer-generation call.

## Costs and privacy

OpenAI receives PDF/query text for embeddings, images for Vision, and the question, selected chart facts, and final reference passages for generation. Optional Cohere receives the semantic query and candidate passages. Images themselves may contain personal information.

Generation uses one structured OpenAI Responses request with `store=False` and no tools. This still involves provider processing. Private chart metadata and structured local filesystem provenance are excluded from prompts/public citations; user and passage text remain escaped, untrusted data.

Do not commit keys, `.env`, PDFs, private images, generated `storage/`, raw Vision responses, or private outputs. Do not log full prompts, passages, provider payloads, credentials, image/base64 content, or birth data. Answers are for reflection, not medical diagnosis or legal/financial decisions.

## Default Verification

Default tests are offline, free, and credential-independent. They use fakes and temporary storage, not private charts or production indexes. Explicit process values below override any paid opt-ins in your `.env`; unsetting inherited keys alone is not enough because `.env` can supply them again.

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

To inspect the web UI without paid calls, use the same `env` prefix and replace `uv run pytest` with `uv run streamlit run scripts/streamlit_app.py`. Focused submissions then show safe configuration guidance.

## Manual answer review

For an intentional paid smoke test, check:

- Relevance: answers the question using only relevant chart facts.
- Groundedness: chart facts match the validated chart; book claims have supporting evidence.
- Citation usefulness: references point to the stated source/page and support the nearby claim.
- Clarity: the answer is understandable and states missing evidence.
- Reflective framing: no diagnosis, guarantees, or deterministic predictions.

## Development reference

See [development tools and contracts](docs/development.md) for shared PDF processing, chart extraction/evaluation, the reading service and retrieval metrics. Short component summaries are in [docs/](docs/).
