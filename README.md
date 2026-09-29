# Human Design RAG

Ask focused Human Design questions through a local Streamlit web app or CLI. Answers include book citations and can use relevant facts from an uploaded BodyGraph image.

Ask about one topic: Authority, Profile, Gate, Channel, Center, Type, Strategy, or Definition. Full-chart readings remain deferred to Phase 3.1. Human Design is presented as reflective/experimental information, not scientifically validated medical guidance.

| How you want to use it | Start here |
| --- | --- |
| Use the provided knowledge base with your own API keys | [Docker quick start](#use-docker) |
| Develop the code or build an index from your own PDFs | [Run from source](#run-from-source) |

## Use Docker

Requires Docker with Compose (Docker Desktop includes both) and your own OpenAI API key. The release image includes the app, Python dependencies, model settings, and the complete Chroma/BM25 knowledge index. You do not need to clone the repository, install Python/uv, supply PDFs, or rebuild the index.

### 1. Download the starter archive and fill in your key

Download `hd-rag-starter-0.1.0.tar.gz` from the project's GitHub Release and extract it into a folder. It contains only:

```text
compose.yaml
.env.example
README.md
```

Copy `.env.example` to `.env` in that folder and fill in your own key:

```dotenv
OPENAI_API_KEY=replace-with-your-openai-key
```

Keep `.env` private. Model names and retrieval settings are already provided by [compose.yaml](compose.yaml); your API account must have access to those models. The image contains model settings, not model weights.

### 2. Download the image and start the app

Run these commands from the extracted folder:

```sh
docker compose pull
docker compose up -d
```

The first command downloads `ghcr.io/ctkuo2438/hd-rag:0.1.0`, including the knowledge index. The second starts the container. Open http://127.0.0.1:8501 and ask a focused question, optionally uploading a chart image. Stop another app using port 8501 first.

**This consumer launcher enables paid OpenAI calls when you submit a question.** Opening the page makes no paid calls. Cohere is disabled by default. Keys are used locally to call the providers; there is no project-hosted application server.

To enable optional Cohere reranking, add these settings to your private `.env`, then rerun `docker compose up -d`:

```dotenv
COHERE_API_KEY=replace-with-your-cohere-key
HD_RAG_RERANK_PROVIDER=cohere
HD_RAG_REAL_RERANK_API=1
```

The Cohere dependency and model default are already included. No `--extra rerank` is needed with Docker. A Cohere key alone does not enable reranking.

### 3. Stop or change settings

```sh
docker compose down
```

The downloaded image retains the bundled index. No host `storage/` directory or volume is required; recreating the container starts with the index from the image.

Compose reads the adjacent `.env`; explicit shell values take precedence. After changing settings, rerun `docker compose up -d`. `restart` alone does not reload environment settings.

The running container also supports the CLI:

```sh
docker compose exec app python scripts/ask_human_design.py "What is Sacral Authority?" --json
```

## Image, starter archive, and Compose files

| Item | Contents and purpose | Distribution |
| --- | --- | --- |
| Docker image | App, Python dependencies, and the complete `storage/hybrid_v1` index | GHCR: `ghcr.io/ctkuo2438/hd-rag:0.1.0` |
| `dist/hd-rag-starter-0.1.0.tar.gz` | Three small launch files listed above; no app, image, or index | GitHub Releases attachment |

The starter archive's `compose.yaml` is a copy of the root [compose.yaml](compose.yaml). Its `.env.example` and `README.md` come from [docker/](docker/). Regenerate the archive after changing any of those source files; existing archives do not update automatically. Generated `dist/` files are ignored by Git.

The image supports Apple Silicon (`linux/arm64`) and Intel/AMD (`linux/amd64`). Both variants include their own runtime and the same knowledge corpus. Docker selects the matching variant for the user's computer. Storing both locally takes more space; it does not mean the embeddings were rebuilt for each CPU architecture.

| File | Who uses it | What it does |
| --- | --- | --- |
| [Dockerfile](Dockerfile) | Maintainer | Builds the app; the `release` stage also copies and verifies the existing hybrid index. |
| [compose.yaml](compose.yaml) | End user | Runs the prebuilt image with its bundled index and model defaults. |
| [compose.local.yaml](compose.local.yaml) | Developer | Builds the local source using the `app` stage and mounts a local index. Not included in the starter archive. |
| [docker/.env.example](docker/.env.example) | Docker user | Minimal key/reranking template; the consumer Compose supplies the other defaults. |
| [.env.example](.env.example) | Source developer | Full configuration template with all paid API switches off by default. |

`compose.local.yaml` is an optional development convenience. Docker Compose loads it only when explicitly selected with `-f compose.local.yaml`; normal users do not need it. See the [Docker guide](docker/README.md#maintainer-build-and-release) for maintainer build, publication and starter archive instructions.

## Run from source

This workflow requires a repository checkout, Python 3.11+ and uv. Run commands from the repository root. A clean checkout includes neither API keys nor a production hybrid index.

### 1. Install and configure

```sh
uv sync
cp -n .env.example .env
```

The copy command preserves an existing `.env`. Fill in your own key and generation model. For intentional paid use, enable the relevant switches:

```dotenv
OPENAI_API_KEY=replace-with-your-openai-key
HD_RAG_GENERATION_MODEL=replace-with-your-responses-model
HD_RAG_REAL_EMBEDDINGS=1
HD_RAG_REAL_GENERATION=1
HD_VISION_REAL_API=1
```

`HD_VISION_MODEL` controls image parsing; `HD_RAG_GENERATION_MODEL` controls the final answer and must support Responses API structured output. For knowledge-only questions, Vision can remain disabled. Other settings are listed in [.env.example](.env.example).

Process-environment values override `.env`, including provider choices and opt-ins. Restart native Streamlit after changing settings.

### 2. Prepare the hybrid index once

Reuse an existing compatible local index, or put your PDF books in `data/pdfs/` and run a full hybrid build:

```sh
HD_RAG_REAL_EMBEDDINGS=1 uv run python scripts/build_hybrid_index.py
```

This sends PDF text to OpenAI for embeddings and incurs cost. The build writes `nodes.jsonl`, `manifest.json`, `chroma/`, and `bm25/` together under `storage/hybrid_v1`. Both indexes use the same canonical nodes, deterministic chunk IDs, and ingestion identity.

Normal questions reuse these artifacts. Existing indexes are not overwritten. See [rebuild details](docs/development.md#hybrid-index-rebuilds) when your PDFs or embedding model change.

### 3. Start Streamlit or use the CLI

Start the web app and open http://127.0.0.1:8501; stop it with Ctrl+C:

```sh
uv run streamlit run scripts/streamlit_app.py
```

For a knowledge-only question through the CLI:

```sh
uv run python scripts/ask_human_design.py "What is Sacral Authority?"
```

For a question about your own local chart image:

```sh
uv run python scripts/ask_human_design.py \
  "How should I make decisions?" \
  --bodygraph data/bodygraph_samples/images/chart.png \
  --json
```

Arguments are defined in `_build_parser()` in [scripts/ask_human_design.py](scripts/ask_human_design.py):

| Argument | Purpose |
| --- | --- |
| `"Your question"` | Required, nonblank, at most 2,000 characters. |
| `--bodygraph PATH` | Optional chart image; omit for knowledge-only questions. |
| `--json` | Print one structured public answer instead of human-readable output. |
| `-h`, `--help` | Show supported arguments without provider calls. |

Each chart-image CLI command performs a new extraction. Streamlit reuses a validated chart within the same browser session. See the [CLI source map](docs/development.md#cli-source-map) for the other tools.

### Optional Cohere reranking

For source execution, also configure `COHERE_API_KEY` and `HD_RAG_RERANK_MODEL` in your private `.env`. To keep the API switches off in `.env` and enable them for one run:

```sh
HD_RAG_RERANK_PROVIDER=cohere \
HD_RAG_REAL_RERANK_API=1 \
HD_VISION_REAL_API=1 \
HD_RAG_REAL_EMBEDDINGS=1 \
HD_RAG_REAL_GENERATION=1 \
  uv run --extra rerank streamlit run scripts/streamlit_app.py
```

Your OpenAI key, generation model, Cohere key/model, and local index must already be configured. The overrides apply to this process and do not modify `.env`. For CLI use, keep the same prefix and replace the command with `uv run --extra rerank python scripts/ask_human_design.py "Your question"`.

`--extra rerank` belongs to **uv**, not the app's argument parser. It includes the optional dependency declared in [pyproject.toml](pyproject.toml); it does not enable API calls by itself. Without Cohere, `HD_RAG_RERANK_PROVIDER=none` preserves RRF order. Selecting `cohere` while `HD_RAG_REAL_RERANK_API=0` is a configuration error.

For local Docker development, after configuring the root `.env` and preparing a local index, use:

```sh
docker compose -f compose.local.yaml up --build -d
```

This rebuilds the app and mounts the host index selected by `HD_RAG_INDEX_DIR` (default `storage/hybrid_v1`). The directory must exist and be writable by the container user (UID 10001 on Linux). Avoid using the same index simultaneously from native and Docker processes.

## How the app works

In Docker, one container runs Streamlit, the Python pipeline, and local Chroma/BM25. There is no separate database server. Dense retrieval uses a semantic query; BM25 uses a separate normalized lexical query. RRF combines their ranks, optional Cohere reranks the candidates, and final selected passages receive citation IDs such as `[S1]`.

The chart-image facade reuses Phase 2 extraction, interpretation and validation to produce `BodyGraphExtractionResult`. The chart core receives that typed result, never image paths, bytes, or raw Vision JSON. Only relevant validated chart facts reach generation.

The web app accepts PNG, JPEG, WebP and GIF images up to 20 MB. First extraction may take several minutes. Successful chart validation is cached in the current browser session; later questions reuse it but still perform retrieval and generation. Reloading the tab or restarting the server clears that cache. Temporary image files are deleted after processing; the upload remains in session memory.

Full-reading requests ask for one topic (`needs_focus`). Invalid charts stop before retrieval (`invalid_chart`); missing reference evidence stops before generation (`insufficient_evidence`). Structured chart facts alone do not trigger generation.

## Costs and privacy

Each user supplies and pays for their own API keys. OpenAI receives query text for embeddings, uploaded images for Vision, and selected passages/chart facts with the question for generation. Building a new index also sends PDF text for embeddings. Optional Cohere receives the semantic query and candidate passages. Uploaded images may contain personal information.

Generation uses one structured OpenAI Responses request with `store=False` and no tools. Private chart metadata and structured local filesystem provenance stay out of prompts/public citations. User and passage text remain escaped, untrusted data. Native Streamlit binds to localhost; Docker publishes its web port only on localhost. Streamlit usage telemetry is disabled.

The release image distributes indexed book passages and vectors. Original PDFs, private images, keys and the maintainer's `.env` are excluded. Do not commit keys, `.env`, PDFs, private charts, generated storage or private outputs. Do not log full prompts, passages, provider payloads, credentials, image/base64 content or birth data.

## Default Verification

Default tests are offline, free, and credential-independent. They use fakes and temporary storage. Explicit process values below override paid opt-ins in `.env`; unsetting inherited keys alone is insufficient because `.env` can supply them again.

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

To view the native web UI offline, use the same `env` prefix and replace `uv run pytest` with `uv run streamlit run scripts/streamlit_app.py`. Focused submissions then show safe configuration guidance.

For an intentional paid answer review, check relevance, groundedness, useful citations, clarity, and reflective framing. Chart facts should match the validated chart; book claims should be supported by the cited passages. Answers must not provide diagnoses, guarantees, or substitute for medical, legal or financial advice.

See [development tools and contracts](docs/development.md) for further implementation and evaluation details.
