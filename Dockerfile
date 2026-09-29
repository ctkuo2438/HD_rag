FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy
WORKDIR /app

# PyStemmer needs a native build on Linux ARM; keep the compiler out of runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra rerank --no-install-project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra rerank --no-editable

FROM python:3.11-slim-bookworm AS app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN useradd --create-home --uid 10001 app

COPY --from=builder /app/.venv /app/.venv
COPY scripts/ scripts/
COPY .streamlit/config.toml .streamlit/config.toml

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).close()"]
CMD ["streamlit", "run", "scripts/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# The maintainer supplies a separate, local build context containing the existing
# index. The normal source context still excludes storage and private inputs.
FROM app AS release
ARG VERSION=0.1.0
LABEL org.opencontainers.image.source="https://github.com/ctkuo2438/HD_rag" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.title="Human Design RAG"
ENV HD_RAG_INDEX_DIR=/app/storage/hybrid_v1
COPY --from=knowledge --chown=10001:10001 /manifest.json /nodes.jsonl /app/storage/hybrid_v1/
COPY --from=knowledge --chown=10001:10001 /chroma/ /app/storage/hybrid_v1/chroma/
COPY --from=knowledge --chown=10001:10001 /bm25/ /app/storage/hybrid_v1/bm25/

# Validate the copied corpus without network, PDF access or embedding construction.
RUN --network=none python -c "from human_design.rag.config import load_config; from human_design.rag.hybrid_index import verify_hybrid_index; m = verify_hybrid_index(load_config(env={})); print('Bundled index verified:', m.chunk_count, 'chunks')"
