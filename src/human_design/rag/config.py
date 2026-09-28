"""Configuration loading for the local Human Design RAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values # for loading .env files


DEFAULT_PDF_DIR = Path("data/pdfs")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_INGESTION_VERSION = "v1"
DEFAULT_INDEX_DIR = Path("storage/hybrid_v1")
DEFAULT_DENSE_TOP_K = 20
DEFAULT_SPARSE_TOP_K = 20
DEFAULT_FUSION_TOP_K = 20
DEFAULT_FINAL_TOP_K = 4
DEFAULT_RRF_K = 60

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})

# Explicit mappings / process environment override .env values and defaults.
ENV_PDF_DIR = "HD_RAG_PDF_DIR"
ENV_EMBED_MODEL = "HD_RAG_EMBED_MODEL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_CHUNK_SIZE = "HD_RAG_CHUNK_SIZE"
ENV_CHUNK_OVERLAP = "HD_RAG_CHUNK_OVERLAP"
ENV_INGESTION_VERSION = "HD_RAG_INGESTION_VERSION"


@dataclass(frozen=True)
class AppConfig:
    pdf_dir: Path
    embedding_model: str
    openai_api_key: str | None = field(repr=False)
    chunk_size: int
    chunk_overlap: int
    ingestion_version: str
    index_dir: Path = DEFAULT_INDEX_DIR
    dense_top_k: int = DEFAULT_DENSE_TOP_K
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K
    fusion_top_k: int = DEFAULT_FUSION_TOP_K
    final_top_k: int = DEFAULT_FINAL_TOP_K
    rrf_k: int = DEFAULT_RRF_K
    rerank_provider: str = "none"
    rerank_model: str | None = None
    generation_model: str | None = None
    real_rerank_api: bool = False
    real_generation: bool = False
    real_embeddings: bool = False
    cohere_api_key: str | None = field(default=None, repr=False)


def load_config(
    env: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> AppConfig:
    """Load app configuration from defaults, optional dotenv values, and env."""
    values = _load_env_values(env=env, env_file=env_file)
    chunk_size = _get_int(values, ENV_CHUNK_SIZE, DEFAULT_CHUNK_SIZE)
    chunk_overlap = _get_int(values, ENV_CHUNK_OVERLAP, DEFAULT_CHUNK_OVERLAP)
    _validate_chunk_settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    config = AppConfig(
        pdf_dir=Path(values.get(ENV_PDF_DIR, str(DEFAULT_PDF_DIR))),
        embedding_model=values.get(ENV_EMBED_MODEL, DEFAULT_EMBEDDING_MODEL),
        openai_api_key=_get_optional_string(values, ENV_OPENAI_API_KEY),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ingestion_version=values.get(ENV_INGESTION_VERSION, DEFAULT_INGESTION_VERSION),
        index_dir=Path(values.get("HD_RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR))),
        dense_top_k=_get_positive_int(values, "HD_RAG_DENSE_TOP_K", DEFAULT_DENSE_TOP_K),
        sparse_top_k=_get_positive_int(values, "HD_RAG_SPARSE_TOP_K", DEFAULT_SPARSE_TOP_K),
        fusion_top_k=_get_positive_int(values, "HD_RAG_FUSION_TOP_K", DEFAULT_FUSION_TOP_K),
        final_top_k=_get_positive_int(values, "HD_RAG_FINAL_TOP_K", DEFAULT_FINAL_TOP_K),
        rrf_k=_get_positive_int(values, "HD_RAG_RRF_K", DEFAULT_RRF_K),
        rerank_provider=values.get("HD_RAG_RERANK_PROVIDER", "none").strip().lower(),
        rerank_model=_get_optional_string(values, "HD_RAG_RERANK_MODEL"),
        generation_model=_get_optional_string(values, "HD_RAG_GENERATION_MODEL"),
        real_rerank_api=_get_bool(values, "HD_RAG_REAL_RERANK_API"),
        real_generation=_get_bool(values, "HD_RAG_REAL_GENERATION"),
        real_embeddings=_get_bool(values, "HD_RAG_REAL_EMBEDDINGS"),
        cohere_api_key=_get_optional_string(values, "COHERE_API_KEY"),
    )
    _validate_phase3_settings(config)
    return config


# Helper function for collecting configuration values.
# Precedence order: explicit env / os.environ > .env file > defaults.
def _load_env_values(
    env: Mapping[str, str] | None,
    env_file: Path | None,
) -> dict[str, str]:

    values: dict[str, str] = {}
    dotenv_path: Path | None = env_file

    # If no explicit env mapping and no env_file are provided,
    # fall back to a local .env file in the current directory.
    if dotenv_path is None and env is None:
        dotenv_path = Path(".env")

    # Load values from .env first, if available.
    # Ignore keys whose value is None.
    if dotenv_path is not None and dotenv_path.exists():
        values.update(
            {
                key: value
                for key, value in dotenv_values(dotenv_path).items()
                if value is not None
            }
        )  
    # Then overlay environment values.
    # This makes env / os.environ override values from .env.
    values.update(os.environ if env is None else env)
    return values


def _get_int(values: Mapping[str, str], env_var: str, default: int) -> int:
    raw_value = values.get(env_var) # ex: HD_RAG_CHUNK_SIZE = 1000, raw_value = "1000"
    if raw_value is None:
        return default
    
    try:
        return int(raw_value) # convert "1000" to 1000
    except ValueError:
        raise ValueError(f"{env_var} must be an integer") from None


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")


def _get_positive_int(values: Mapping[str, str], env_var: str, default: int) -> int:
    value = _get_int(values, env_var, default)
    if value <= 0:
        raise ValueError(f"{env_var} must be a positive integer")
    return value


def _get_bool(values: Mapping[str, str], env_var: str) -> bool:
    value = values.get(env_var, "").strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"{env_var} must be one of: 1, true, yes, on, 0, false, no, off")


def _get_optional_string(values: Mapping[str, str], env_var: str) -> str | None:
    return values.get(env_var, "").strip() or None


def _validate_phase3_settings(config: AppConfig) -> None:
    if config.final_top_k > config.fusion_top_k:
        raise ValueError("HD_RAG_FINAL_TOP_K must be less than or equal to HD_RAG_FUSION_TOP_K")
    if config.rerank_provider not in {"none", "cohere"}:
        raise ValueError("HD_RAG_RERANK_PROVIDER must be none or cohere")
    if config.rerank_provider == "cohere":
        if not config.real_rerank_api:
            raise ValueError("HD_RAG_REAL_RERANK_API must be enabled for cohere")
        if config.cohere_api_key is None:
            raise ValueError("COHERE_API_KEY is required for cohere")
        if config.rerank_model is None:
            raise ValueError("HD_RAG_RERANK_MODEL is required for cohere")
    if config.real_generation:
        if config.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when HD_RAG_REAL_GENERATION is enabled")
        if config.generation_model is None:
            raise ValueError("HD_RAG_GENERATION_MODEL is required when HD_RAG_REAL_GENERATION is enabled")
