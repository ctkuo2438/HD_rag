"""Configuration loading for the local Human Design RAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values # for loading .env files


DEFAULT_PDF_DIR = Path("data/pdfs")
DEFAULT_CHROMA_DIR = Path("storage/chroma")
DEFAULT_COLLECTION_NAME = "human_design"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_INGESTION_VERSION = "v1"

# Environment variable names recognized by load_config().
#
# Precedence order:
# explicit env / shell environment > .env file > defaults
#
# In my local .env, these variables are currently set, so load_config()
# will use the .env values unless they are overridden by shell environment variables.
ENV_PDF_DIR = "HD_RAG_PDF_DIR"
ENV_CHROMA_DIR = "HD_RAG_CHROMA_DIR"
ENV_COLLECTION = "HD_RAG_COLLECTION"
ENV_EMBED_MODEL = "HD_RAG_EMBED_MODEL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"

# In my local .env, these variables are currently not set, so load_config()
# will use DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, and DEFAULT_INGESTION_VERSION
# unless they are later set in .env or overridden by shell environment variables.
ENV_CHUNK_SIZE = "HD_RAG_CHUNK_SIZE"
ENV_CHUNK_OVERLAP = "HD_RAG_CHUNK_OVERLAP"
ENV_INGESTION_VERSION = "HD_RAG_INGESTION_VERSION"


@dataclass(frozen=True)
class AppConfig:
    pdf_dir: Path
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    openai_api_key: str | None
    chunk_size: int
    chunk_overlap: int
    ingestion_version: str


def load_config(
    env: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> AppConfig:
    """Load app configuration from defaults, optional dotenv values, and env."""
    values = _load_env_values(env=env, env_file=env_file)
    chunk_size = _get_int(values, ENV_CHUNK_SIZE, DEFAULT_CHUNK_SIZE)
    chunk_overlap = _get_int(values, ENV_CHUNK_OVERLAP, DEFAULT_CHUNK_OVERLAP)
    _validate_chunk_settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return AppConfig(
        pdf_dir=Path(values.get(ENV_PDF_DIR, str(DEFAULT_PDF_DIR))),
        chroma_dir=Path(values.get(ENV_CHROMA_DIR, str(DEFAULT_CHROMA_DIR))),
        collection_name=values.get(ENV_COLLECTION, DEFAULT_COLLECTION_NAME),
        embedding_model=values.get(ENV_EMBED_MODEL, DEFAULT_EMBEDDING_MODEL),
        openai_api_key=values.get(ENV_OPENAI_API_KEY),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ingestion_version=values.get(ENV_INGESTION_VERSION, DEFAULT_INGESTION_VERSION),
    )


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
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer: {raw_value!r}") from exc


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
