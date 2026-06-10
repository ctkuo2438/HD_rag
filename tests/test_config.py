from dataclasses import is_dataclass
from pathlib import Path

import pytest

from human_design.rag.config import AppConfig, load_config


def test_default_config_values_do_not_require_openai_key() -> None:
    config = load_config(env={})

    assert is_dataclass(AppConfig)
    assert config == AppConfig(
        pdf_dir=Path("data/pdfs"),
        chroma_dir=Path("storage/chroma"),
        collection_name="human_design",
        embedding_model="text-embedding-3-small",
        chunk_size=800,
        chunk_overlap=80,
        ingestion_version="v1",
    )


def test_path_environment_overrides() -> None:
    config = load_config(
        env={
            "HD_RAG_PDF_DIR": "custom/pdfs",
            "HD_RAG_CHROMA_DIR": "custom/chroma",
        }
    )

    assert config.pdf_dir == Path("custom/pdfs")
    assert config.chroma_dir == Path("custom/chroma")


def test_name_and_model_environment_overrides() -> None:
    config = load_config(
        env={
            "HD_RAG_COLLECTION": "charts",
            "HD_RAG_EMBED_MODEL": "custom-embedding-model",
        }
    )

    assert config.collection_name == "charts"
    assert config.embedding_model == "custom-embedding-model"


def test_chunk_environment_overrides() -> None:
    config = load_config(
        env={
            "HD_RAG_CHUNK_SIZE": "1200",
            "HD_RAG_CHUNK_OVERLAP": "120",
        }
    )

    assert config.chunk_size == 1200
    assert config.chunk_overlap == 120


def test_ingestion_version_environment_override() -> None:
    config = load_config(env={"HD_RAG_INGESTION_VERSION": "v2"})

    assert config.ingestion_version == "v2"


def test_load_config_can_read_explicit_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HD_RAG_PDF_DIR=env-file/pdfs",
                "HD_RAG_CHROMA_DIR=env-file/chroma",
                "HD_RAG_COLLECTION=env_file_collection",
                "HD_RAG_EMBED_MODEL=env-file-model",
                "HD_RAG_CHUNK_SIZE=900",
                "HD_RAG_CHUNK_OVERLAP=90",
                "HD_RAG_INGESTION_VERSION=env-file-v1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(env={}, env_file=env_file)

    assert config.pdf_dir == Path("env-file/pdfs")
    assert config.chroma_dir == Path("env-file/chroma")
    assert config.collection_name == "env_file_collection"
    assert config.embedding_model == "env-file-model"
    assert config.chunk_size == 900
    assert config.chunk_overlap == 90
    assert config.ingestion_version == "env-file-v1"


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("HD_RAG_CHUNK_SIZE", "not-an-int"),
        ("HD_RAG_CHUNK_OVERLAP", "not-an-int"),
    ],
)
def test_invalid_integer_values_raise_value_error(env_var: str, value: str) -> None:
    with pytest.raises(ValueError, match=env_var):
        load_config(env={env_var: value})


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_chunk_size_raises_value_error(value: str) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        load_config(env={"HD_RAG_CHUNK_SIZE": value})


@pytest.mark.parametrize("value", ["-1", "-20"])
def test_invalid_chunk_overlap_raises_value_error(value: str) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        load_config(env={"HD_RAG_CHUNK_OVERLAP": value})


def test_chunk_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        load_config(
            env={
                "HD_RAG_CHUNK_SIZE": "80",
                "HD_RAG_CHUNK_OVERLAP": "80",
            }
        )
