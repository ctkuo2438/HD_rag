import os
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
        openai_api_key=None,
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


def test_load_config_reads_openai_api_key_from_explicit_env() -> None:
    config = load_config(env={"OPENAI_API_KEY": "env-api-key"})

    assert config.openai_api_key == "env-api-key"


def test_config_repr_redacts_api_key() -> None:
    api_key = "fake-api-key-that-must-never-be-printed"
    config = load_config(env={"OPENAI_API_KEY": api_key})

    assert config.openai_api_key == api_key
    assert api_key not in repr(config)
    assert "openai_api_key" not in repr(config)
    assert "text-embedding-3-small" in repr(config)


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
                "OPENAI_API_KEY=env-file-api-key",
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
    assert config.openai_api_key == "env-file-api-key"
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


PHASE3_NUMERIC_SETTINGS = (
    ("HD_RAG_DENSE_TOP_K", "dense_top_k", 20),
    ("HD_RAG_SPARSE_TOP_K", "sparse_top_k", 20),
    ("HD_RAG_FUSION_TOP_K", "fusion_top_k", 20),
    ("HD_RAG_FINAL_TOP_K", "final_top_k", 4),
    ("HD_RAG_RRF_K", "rrf_k", 60),
)
PHASE3_BOOLEAN_SETTINGS = (
    ("HD_RAG_REAL_RERANK_API", "real_rerank_api"),
    ("HD_RAG_REAL_GENERATION", "real_generation"),
    ("HD_RAG_REAL_EMBEDDINGS", "real_embeddings"),
)


def test_phase3_defaults_are_offline_and_keep_legacy_paths() -> None:
    config = load_config(env={})
    assert config.index_dir == Path("storage/hybrid_v1")
    assert config.chroma_dir == Path("storage/chroma")
    assert config.collection_name == "human_design"
    assert config.rerank_provider == "none"
    assert config.rerank_model is None
    assert config.generation_model is None
    assert config.cohere_api_key is None
    for _, attr, default in PHASE3_NUMERIC_SETTINGS:
        assert getattr(config, attr) == default
    for _, attr in PHASE3_BOOLEAN_SETTINGS:
        assert getattr(config, attr) is False


def test_phase3_overrides() -> None:
    config = load_config(env={
        "HD_RAG_INDEX_DIR": "custom/hybrid",
        "HD_RAG_DENSE_TOP_K": "11",
        "HD_RAG_SPARSE_TOP_K": "12",
        "HD_RAG_FUSION_TOP_K": "30",
        "HD_RAG_FINAL_TOP_K": "6",
        "HD_RAG_RRF_K": "50",
        "HD_RAG_RERANK_PROVIDER": " COHERE ",
        "HD_RAG_RERANK_MODEL": " rerank-fixture ",
        "HD_RAG_GENERATION_MODEL": " generation-fixture ",
        "COHERE_API_KEY": " fake-cohere-key ",
        "OPENAI_API_KEY": " fake-openai-key ",
        "HD_RAG_REAL_RERANK_API": "1",
        "HD_RAG_REAL_GENERATION": "1",
        "HD_RAG_REAL_EMBEDDINGS": "1",
    })
    assert config.index_dir == Path("custom/hybrid")
    assert [getattr(config, attr) for _, attr, _ in PHASE3_NUMERIC_SETTINGS] == [
        11, 12, 30, 6, 50,
    ]
    assert config.rerank_provider == "cohere"
    assert config.rerank_model == "rerank-fixture"
    assert config.generation_model == "generation-fixture"
    assert config.cohere_api_key == "fake-cohere-key"
    assert config.openai_api_key == "fake-openai-key"
    for _, attr in PHASE3_BOOLEAN_SETTINGS:
        assert getattr(config, attr) is True


@pytest.mark.parametrize(("env_var", "attr", "default"), PHASE3_NUMERIC_SETTINGS)
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc", "", "True"])
def test_phase3_rejects_invalid_numbers(
    env_var: str, attr: str, default: int, value: str,
) -> None:
    with pytest.raises(ValueError, match=env_var):
        load_config(env={env_var: value})


def test_final_top_k_cannot_exceed_fusion_top_k() -> None:
    with pytest.raises(ValueError, match="HD_RAG_FINAL_TOP_K.*HD_RAG_FUSION_TOP_K"):
        load_config(env={"HD_RAG_FINAL_TOP_K": "21"})


def test_fusion_top_k_may_exceed_dense_plus_sparse() -> None:
    config = load_config(env={
        "HD_RAG_DENSE_TOP_K": "1",
        "HD_RAG_SPARSE_TOP_K": "1",
        "HD_RAG_FUSION_TOP_K": "50",
    })
    assert config.fusion_top_k == 50


@pytest.mark.parametrize(("env_var", "attr"), PHASE3_BOOLEAN_SETTINGS)
@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("TRUE", True), (" yes ", True), ("on", True),
     ("0", False), ("FALSE", False), (" no ", False), ("off", False),
     ("", False), ("  ", False)],
)
def test_phase3_strict_boolean_values(
    env_var: str, attr: str, value: str, expected: bool,
) -> None:
    config = load_config(env={
        env_var: value,
        "OPENAI_API_KEY": "fake-key",
        "HD_RAG_GENERATION_MODEL": "fixture-model",
    })
    assert getattr(config, attr) is expected


@pytest.mark.parametrize(("env_var", "attr"), PHASE3_BOOLEAN_SETTINGS)
@pytest.mark.parametrize("value", ["2", "sometimes", "enabled", "-1"])
def test_phase3_rejects_invalid_booleans(env_var: str, attr: str, value: str) -> None:
    with pytest.raises(ValueError, match=env_var):
        load_config(env={env_var: value})


@pytest.mark.parametrize("provider", ["none", " NONE ", "None"])
def test_no_reranker_requires_no_opt_in_credentials_or_model(provider: str) -> None:
    assert load_config(env={"HD_RAG_RERANK_PROVIDER": provider}).rerank_provider == "none"


@pytest.mark.parametrize("provider", ["bge", "openai", "unknown", "", "  "])
def test_invalid_rerank_provider(provider: str) -> None:
    with pytest.raises(ValueError, match="HD_RAG_RERANK_PROVIDER"):
        load_config(env={"HD_RAG_RERANK_PROVIDER": provider})


@pytest.mark.parametrize(
    "missing", ["HD_RAG_REAL_RERANK_API", "COHERE_API_KEY", "HD_RAG_RERANK_MODEL"],
)
@pytest.mark.parametrize("blank", [None, "", "  "])
def test_cohere_requires_opt_in_key_and_model(missing: str, blank: str | None) -> None:
    env = {
        "HD_RAG_RERANK_PROVIDER": " CoHeRe ",
        "HD_RAG_REAL_RERANK_API": "1",
        "COHERE_API_KEY": "fake-cohere-secret",
        "HD_RAG_RERANK_MODEL": "fixture-model",
    }
    if blank is None:
        del env[missing]
    else:
        env[missing] = blank
    with pytest.raises(ValueError, match=missing) as exc:
        load_config(env=env)
    assert "fake-cohere-secret" not in str(exc.value)


@pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "HD_RAG_GENERATION_MODEL"])
@pytest.mark.parametrize("blank", [None, "", "  "])
def test_real_generation_requires_key_and_model(missing: str, blank: str | None) -> None:
    env = {
        "HD_RAG_REAL_GENERATION": "1",
        "OPENAI_API_KEY": "fake-openai-secret",
        "HD_RAG_GENERATION_MODEL": "fixture-model",
    }
    if blank is None:
        del env[missing]
    else:
        env[missing] = blank
    with pytest.raises(ValueError, match=missing) as exc:
        load_config(env=env)
    assert "fake-openai-secret" not in str(exc.value)


def test_embedding_key_validation_stays_at_call_boundary() -> None:
    config = load_config(env={"HD_RAG_REAL_EMBEDDINGS": "1"})
    assert config.real_embeddings is True
    assert config.openai_api_key is None


@pytest.mark.parametrize("blank", ["", "  "])
def test_optional_keys_and_models_normalize_blank_to_none(blank: str) -> None:
    config = load_config(env={name: blank for name in (
        "OPENAI_API_KEY", "COHERE_API_KEY", "HD_RAG_RERANK_MODEL", "HD_RAG_GENERATION_MODEL",
    )})
    assert config.openai_api_key is None
    assert config.cohere_api_key is None
    assert config.rerank_model is None
    assert config.generation_model is None


def test_both_secret_fields_are_redacted() -> None:
    config = load_config(env={
        "OPENAI_API_KEY": "distinct-fake-openai-secret",
        "COHERE_API_KEY": "distinct-fake-cohere-secret",
    })
    for secret in (config.openai_api_key, config.cohere_api_key):
        assert secret not in repr(config)
    assert "cohere_api_key" not in repr(config)
    assert "openai_api_key" not in repr(config)


def test_explicit_mapping_is_isolated_from_local_env_and_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HD_RAG_REAL_GENERATION=1\nCOHERE_API_KEY=fake-dotenv\n")
    monkeypatch.setenv("HD_RAG_REAL_GENERATION", "1")
    monkeypatch.setenv("COHERE_API_KEY", "fake-process")
    config = load_config(env={})
    assert config.real_generation is False
    assert config.cohere_api_key is None


@pytest.mark.parametrize("explicit_mapping", [True, False])
def test_phase3_env_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_mapping: bool,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "HD_RAG_INDEX_DIR=dotenv/index\nHD_RAG_DENSE_TOP_K=7\n"
        "HD_RAG_SPARSE_TOP_K=8\nHD_RAG_REAL_GENERATION=1\n"
        "HD_RAG_RERANK_PROVIDER=cohere\nHD_RAG_REAL_RERANK_API=1\n",
    )
    overrides = {
        "HD_RAG_INDEX_DIR": "override/index", "HD_RAG_DENSE_TOP_K": "9",
        "HD_RAG_REAL_GENERATION": "0", "HD_RAG_RERANK_PROVIDER": "none",
        "HD_RAG_REAL_RERANK_API": "0", "HD_RAG_REAL_EMBEDDINGS": "0",
    }
    if not explicit_mapping:
        for key in tuple(os.environ):
            if key.startswith("HD_RAG_") or key in ("OPENAI_API_KEY", "COHERE_API_KEY"):
                monkeypatch.delenv(key)
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)
    config = load_config(env=overrides if explicit_mapping else None, env_file=env_file)
    assert config.index_dir == Path("override/index")
    assert config.dense_top_k == 9
    assert config.sparse_top_k == 8
    assert config.rrf_k == 60
    assert config.real_generation is False
    assert config.rerank_provider == "none"
