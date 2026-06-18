from dataclasses import replace

import pytest

from human_design.rag import embeddings
from human_design.rag.config import DEFAULT_EMBEDDING_MODEL, load_config


def test_default_embedding_model_is_text_embedding_3_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, str | None] = {}

    class FakeOpenAIEmbedding:
        def __init__(self, **kwargs: str | None) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(embeddings, "OpenAIEmbedding", FakeOpenAIEmbedding)

    embedding_model = embeddings.create_openai_embedding_model()

    assert isinstance(embedding_model, FakeOpenAIEmbedding)
    assert captured_kwargs["model"] == DEFAULT_EMBEDDING_MODEL
    assert captured_kwargs["api_key"] is None


def test_embedding_model_override_is_passed_to_openai_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, str | None] = {}

    class FakeOpenAIEmbedding:
        def __init__(self, **kwargs: str | None) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(embeddings, "OpenAIEmbedding", FakeOpenAIEmbedding)

    embeddings.create_openai_embedding_model(embedding_model="custom-model")

    assert captured_kwargs["model"] == "custom-model"


def test_api_key_argument_is_passed_to_openai_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, str | None] = {}

    class FakeOpenAIEmbedding:
        def __init__(self, **kwargs: str | None) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(embeddings, "OpenAIEmbedding", FakeOpenAIEmbedding)

    embeddings.create_openai_embedding_model(api_key="test-api-key")

    assert captured_kwargs["api_key"] == "test-api-key"


def test_create_openai_embedding_model_from_config_uses_model_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, str | None] = {}

    class FakeOpenAIEmbedding:
        def __init__(self, **kwargs: str | None) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(embeddings, "OpenAIEmbedding", FakeOpenAIEmbedding)
    config = replace(
        load_config(env={}),
        embedding_model="config-model",
        openai_api_key="config-api-key",
    )

    embeddings.create_openai_embedding_model_from_config(config)

    assert captured_kwargs == {
        "model": "config-model",
        "api_key": "config-api-key",
    }


def test_embedding_factory_does_not_require_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeOpenAIEmbedding:
        def __init__(self, **kwargs: str | None) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(embeddings, "OpenAIEmbedding", FakeOpenAIEmbedding)

    embedding_model = embeddings.create_openai_embedding_model()

    assert isinstance(embedding_model, FakeOpenAIEmbedding)
    assert embedding_model.kwargs["api_key"] is None
