from dataclasses import replace
from unittest.mock import Mock

import pytest

from human_design.rag import embeddings
from human_design.rag.config import DEFAULT_EMBEDDING_MODEL, load_config


@pytest.mark.parametrize(("options", "model", "api_key"), [
    ({}, DEFAULT_EMBEDDING_MODEL, None),
    ({"embedding_model": "custom-model"}, "custom-model", None),
    ({"api_key": "fake-api-key"}, DEFAULT_EMBEDDING_MODEL, "fake-api-key"),
])
def test_embedding_factory_passes_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, str], model: str, api_key: str | None,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    constructor = Mock(return_value=object())
    monkeypatch.setattr(embeddings, "OpenAIEmbedding", constructor)

    result = embeddings.create_openai_embedding_model(**options)

    constructor.assert_called_once_with(model=model, api_key=api_key)
    assert result is constructor.return_value


def test_create_openai_embedding_model_from_config_uses_model_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = Mock(return_value=object())
    monkeypatch.setattr(embeddings, "OpenAIEmbedding", constructor)
    config = replace(load_config(env={}), embedding_model="config-model", openai_api_key="fake-config-key")

    result = embeddings.create_openai_embedding_model_from_config(config)

    constructor.assert_called_once_with(model="config-model", api_key="fake-config-key")
    assert result is constructor.return_value
