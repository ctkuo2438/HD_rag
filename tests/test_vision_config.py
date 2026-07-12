from pathlib import Path

import pytest

from human_design.vision.config import VisionConfig, load_vision_config


def test_default_config_values() -> None:
    config = load_vision_config({})

    assert config.model == "gpt-5.5"
    assert config.reasoning_effort == "high"
    assert config.real_api_enabled is False
    assert config.bodygraph_sample_dir == Path("data/bodygraph_samples/images")
    assert config.golden_labels_path == Path(
        "data/bodygraph_samples/golden_labels.example.json"
    )
    assert config.openai_api_key is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_truthy_real_api_values(value: str) -> None:
    assert load_vision_config({"HD_VISION_REAL_API": value}).real_api_enabled is True


@pytest.mark.parametrize(
    "value",
    [None, "", "0", "false", "FALSE", "no", "off", " Off "],
)
def test_falsey_real_api_values(value: str | None) -> None:
    env = {} if value is None else {"HD_VISION_REAL_API": value}

    assert load_vision_config(env).real_api_enabled is False


def test_invalid_real_api_value_raises() -> None:
    with pytest.raises(ValueError, match="HD_VISION_REAL_API"):
        load_vision_config({"HD_VISION_REAL_API": "sometimes"})


def test_empty_api_key_becomes_none() -> None:
    assert load_vision_config({"OPENAI_API_KEY": "  "}).openai_api_key is None


def test_custom_values_are_respected() -> None:
    config = load_vision_config(
        {
            "HD_VISION_MODEL": "custom-model",
            "HD_VISION_REASONING_EFFORT": "xhigh",
            "HD_BODYGRAPH_SAMPLE_DIR": "local/samples",
            "HD_BODYGRAPH_GOLDEN_LABELS": "local/labels.json",
        }
    )

    assert config.model == "custom-model"
    assert config.reasoning_effort == "xhigh"
    assert config.bodygraph_sample_dir == Path("local/samples")
    assert config.golden_labels_path == Path("local/labels.json")


@pytest.mark.parametrize("value", ["none", "low", "medium", "high", "xhigh"])
def test_allowed_reasoning_efforts_are_respected(value: str) -> None:
    config = load_vision_config({"HD_VISION_REASONING_EFFORT": value})

    assert config.reasoning_effort == value


def test_reasoning_effort_is_stripped_and_normalized_to_lowercase() -> None:
    config = load_vision_config({"HD_VISION_REASONING_EFFORT": " HIGH "})

    assert config.reasoning_effort == "high"


def test_blank_reasoning_effort_uses_default() -> None:
    config = load_vision_config({"HD_VISION_REASONING_EFFORT": "  "})

    assert config.reasoning_effort == "high"


def test_invalid_reasoning_effort_raises() -> None:
    with pytest.raises(ValueError, match="HD_VISION_REASONING_EFFORT"):
        load_vision_config({"HD_VISION_REASONING_EFFORT": "fast"})


def test_config_repr_redacts_api_key() -> None:
    secret = "fake-secret-for-repr"
    config = VisionConfig(
        model="model",
        reasoning_effort="high",
        real_api_enabled=True,
        bodygraph_sample_dir=Path("samples"),
        golden_labels_path=Path("labels.json"),
        openai_api_key=secret,
    )

    assert secret not in repr(config)
