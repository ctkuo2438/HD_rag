import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from human_design.vision.client import VisionClientError, extract_bodygraph_raw_json
from human_design.vision.config import VisionConfig
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.parser import parse_bodygraph_raw_extraction_json
from human_design.vision.validation import validate_bodygraph_extraction


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/extract_bodygraph.py"
IMAGE_PATH = REPO_ROOT / "tests/fixtures/bodygraph/test1.png"
MOCK_RESPONSE_PATH = (
    REPO_ROOT / "tests/fixtures/bodygraph/test1_raw_response.json"
)
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
FAKE_API_KEY = "fake-api-key-that-must-never-be-printed"
FAKE_BASE64 = "A" * 180
VISION_ENV_VARS = (
    "OPENAI_API_KEY",
    "HD_VISION_MODEL",
    "HD_VISION_REASONING_EFFORT",
    "HD_VISION_REAL_API",
    "HD_BODYGRAPH_SAMPLE_DIR",
    "HD_BODYGRAPH_GOLDEN_LABELS",
)


def _config(*, real_api_enabled: bool, api_key: str | None = None) -> VisionConfig:
    return VisionConfig(
        model="gpt-5.5",
        reasoning_effort="high",
        real_api_enabled=real_api_enabled,
        bodygraph_sample_dir=Path("data/bodygraph_samples/images"),
        golden_labels_path=Path(
            "data/bodygraph_samples/golden_labels.example.json"
        ),
        openai_api_key=api_key,
    )


def _run_cli(
    *args: str,
    cwd: Path,
    env_updates: dict[str, str | None] | None = None,
):
    env = os.environ.copy()
    for name in VISION_ENV_VARS:
        env.pop(name, None)
    for key, value in (env_updates or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_mock_client_returns_json_without_api_key() -> None:
    raw_json = extract_bodygraph_raw_json(
        image_path=IMAGE_PATH,
        config=_config(real_api_enabled=False),
        mock_response_path=MOCK_RESPONSE_PATH,
    )

    assert json.loads(raw_json)["personality"]["sun"] == "61.4"


def test_client_requires_mock_or_explicit_real_api() -> None:
    with pytest.raises(VisionClientError, match="HD_VISION_REAL_API=1|mock response"):
        extract_bodygraph_raw_json(
            image_path=IMAGE_PATH,
            config=_config(real_api_enabled=False),
        )


def test_real_client_requires_api_key_before_api_call() -> None:
    with pytest.raises(VisionClientError, match="OPENAI_API_KEY"):
        extract_bodygraph_raw_json(
            image_path=IMAGE_PATH,
            config=_config(real_api_enabled=True),
        )


def test_real_client_passes_model_and_reasoning_effort_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, object] = {}

    class FakeResponses:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured_request.update(kwargs)
            return SimpleNamespace(output_text='{"mock": "response"}')

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "offline-fake-key"
            self.responses = FakeResponses()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    config = VisionConfig(
        model="gpt-5.5",
        reasoning_effort="xhigh",
        real_api_enabled=True,
        bodygraph_sample_dir=Path("data/bodygraph_samples/images"),
        golden_labels_path=Path(
            "data/bodygraph_samples/golden_labels.example.json"
        ),
        openai_api_key="offline-fake-key",
    )

    raw_json = extract_bodygraph_raw_json(
        image_path=IMAGE_PATH,
        config=config,
    )

    assert raw_json == '{"mock": "response"}'
    assert captured_request["model"] == "gpt-5.5"
    assert captured_request["reasoning"] == {"effort": "xhigh"}


def test_client_rejects_missing_image_in_mock_mode(tmp_path: Path) -> None:
    with pytest.raises(VisionClientError, match="Image file not found"):
        extract_bodygraph_raw_json(
            image_path=tmp_path / "missing.png",
            config=_config(real_api_enabled=False),
            mock_response_path=MOCK_RESPONSE_PATH,
        )


def test_mock_response_runs_through_parser_interpreter_and_validation() -> None:
    raw_json = extract_bodygraph_raw_json(
        image_path=IMAGE_PATH,
        config=_config(real_api_enabled=False),
        mock_response_path=MOCK_RESPONSE_PATH,
    )
    parse_result = parse_bodygraph_raw_extraction_json(raw_json)
    interpretation_result = interpret_bodygraph(parse_result.raw_vision)
    validation_result = validate_bodygraph_extraction(
        parse_result=parse_result,
        interpretation_result=interpretation_result,
    )

    assert parse_result.raw_vision.personality.sun is not None
    assert interpretation_result.derived_chart_data.basic_info.profile == "4/6"
    assert validation_result.is_valid is True


def test_human_readable_cli_mock_smoke_flow_is_safe(tmp_path: Path) -> None:
    result = _run_cli(
        str(IMAGE_PATH),
        "--mock-response",
        str(MOCK_RESPONSE_PATH),
        cwd=tmp_path,
        env_updates={"OPENAI_API_KEY": FAKE_API_KEY},
    )

    assert result.returncode == 0, result.stderr
    assert "Raw Vision extraction" in result.stdout
    assert "Derived chart data" in result.stdout
    assert "Validation warnings" in result.stdout
    assert "Validation is_valid" in result.stdout
    assert FAKE_API_KEY not in result.stdout + result.stderr
    assert "OPENAI_API_KEY" not in result.stdout
    assert "data:image" not in result.stdout
    assert FAKE_BASE64 not in result.stdout


def test_json_cli_mock_smoke_flow_is_serializable_and_safe(tmp_path: Path) -> None:
    result = _run_cli(
        str(IMAGE_PATH),
        "--mock-response",
        str(MOCK_RESPONSE_PATH),
        "--json",
        cwd=tmp_path,
        env_updates={"OPENAI_API_KEY": FAKE_API_KEY},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "raw_vision",
        "derived_chart_data",
        "validation_result",
    }
    rendered = json.dumps(payload)
    assert FAKE_API_KEY not in rendered
    assert "data:image" not in rendered
    assert FAKE_BASE64 not in rendered


def test_cli_disabled_real_api_exits_clearly_without_openai_call(
    tmp_path: Path,
) -> None:
    result = _run_cli(str(IMAGE_PATH), cwd=tmp_path)

    assert result.returncode != 0
    assert "HD_VISION_REAL_API=1" in result.stderr
    assert "mock response" in result.stderr


def test_cli_real_api_without_key_exits_clearly(tmp_path: Path) -> None:
    result = _run_cli(
        str(IMAGE_PATH),
        cwd=tmp_path,
        env_updates={"HD_VISION_REAL_API": "1", "OPENAI_API_KEY": None},
    )

    assert result.returncode != 0
    assert "OPENAI_API_KEY" in result.stderr
    assert FAKE_API_KEY not in result.stderr


def test_cli_missing_image_exits_nonzero(tmp_path: Path) -> None:
    result = _run_cli(
        str(tmp_path / "missing.png"),
        "--mock-response",
        str(MOCK_RESPONSE_PATH),
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "Image file not found" in result.stderr


def test_cli_loads_only_controlled_working_directory_dotenv(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text(
        "HD_VISION_REAL_API=invalid-controlled-value\n",
        encoding="utf-8",
    )

    result = _run_cli(str(IMAGE_PATH), cwd=tmp_path)

    assert result.returncode != 0
    assert "HD_VISION_REAL_API must be one of" in result.stderr
    assert "OPENAI_API_KEY" not in result.stderr


def test_env_example_documents_safe_phase2_defaults() -> None:
    contents = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "HD_VISION_MODEL=gpt-5.5" in contents
    assert "HD_VISION_REASONING_EFFORT=high" in contents
    assert "HD_VISION_REAL_API=0" in contents
    assert "HD_BODYGRAPH_SAMPLE_DIR=data/bodygraph_samples/images" in contents
    assert (
        "HD_BODYGRAPH_GOLDEN_LABELS="
        "data/bodygraph_samples/golden_labels.example.json"
    ) in contents
    assert "sk-" not in contents
    assert "/Users/" not in contents
    assert "/home/" not in contents
