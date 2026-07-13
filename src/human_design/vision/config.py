"""Environment configuration for Phase 2 BodyGraph Vision extraction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_REASONING_EFFORT = "high"
_ALLOWED_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh"}
)
_DEFAULT_SAMPLE_DIR = Path("data/bodygraph_samples/images")
_DEFAULT_GOLDEN_LABELS = Path(
    "data/bodygraph_samples/golden_labels.example.json"
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


@dataclass(frozen=True)
class VisionConfig:
    """Typed configuration for one local BodyGraph extraction run."""

    model: str
    reasoning_effort: str
    real_api_enabled: bool
    bodygraph_sample_dir: Path
    golden_labels_path: Path
    openai_api_key: str | None = field(default=None, repr=False)


def load_vision_config(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> VisionConfig:
    """Load Phase 2 settings from dotenv, the process, or an explicit mapping."""
    if env is None:
        load_dotenv(
            dotenv_path=dotenv_path or Path(".env"),
            override=False,
        )
        source = os.environ
    else:
        source = env
    model = source.get("HD_VISION_MODEL", "").strip() or _DEFAULT_MODEL
    reasoning_effort = _parse_reasoning_effort(
        source.get("HD_VISION_REASONING_EFFORT", "")
    )
    sample_dir = source.get("HD_BODYGRAPH_SAMPLE_DIR", "").strip()
    golden_labels = source.get("HD_BODYGRAPH_GOLDEN_LABELS", "").strip()
    api_key = source.get("OPENAI_API_KEY", "").strip() or None

    return VisionConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        real_api_enabled=_parse_real_api_enabled(
            source.get("HD_VISION_REAL_API", "")
        ),
        bodygraph_sample_dir=Path(sample_dir) if sample_dir else _DEFAULT_SAMPLE_DIR,
        golden_labels_path=(
            Path(golden_labels) if golden_labels else _DEFAULT_GOLDEN_LABELS
        ),
        openai_api_key=api_key,
    )


def _parse_real_api_enabled(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        "HD_VISION_REAL_API must be one of: 1, true, yes, on, "
        "0, false, no, off"
    )


def _parse_reasoning_effort(value: str) -> str:
    normalized = value.strip().lower() or _DEFAULT_REASONING_EFFORT
    if normalized not in _ALLOWED_REASONING_EFFORTS:
        allowed_values = ", ".join(sorted(_ALLOWED_REASONING_EFFORTS))
        raise ValueError(
            f"HD_VISION_REASONING_EFFORT must be one of: {allowed_values}"
        )
    return normalized


__all__ = ["VisionConfig", "load_vision_config"]
