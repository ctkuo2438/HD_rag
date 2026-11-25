"""Environment configuration for Phase 2 BodyGraph Vision extraction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_REASONING_EFFORT = "high"
_ALLOWED_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


@dataclass(frozen=True)
class VisionConfig:
    model: str
    reasoning_effort: str
    real_api_enabled: bool
    openai_api_key: str | None = field(default=None, repr=False)


def load_vision_config(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> VisionConfig:
    source: Mapping[str, str]
    if env is None:
        # Read .env without load_dotenv, which would copy its values, including the
        # API key, into os.environ for every library in the process.
        file_values = dotenv_values(dotenv_path or Path(".env"))
        source = {**{key: value for key, value in file_values.items() if value is not None}, **os.environ}
    else:
        # testability: allow passing a custom env mapping for testing
        source = env

    # Source precedence:
    # - explicit env mapping > defaults
    # - otherwise: process environment > .env > defaults
    model = source.get("HD_VISION_MODEL", "").strip() or _DEFAULT_MODEL
    reasoning_effort = _parse_reasoning_effort(source.get("HD_VISION_REASONING_EFFORT", ""))
    api_key = source.get("OPENAI_API_KEY", "").strip() or None

    return VisionConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        real_api_enabled=_parse_real_api_enabled(source.get("HD_VISION_REAL_API", "")),
        openai_api_key=api_key,
    )


def _parse_real_api_enabled(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError("HD_VISION_REAL_API must be one of: 1, true, yes, on, 0, false, no, off")


def _parse_reasoning_effort(value: str) -> str:
    normalized = value.strip().lower() or _DEFAULT_REASONING_EFFORT
    if normalized not in _ALLOWED_REASONING_EFFORTS:
        allowed_values = ", ".join(sorted(_ALLOWED_REASONING_EFFORTS))
        raise ValueError(f"HD_VISION_REASONING_EFFORT must be one of: {allowed_values}")
    return normalized


__all__ = ["VisionConfig", "load_vision_config"]
