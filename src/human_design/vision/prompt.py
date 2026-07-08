"""Prompt loading for local BodyGraph raw Vision extraction."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = _REPO_ROOT / "prompts" / "bodygraph_raw_extraction.txt"


def load_bodygraph_raw_extraction_prompt() -> str:
    """Load the raw BodyGraph extraction prompt text."""
    if not _PROMPT_PATH.is_file():
        raise FileNotFoundError(f"BodyGraph extraction prompt not found: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8")


__all__ = ["load_bodygraph_raw_extraction_prompt"]
