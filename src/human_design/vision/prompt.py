"""Prompt loading for local BodyGraph raw Vision extraction."""

from __future__ import annotations

from importlib.resources import files

# The prompt ships as package data under human_design/vision/prompts/, so
# loading works regardless of install layout (editable, wheel, zip import).
_PROMPT_PACKAGE = "human_design.vision"
_PROMPT_RESOURCE = "bodygraph_raw_extraction.txt"


def load_bodygraph_raw_extraction_prompt() -> str:
    """Load the raw BodyGraph extraction prompt text from package data."""
    resource = files(_PROMPT_PACKAGE) / "prompts" / _PROMPT_RESOURCE
    if not resource.is_file():
        raise FileNotFoundError(
            f"BodyGraph extraction prompt not found: {resource}"
        )
    return resource.read_text(encoding="utf-8")


__all__ = ["load_bodygraph_raw_extraction_prompt"]