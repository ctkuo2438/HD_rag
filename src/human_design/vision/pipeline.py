"""Typed orchestration for one Phase 2 BodyGraph extraction."""

from __future__ import annotations

from pathlib import Path

from human_design.vision.client import extract_bodygraph_raw_json
from human_design.vision.config import VisionConfig
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.models import BodyGraphExtractionResult
from human_design.vision.parser import parse_bodygraph_raw_extraction_json
from human_design.vision.validation import validate_bodygraph_extraction


def extract_bodygraph(
    *,
    image_path: Path,
    config: VisionConfig,
    mock_response_path: Path | None = None,
) -> BodyGraphExtractionResult:
    """Extract, parse, interpret, and validate one local BodyGraph image."""
    raw_json = extract_bodygraph_raw_json(
        image_path=image_path,
        config=config,
        mock_response_path=mock_response_path,
    )
    parse_result = parse_bodygraph_raw_extraction_json(raw_json)
    interpretation_result = interpret_bodygraph(parse_result.raw_vision)
    validation_result = validate_bodygraph_extraction(
        parse_result=parse_result,
        interpretation_result=interpretation_result,
    )
    return BodyGraphExtractionResult(
        raw_vision=parse_result.raw_vision,
        derived_chart_data=interpretation_result.derived_chart_data,
        validation_result=validation_result,
    )


__all__ = ["extract_bodygraph"]
