"""Run one local BodyGraph image through the Phase 2 extraction pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from human_design.vision.client import VisionClientError
from human_design.vision.config import load_vision_config
from human_design.vision.models import BodyGraphExtractionResult
from human_design.vision.parser import BodyGraphParseError
from human_design.vision.pipeline import extract_bodygraph


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract raw and deterministic facts from one BodyGraph image."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--mock-response",
        type=Path,
        help="Use a local mock Vision JSON fixture instead of calling the real API.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one machine-readable JSON object.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Configuration errors exit with 2 (same as argparse usage errors),
    # so callers can distinguish "your setup is wrong" from "this image failed".
    try:
        config = load_vision_config()
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    try:
        extraction_result = extract_bodygraph(
            image_path=args.image_path,
            config=config,
            mock_response_path=args.mock_response,
        )
    except (VisionClientError, BodyGraphParseError) as exc:
        print(f"BodyGraph extraction failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Unexpected errors are bugs: keep the full traceback for debugging.
        traceback.print_exc()
        print("BodyGraph extraction failed unexpectedly.", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "raw_vision": asdict(extraction_result.raw_vision),
            "derived_chart_data": asdict(extraction_result.derived_chart_data),
            "validation_result": asdict(extraction_result.validation_result),
        }
        print(json.dumps(payload, separators=(",", ":")))
    else:
        _print_human_readable(extraction_result)
    return 0


def _print_human_readable(result: BodyGraphExtractionResult) -> None:
    print("Raw Vision extraction:")
    print(json.dumps(asdict(result.raw_vision), indent=2))
    print("Derived chart data:")
    print(json.dumps(asdict(result.derived_chart_data), indent=2))
    validation_result = asdict(result.validation_result)
    print("Validation warnings:")
    print(json.dumps(validation_result["warnings"], indent=2))
    print(f"Validation is_valid: {json.dumps(validation_result['is_valid'])}")


if __name__ == "__main__":
    raise SystemExit(main())