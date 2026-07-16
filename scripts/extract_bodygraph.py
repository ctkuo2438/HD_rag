"""Run one local BodyGraph image through the Phase 2 extraction pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from human_design.vision.client import VisionClientError
from human_design.vision.config import load_vision_config
from human_design.vision.parser import BodyGraphParseError
from human_design.vision.pipeline import extract_bodygraph


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract raw and deterministic facts from one BodyGraph image."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--mock-response", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one machine-readable JSON object.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_vision_config()
        extraction_result = extract_bodygraph(
            image_path=args.image_path,
            config=config,
            mock_response_path=args.mock_response,
        )
    except (VisionClientError, BodyGraphParseError, ValueError) as exc:
        print(f"BodyGraph extraction failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"BodyGraph extraction failed unexpectedly ({type(exc).__name__}).",
            file=sys.stderr,
        )
        return 1

    payload = {
        "raw_vision": asdict(extraction_result.raw_vision),
        "derived_chart_data": asdict(extraction_result.derived_chart_data),
        "validation_result": asdict(extraction_result.validation),
    }
    if args.json:
        print(json.dumps(payload, separators=(",", ":")))
    else:
        _print_human_readable(payload)
    return 0


def _print_human_readable(payload: dict[str, object]) -> None:
    print("Raw Vision extraction:")
    print(json.dumps(payload["raw_vision"], indent=2))
    print("Derived chart data:")
    print(json.dumps(payload["derived_chart_data"], indent=2))
    validation = payload["validation_result"]
    assert isinstance(validation, dict)
    print("Validation warnings:")
    print(json.dumps(validation["warnings"], indent=2))
    print(f"Validation is_valid: {json.dumps(validation['is_valid'])}")


if __name__ == "__main__":
    raise SystemExit(main())
