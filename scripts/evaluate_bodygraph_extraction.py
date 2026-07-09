#!/usr/bin/env python
"""Offline BodyGraph extraction evaluation script."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from human_design.vision.evaluation import (
    EvaluationCaseResult,
    EvaluationSummary,
    evaluate_bodygraph_predictions,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run offline evaluation from saved golden labels and predictions."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    golden_cases = _load_golden_cases(args.golden)
    predictions = _load_predictions(args.predictions)
    thresholds = dict(args.threshold or [])
    summary = evaluate_bodygraph_predictions(
        golden_cases=golden_cases,
        predictions=predictions,
        thresholds=thresholds,
    )

    if args.json:
        print(json.dumps(_summary_payload(summary), indent=2, sort_keys=True))
    else:
        _print_human_summary(summary)

    return 0 if summary.passed_thresholds else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved BodyGraph extraction predictions offline.",
    )
    parser.add_argument(
        "--golden",
        "--golden-labels",
        type=Path,
        required=True,
        dest="golden",
        help="Path to a golden labels JSON file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to a saved predictions JSON file.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        type=_threshold_arg,
        metavar="METRIC=VALUE",
        help="Minimum aggregate metric threshold. May be repeated.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser


def _threshold_arg(value: str) -> tuple[str, float]:
    metric_name, separator, raw_threshold = value.partition("=")
    if not separator or not metric_name:
        raise argparse.ArgumentTypeError("threshold must be formatted as metric=value")
    try:
        threshold = float(raw_threshold)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold value must be numeric") from exc
    return metric_name, threshold


def _load_golden_cases(path: Path) -> list[Mapping[str, object]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [_require_mapping(item, "golden case") for item in payload]
    if isinstance(payload, Mapping):
        cases = payload.get("cases")
        if isinstance(cases, list):
            return [_require_mapping(item, "golden case") for item in cases]
    raise ValueError("golden labels JSON must be a list or object with a cases list")


def _load_predictions(path: Path) -> dict[str, Mapping[str, object]]:
    payload = _load_json(path)
    if isinstance(payload, Mapping):
        nested_predictions = payload.get("predictions")
        if isinstance(nested_predictions, Mapping):
            return _prediction_mapping(nested_predictions)

        cases = payload.get("cases")
        if isinstance(cases, list):
            return _prediction_sequence(cases)

        if isinstance(payload.get("case_id"), str):
            return {payload["case_id"]: payload}

        return _prediction_mapping(payload)

    if isinstance(payload, list):
        return _prediction_sequence(payload)

    raise ValueError("predictions JSON must be a mapping or list")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_mapping(
    predictions: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    normalized: dict[str, Mapping[str, object]] = {}
    for case_id, prediction in predictions.items():
        normalized[case_id] = _require_mapping(prediction, "prediction")
    return normalized


def _prediction_sequence(cases: Sequence[object]) -> dict[str, Mapping[str, object]]:
    normalized: dict[str, Mapping[str, object]] = {}
    for index, prediction in enumerate(cases):
        prediction_mapping = _require_mapping(prediction, "prediction")
        case_id = prediction_mapping.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            case_id = f"case_{index + 1}"
        normalized[case_id] = prediction_mapping
    return normalized


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _print_human_summary(summary: EvaluationSummary) -> None:
    print("Per-case metrics")
    for case_result in summary.per_case:
        _print_case_metrics(case_result)

    print("Aggregate metrics")
    for metric_name, value in sorted(summary.aggregate_metrics.items()):
        print(f"  {metric_name}: {value:.6f}")
    print(f"Thresholds passed: {summary.passed_thresholds}")


def _print_case_metrics(case_result: EvaluationCaseResult) -> None:
    print(f"{case_result.case_id}:")
    for metric_name, value in sorted(case_result.metrics.items()):
        print(f"  {metric_name}: {value:.6f}")
    for warning in case_result.warnings:
        print(f"  warning: {warning}")


def _summary_payload(summary: EvaluationSummary) -> dict[str, Any]:
    return {
        "per_case": [
            {
                "case_id": case_result.case_id,
                "metrics": dict(case_result.metrics),
                "passed_thresholds": case_result.passed_thresholds,
                "warnings": list(case_result.warnings),
            }
            for case_result in summary.per_case
        ],
        "aggregate_metrics": dict(summary.aggregate_metrics),
        "passed_thresholds": summary.passed_thresholds,
    }


if __name__ == "__main__":
    raise SystemExit(main())
