#!/usr/bin/env python
"""Offline BodyGraph extraction evaluation script."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from human_design.vision.constants import PLANETARY_FIELDS
from human_design.vision.evaluation import (
    EvaluationCaseResult,
    EvaluationSummary,
    evaluate_bodygraph_predictions,
)
from human_design.vision.models import ValidationCode


_PLANETARY_FIELDS = set(PLANETARY_FIELDS)
_RAW_LABEL_FIELDS = {
    "personality",
    "design",
    "visually_defined_centers",
    "visually_active_gates",
    "visible_colored_channels",
    "uncertain_items",
}
_DERIVED_LABEL_FIELDS = {
    "basic_info",
    "active_gates",
    "active_channels",
    "defined_centers",
}
_BASIC_INFO_FIELDS = {
    "type",
    "authority",
    "profile",
    "strategy",
    "definition",
    "not_self_theme",
    "signature",
}
_WARNING_FIELDS = {
    "code",
    "message",
    "severity",
    "affects_validity",
    "source",
    "field_path",
}
_KNOWN_WARNING_CODES = frozenset(code.value for code in ValidationCode)


def main(argv: Sequence[str] | None = None) -> int:
    """Run offline evaluation from saved golden labels and predictions."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Input problems (missing file, bad JSON, schema violations) exit with 2,
    # so callers can distinguish them from a threshold failure (exit 1).
    try:
        golden_cases = _load_golden_cases(args.golden)
        predictions = _load_predictions(args.predictions)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Invalid evaluation input: {exc}", file=sys.stderr)
        return 2

    thresholds = dict(args.threshold or [])
    # evaluate_bodygraph_predictions raises ValueError on bad labels (e.g. an
    # unknown warning code inside a prediction file), which is still an input
    # problem: exit 2, same as load-time failures.
    try:
        summary = evaluate_bodygraph_predictions(
            golden_cases=golden_cases,
            predictions=predictions,
            thresholds=thresholds,
        )
    except ValueError as exc:
        print(f"Invalid evaluation input: {exc}", file=sys.stderr)
        return 2

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
    root = _require_mapping(payload, "golden labels")
    _require_exact_keys(
        root,
        {
            "schema_version",
            "documentation",
            "recommended_sample_coverage",
            "cases",
        },
        "golden labels",
    )
    if root["schema_version"] != "phase2_golden_labels_v2":
        raise ValueError("golden labels schema_version must be phase2_golden_labels_v2")
    _require_mapping(root["documentation"], "golden labels documentation")
    _require_mapping(
        root["recommended_sample_coverage"],
        "golden labels recommended_sample_coverage",
    )
    cases = root["cases"]
    if not isinstance(cases, list):
        raise ValueError("golden labels cases must be a list")

    validated_cases: list[Mapping[str, object]] = []
    seen_case_ids: set[str] = set()
    for index, item in enumerate(cases):
        case_id, case = _validate_golden_case(item, index)
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate golden case_id: {case_id}")
        seen_case_ids.add(case_id)
        validated_cases.append(case)
    return validated_cases


def _load_predictions(path: Path) -> dict[str, Mapping[str, object]]:
    payload = _load_json(path)
    root = _require_mapping(payload, "predictions")
    _require_exact_keys(root, {"schema_version", "predictions"}, "predictions")
    if root["schema_version"] != "phase2_predictions_v1":
        raise ValueError("predictions schema_version must be phase2_predictions_v1")
    entries = root["predictions"]
    if not isinstance(entries, list):
        raise ValueError("predictions must be a list")

    normalized: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(entries):
        prediction = _require_mapping(value, f"predictions[{index}]")
        _require_exact_keys(
            prediction,
            {"case_id", "raw_vision", "derived_chart_data", "validation_result"},
            f"predictions[{index}]",
        )
        case_id = prediction["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"predictions[{index}].case_id must be a non-empty string")
        if case_id in normalized:
            raise ValueError(f"duplicate prediction case_id: {case_id}")
        _require_mapping(prediction["raw_vision"], f"predictions[{index}].raw_vision")
        _require_mapping(
            prediction["derived_chart_data"],
            f"predictions[{index}].derived_chart_data",
        )
        _require_mapping(
            prediction["validation_result"],
            f"predictions[{index}].validation_result",
        )
        normalized[case_id] = prediction
    return normalized


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    missing = expected - set(value)
    if missing:
        raise ValueError(
            f"{label} missing required fields: {', '.join(sorted(missing))}"
        )
    extra = set(value) - expected
    if extra:
        raise ValueError(
            f"{label} has unexpected fields: {', '.join(sorted(extra))}"
        )


def _validate_golden_case(
    value: object,
    index: int,
) -> tuple[str, Mapping[str, object]]:
    case = _require_mapping(value, f"golden labels cases[{index}]")
    _require_exact_keys(
        case,
        {
            "case_id",
            "image_filename",
            "label_source",
            "notes",
            "evaluation_scope",
            "expected_raw_labels",
            "expected_derived_labels",
            "expected_validation_result",
        },
        f"golden labels cases[{index}]",
    )
    case_id = case["case_id"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"golden labels cases[{index}].case_id must be a non-empty string")
    for field in ("image_filename", "label_source", "notes"):
        if not isinstance(case[field], str):
            raise ValueError(f"golden labels cases[{index}].{field} must be a string")
    scope = _require_mapping(
        case["evaluation_scope"],
        f"golden labels cases[{index}].evaluation_scope",
    )
    _require_exact_keys(
        scope,
        {"include_raw_visual_metrics", "include_derived_metrics"},
        f"golden labels cases[{index}].evaluation_scope",
    )
    for flag in ("include_raw_visual_metrics", "include_derived_metrics"):
        if not isinstance(scope[flag], bool):
            raise ValueError(
                f"golden labels cases[{index}].evaluation_scope.{flag} must be a bool"
            )
    include_derived_metrics = scope["include_derived_metrics"]
    derived = case["expected_derived_labels"]
    if include_derived_metrics and derived is None:
        raise ValueError(
            "expected_derived_labels must be an object when "
            "include_derived_metrics is true"
        )
    if not include_derived_metrics and derived is not None:
        raise ValueError(
            "expected_derived_labels must be null when "
            "include_derived_metrics is false"
        )
    raw_labels = _require_mapping(
        case["expected_raw_labels"],
        f"golden labels cases[{index}].expected_raw_labels",
    )
    _require_exact_keys(
        raw_labels,
        _RAW_LABEL_FIELDS,
        f"golden labels cases[{index}].expected_raw_labels",
    )
    for column_name in ("personality", "design"):
        column = _require_mapping(
            raw_labels[column_name],
            f"golden labels cases[{index}].expected_raw_labels.{column_name}",
        )
        _require_exact_keys(
            column,
            _PLANETARY_FIELDS,
            f"golden labels cases[{index}].expected_raw_labels.{column_name}",
        )
    for field in (
        "visually_defined_centers",
        "visually_active_gates",
        "visible_colored_channels",
        "uncertain_items",
    ):
        if not isinstance(raw_labels[field], list):
            raise ValueError(
                f"golden labels cases[{index}].expected_raw_labels.{field} must be a list"
            )
    if derived is not None:
        derived_mapping = _require_mapping(
            derived,
            f"golden labels cases[{index}].expected_derived_labels",
        )
        _require_exact_keys(
            derived_mapping,
            _DERIVED_LABEL_FIELDS,
            f"golden labels cases[{index}].expected_derived_labels",
        )
        basic_info = _require_mapping(
            derived_mapping["basic_info"],
            f"golden labels cases[{index}].expected_derived_labels.basic_info",
        )
        _require_exact_keys(
            basic_info,
            _BASIC_INFO_FIELDS,
            f"golden labels cases[{index}].expected_derived_labels.basic_info",
        )
        for field in ("active_gates", "active_channels", "defined_centers"):
            if not isinstance(derived_mapping[field], list):
                raise ValueError(
                    f"golden labels cases[{index}].expected_derived_labels.{field} must be a list"
                )
    validation_result = _require_mapping(
        case["expected_validation_result"],
        f"golden labels cases[{index}].expected_validation_result",
    )
    _require_exact_keys(
        validation_result,
        {"is_valid", "warnings"},
        f"golden labels cases[{index}].expected_validation_result",
    )
    if not isinstance(validation_result["is_valid"], bool):
        raise ValueError(
            f"golden labels cases[{index}].expected_validation_result.is_valid must be a bool"
        )
    warnings = validation_result["warnings"]
    if not isinstance(warnings, list):
        raise ValueError(
            f"golden labels cases[{index}].expected_validation_result.warnings must be a list"
        )
    # A golden warning may be a bare code string ("MISSING_ACTIVATION") or a
    # full warning object; either way the code must be a known ValidationCode
    # so a labeling typo fails at load time (exit 2), not mid-evaluation.
    for warning_index, warning_value in enumerate(warnings):
        label = (
            f"golden labels cases[{index}].expected_validation_result"
            f".warnings[{warning_index}]"
        )
        if isinstance(warning_value, str):
            code = warning_value
        else:
            warning = _require_mapping(warning_value, label)
            _require_exact_keys(warning, _WARNING_FIELDS, label)
            code = warning.get("code")
        if not isinstance(code, str) or code not in _KNOWN_WARNING_CODES:
            raise ValueError(f"{label} has unknown warning code: {code!r}")
    return case_id, case


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
                "warnings": list(case_result.warnings),
            }
            for case_result in summary.per_case
        ],
        "aggregate_metrics": dict(summary.aggregate_metrics),
        "passed_thresholds": summary.passed_thresholds,
    }


if __name__ == "__main__":
    raise SystemExit(main())