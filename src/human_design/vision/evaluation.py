"""Offline metrics for BodyGraph extraction predictions.

Inputs are plain JSON-shaped dicts:

- prediction: the ``--json`` output of scripts/extract_bodygraph.py
  (keys: raw_vision, derived_chart_data, validation_result).
- golden: a hand-labeled case::

    {
      "case_id": "test1",
      "expected_raw_labels": {...same shape as raw Vision JSON...},
      "expected_derived_labels": {"basic_info": {...}, "active_gates": [...], ...},
      "expected_validation_result": {"is_valid": true, "warnings": []}
    }

Scoring conventions:

- You label it, we score it. Any golden field that is absent is simply not
  scored; there are no separate scope flags. An activation labeled JSON null
  means "unreadable in the image" and is scored via the null-respect metric.
- Gates are compared as ints ("34" == 34) and channels in canonical
  direction ("60-3" == "3-60"), so label formatting cannot silently zero
  a score.
- Warning comparison is by code set only. Golden warnings may be written as
  bare code strings ("MISSING_ACTIVATION") or as objects with a "code"
  field; unknown codes or other shapes raise ValueError instead of silently
  scoring zero. Severity / affects_validity are a deterministic function of
  the code (models.warning_defaults), so matching codes implies matching
  metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from human_design.vision.constants import ALL_CHANNELS, PLANETARY_FIELDS
from human_design.vision.models import ValidationCode

BASIC_INFO_FIELDS: tuple[str, ...] = (
    "profile",
    "type",
    "authority",
    "definition",
    "strategy",
    "not_self_theme",
    "signature",
)

_REVERSED_TO_CANONICAL = {
    "-".join(reversed(channel.split("-"))): channel for channel in ALL_CHANNELS
}

_KNOWN_WARNING_CODES = frozenset(code.value for code in ValidationCode)


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Metrics for one golden-label case."""

    case_id: str
    metrics: Mapping[str, float]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationSummary:
    """Per-case and aggregate evaluation metrics."""

    per_case: tuple[EvaluationCaseResult, ...]
    aggregate_metrics: Mapping[str, float]
    passed_thresholds: bool


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def evaluate_bodygraph_prediction(
    *,
    case_id: str,
    golden: Mapping[str, object],
    prediction: Mapping[str, object],
) -> EvaluationCaseResult:
    """Compare one saved prediction against one golden label mapping."""
    metrics: dict[str, float] = {}
    case_warnings: list[str] = []

    expected_raw = _mapping(golden.get("expected_raw_labels"))
    predicted_raw = _mapping(prediction.get("raw_vision"))
    metrics.update(_activation_metrics(expected_raw, predicted_raw))
    metrics.update(_labeled_set_metrics(expected_raw, predicted_raw, _VISUAL_SET_SPECS))

    expected_derived = _mapping(golden.get("expected_derived_labels"))
    if expected_derived:
        predicted_derived = _mapping(prediction.get("derived_chart_data"))
        if not predicted_derived:
            case_warnings.append("Prediction is missing derived_chart_data.")
        metrics.update(
            _labeled_set_metrics(expected_derived, predicted_derived, _DERIVED_SET_SPECS)
        )
        metrics.update(_basic_info_metrics(expected_derived, predicted_derived))

    expected_validation = _mapping(golden.get("expected_validation_result"))
    predicted_validation = _mapping(prediction.get("validation_result"))
    if isinstance(expected_validation.get("is_valid"), bool):
        metrics["validation_is_valid_exact_match"] = (
            1.0
            if predicted_validation.get("is_valid") is expected_validation["is_valid"]
            else 0.0
        )
    if "warnings" in expected_validation:
        metrics.update(
            warning_code_metrics(
                _tuple(expected_validation.get("warnings")),
                _tuple(predicted_validation.get("warnings")),
            )
        )

    return EvaluationCaseResult(
        case_id=case_id,
        metrics=metrics,
        warnings=tuple(case_warnings),
    )


def evaluate_bodygraph_predictions(
    *,
    golden_cases: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, object]],
    thresholds: Mapping[str, float] | None = None,
) -> EvaluationSummary:
    """Evaluate saved predictions and macro-average metrics across cases."""
    per_case: list[EvaluationCaseResult] = []
    for golden in golden_cases:
        case_id = _case_id(golden)
        case_result = evaluate_bodygraph_prediction(
            case_id=case_id,
            golden=golden,
            prediction=_mapping(predictions.get(case_id)),
        )
        if case_id not in predictions:
            case_result = EvaluationCaseResult(
                case_id=case_id,
                metrics=case_result.metrics,
                warnings=(*case_result.warnings, "Prediction missing for this case."),
            )
        per_case.append(case_result)

    aggregate_metrics = _aggregate_case_metrics(per_case)
    return EvaluationSummary(
        per_case=tuple(per_case),
        aggregate_metrics=aggregate_metrics,
        passed_thresholds=check_thresholds(aggregate_metrics, thresholds or {}),
    )


def check_thresholds(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> bool:
    """Return true when every configured metric meets its minimum threshold.

    A threshold on a metric name that never appears counts as 0.0 and fails,
    so a typo in a threshold name fails loudly instead of silently passing.
    """
    return all(
        metrics.get(metric_name, 0.0) >= threshold
        for metric_name, threshold in thresholds.items()
    )


# ---------------------------------------------------------------------------
# Activation metrics
# ---------------------------------------------------------------------------


def activation_exact_match(expected: object, predicted: object) -> bool:
    """Return true only when gate and line both match."""
    expected_activation = _activation(expected)
    predicted_activation = _activation(predicted)
    return expected_activation is not None and expected_activation == predicted_activation


def activation_match_rate(
    expected_column: Mapping[str, object],
    predicted_column: Mapping[str, object] | None,
) -> float:
    """Return exact activation matches divided by expected activation labels."""
    matches, eligible, _, _ = _activation_counts(expected_column, predicted_column or {})
    return matches / eligible if eligible else 1.0


def _activation_metrics(
    expected_raw: Mapping[str, object],
    predicted_raw: Mapping[str, object],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total_matches = total_eligible = total_respected = total_nulls = 0

    for column_name in ("personality", "design"):
        expected_column = _mapping(expected_raw.get(column_name))
        if not expected_column:
            continue
        predicted_column = _mapping(predicted_raw.get(column_name))
        matches, eligible, respected, nulls = _activation_counts(
            expected_column, predicted_column
        )
        total_matches += matches
        total_eligible += eligible
        total_respected += respected
        total_nulls += nulls
        if eligible:
            metrics[f"{column_name}_activation_exact_match_rate"] = matches / eligible

    if total_eligible:
        metrics["activation_exact_match_rate"] = total_matches / total_eligible
    if total_nulls:
        # Share of expected-null (unreadable) activations that the prediction
        # also left null. Below 1.0 means the model hallucinated values for
        # fields the golden label says cannot be read from the image.
        metrics["activation_null_respect_rate"] = total_respected / total_nulls
    return metrics


def _activation_counts(
    expected_column: Mapping[str, object],
    predicted_column: Mapping[str, object],
) -> tuple[int, int, int, int]:
    """Return (matches, eligible, nulls_respected, nulls_labeled)."""
    matches = eligible = respected = nulls = 0
    for field_name in PLANETARY_FIELDS:
        if field_name not in expected_column:
            continue  # unlabeled: not scored
        if expected_column[field_name] is None:
            nulls += 1
            if predicted_column.get(field_name) is None:
                respected += 1
            continue
        eligible += 1
        if activation_exact_match(
            expected_column[field_name], predicted_column.get(field_name)
        ):
            matches += 1
    return matches, eligible, respected, nulls


def _activation(value: object) -> tuple[int, int] | None:
    """Coerce '61.4' or {'gate': 61, 'line': 4} to (61, 4); else None."""
    if isinstance(value, str):
        gate, sep, line = value.strip().partition(".")
        if sep and gate.isdecimal() and line.isdecimal():
            return int(gate), int(line)
        return None
    if isinstance(value, Mapping):
        gate = value.get("gate")
        line = value.get("line")
        if (
            isinstance(gate, int)
            and isinstance(line, int)
            and not isinstance(gate, bool)
            and not isinstance(line, bool)
        ):
            return gate, line
    return None


# ---------------------------------------------------------------------------
# Set-valued metrics (gates / channels / centers), with normalization so
# label formatting ("34" vs 34, "60-3" vs "3-60") cannot silently zero scores
# ---------------------------------------------------------------------------


def precision_recall_f1(
    expected_set: Iterable[object],
    predicted_set: Iterable[object],
) -> tuple[float, float, float]:
    """Return set-valued precision, recall, and F1 using project conventions."""
    expected = set(expected_set)
    predicted = set(predicted_set)
    if not expected and not predicted:
        return 1.0, 1.0, 1.0

    true_positive_count = len(expected & predicted)
    precision = true_positive_count / len(predicted) if predicted else 0.0
    recall = true_positive_count / len(expected) if expected else 1.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _normalize_gate(value: object) -> object:
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value.strip())
    return value


def _normalize_channel(value: object) -> object:
    if not isinstance(value, str):
        return value
    channel = value.strip()
    return _REVERSED_TO_CANONICAL.get(channel, channel)


def _normalize_center(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


# (metric prefix, field name, normalizer)
_VISUAL_SET_SPECS = (
    ("visually_defined_center", "visually_defined_centers", _normalize_center),
    ("visually_active_gate", "visually_active_gates", _normalize_gate),
    ("visible_channel", "visible_colored_channels", _normalize_channel),
)
_DERIVED_SET_SPECS = (
    ("derived_center", "defined_centers", _normalize_center),
    ("active_gate", "active_gates", _normalize_gate),
    ("active_channel", "active_channels", _normalize_channel),
)


def _labeled_set_metrics(
    expected: Mapping[str, object],
    predicted: Mapping[str, object],
    specs: tuple,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix, field_name, normalize in specs:
        expected_values = _collection(expected, field_name)
        if expected_values is None:
            continue  # unlabeled: not scored
        predicted_values = _collection(predicted, field_name) or ()
        precision, recall, f1 = precision_recall_f1(
            (normalize(value) for value in expected_values),
            (normalize(value) for value in predicted_values),
        )
        metrics[f"{prefix}_precision"] = precision
        metrics[f"{prefix}_recall"] = recall
        metrics[f"{prefix}_f1"] = f1
    return metrics


# ---------------------------------------------------------------------------
# Basic info and warnings
# ---------------------------------------------------------------------------


def _basic_info_metrics(
    expected_derived: Mapping[str, object],
    predicted_derived: Mapping[str, object],
) -> dict[str, float]:
    expected_info = _mapping(expected_derived.get("basic_info"))
    if not expected_info:
        return {}
    predicted_info = _mapping(predicted_derived.get("basic_info"))

    metrics: dict[str, float] = {}
    scores: list[float] = []
    for field_name in BASIC_INFO_FIELDS:
        expected_value = expected_info.get(field_name)
        if expected_value is None:
            continue  # unlabeled: not scored
        score = 1.0 if predicted_info.get(field_name) == expected_value else 0.0
        metrics[f"{field_name}_exact_match"] = score
        scores.append(score)

    if scores:
        metrics["overall_basic_info_accuracy"] = macro_average(scores)
    return metrics


def warning_code_metrics(
    expected_warnings: Iterable[object],
    predicted_warnings: Iterable[object],
) -> dict[str, float]:
    """Compare warning codes as sets, without requiring message text matches.

    Raises ValueError for entries that are neither a code string nor a
    mapping with a string "code" field, or whose code is not a known
    ValidationCode, so bad labels fail loudly instead of scoring zero.
    """
    precision, recall, f1 = precision_recall_f1(
        _warning_codes(expected_warnings),
        _warning_codes(predicted_warnings),
    )
    return {
        "warning_code_precision": precision,
        "warning_code_recall": recall,
        "warning_code_f1": f1,
    }


def _warning_codes(warnings: Iterable[object]) -> set[str]:
    """Coerce warning entries to a code set, failing loudly on bad labels."""
    codes: set[str] = set()
    for index, warning in enumerate(warnings):
        code = _warning_code(warning)
        if code is None:
            raise ValueError(
                f"warnings[{index}] must be a warning code string or a "
                "mapping with a string 'code' field"
            )
        if code not in _KNOWN_WARNING_CODES:
            raise ValueError(
                f"warnings[{index}] has unknown warning code: {code!r}"
            )
        codes.add(code)
    return codes


def _warning_code(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        code = value.get("code")
        return code if isinstance(code, str) else None
    return None


# ---------------------------------------------------------------------------
# Aggregation and shared coercion helpers
# ---------------------------------------------------------------------------


def macro_average(values: Iterable[float]) -> float:
    """Return the arithmetic mean of per-case metric values."""
    value_tuple = tuple(values)
    return sum(value_tuple) / len(value_tuple) if value_tuple else 0.0


def _aggregate_case_metrics(
    per_case: Sequence[EvaluationCaseResult],
) -> dict[str, float]:
    """Macro-average each metric over the cases that actually report it.

    Metrics only labeled on a subset of cases are averaged over that subset,
    so their support differs; keep golden labels consistent per metric if you
    want the aggregate to reflect the whole set.
    """
    metric_names = sorted(
        {name for case_result in per_case for name in case_result.metrics}
    )
    return {
        name: macro_average(
            case_result.metrics[name]
            for case_result in per_case
            if name in case_result.metrics
        )
        for name in metric_names
    }


def _case_id(golden: Mapping[str, object]) -> str:
    value = golden.get("case_id")
    if not isinstance(value, str) or not value:
        raise ValueError("golden case case_id must be a non-empty string")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _tuple(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _collection(
    source: Mapping[str, object],
    field_name: str,
) -> tuple[object, ...] | None:
    value = source.get(field_name)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return None


__all__ = [
    "EvaluationCaseResult",
    "EvaluationSummary",
    "evaluate_bodygraph_prediction",
    "evaluate_bodygraph_predictions",
    "check_thresholds",
    "activation_exact_match",
    "activation_match_rate",
    "precision_recall_f1",
    "macro_average",
    "warning_code_metrics",
]