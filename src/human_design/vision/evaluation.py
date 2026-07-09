"""Offline metrics for BodyGraph extraction predictions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


PLANETARY_FIELDS: tuple[str, ...] = (
    "sun",
    "earth",
    "north_node",
    "south_node",
    "moon",
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "pluto",
)

BASIC_INFO_FIELDS: tuple[str, ...] = (
    "profile",
    "type",
    "authority",
    "definition",
    "strategy",
    "not_self_theme",
    "signature",
)


@dataclass(frozen=True)
class MetricResult:
    """A named metric value with the number of eligible cases."""

    name: str
    value: float
    eligible_count: int


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Metrics for one golden-label case."""

    case_id: str
    metrics: Mapping[str, float]
    passed_thresholds: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationSummary:
    """Per-case and aggregate evaluation metrics."""

    per_case: tuple[EvaluationCaseResult, ...]
    aggregate_metrics: Mapping[str, float]
    passed_thresholds: bool


def evaluate_bodygraph_prediction(
    *,
    case_id: str,
    golden: Mapping[str, object],
    prediction: Mapping[str, object],
) -> EvaluationCaseResult:
    """Compare one saved prediction against one golden label mapping."""
    metrics: dict[str, float] = {}
    warnings: list[str] = []

    expected_raw = _expected_raw_labels(golden)
    predicted_raw = _prediction_raw_labels(prediction)
    metrics.update(_activation_metrics(expected_raw, predicted_raw))
    metrics.update(_visual_metrics(expected_raw, predicted_raw))

    expected_derived = _expected_derived_labels(golden)
    predicted_derived = _prediction_derived_chart_data(prediction)
    if _include_derived_metrics(golden, expected_derived):
        if predicted_derived is None:
            warnings.append("Prediction is missing derived_chart_data.")
            predicted_derived = {}
        metrics.update(_derived_metrics(expected_derived, predicted_derived))
        metrics.update(_basic_info_metrics(expected_derived, predicted_derived))

    expected_warnings = _expected_warnings(golden)
    predicted_warnings = _prediction_warnings(prediction)
    metrics.update(warning_code_metrics(expected_warnings, predicted_warnings))
    if _has_warning_metadata(expected_warnings) or _has_warning_metadata(
        predicted_warnings
    ):
        metrics["warning_metadata_exact_match_rate"] = _warning_metadata_match_rate(
            expected_warnings,
            predicted_warnings,
        )

    return EvaluationCaseResult(
        case_id=case_id,
        metrics=metrics,
        passed_thresholds=True,
        warnings=tuple(warnings),
    )


def evaluate_bodygraph_predictions(
    *,
    golden_cases: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, object]],
    thresholds: Mapping[str, float] | None = None,
) -> EvaluationSummary:
    """Evaluate saved predictions and macro-average metrics across cases."""
    per_case: list[EvaluationCaseResult] = []
    for index, golden in enumerate(golden_cases):
        case_id = _case_id(golden, index)
        prediction = predictions.get(case_id, {})
        per_case.append(
            evaluate_bodygraph_prediction(
                case_id=case_id,
                golden=golden,
                prediction=prediction,
            )
        )

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
    """Return true when every configured metric meets its minimum threshold."""
    return all(metrics.get(metric_name, 0.0) >= threshold for metric_name, threshold in thresholds.items())


def activation_exact_match(expected: object, predicted: object) -> bool:
    """Return true only when gate and line both match."""
    expected_activation = _coerce_activation(expected)
    predicted_activation = _coerce_activation(predicted)
    return (
        expected_activation is not None
        and predicted_activation is not None
        and expected_activation == predicted_activation
    )


def activation_match_rate(
    expected_column: Mapping[str, object],
    predicted_column: Mapping[str, object] | None,
) -> float:
    """Return exact activation matches divided by expected activation labels."""
    matches, eligible_count = _activation_match_counts(expected_column, predicted_column)
    if eligible_count == 0:
        return 1.0
    return matches / eligible_count


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
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def macro_average(values: Iterable[float]) -> float:
    """Return the arithmetic mean of per-case metric values."""
    value_tuple = tuple(values)
    if not value_tuple:
        return 0.0
    return sum(value_tuple) / len(value_tuple)


def warning_code_metrics(
    expected_warnings: Iterable[object],
    predicted_warnings: Iterable[object],
) -> dict[str, float]:
    """Compare warning codes as sets, without requiring message text matches."""
    precision, recall, f1 = precision_recall_f1(
        _warning_codes(expected_warnings),
        _warning_codes(predicted_warnings),
    )
    return {
        "warning_code_precision": precision,
        "warning_code_recall": recall,
        "warning_code_f1": f1,
    }


def _case_id(golden: Mapping[str, object], index: int) -> str:
    for key in ("case_id", "id"):
        value = golden.get(key)
        if isinstance(value, str) and value:
            return value
    return f"case_{index + 1}"


def _expected_raw_labels(golden: Mapping[str, object]) -> Mapping[str, object]:
    raw_labels = _mapping_or_none(golden.get("expected_raw_labels"))
    if raw_labels is not None:
        return raw_labels
    raw_vision = _mapping_or_none(golden.get("raw_vision"))
    if raw_vision is not None:
        return raw_vision
    return golden


def _prediction_raw_labels(prediction: Mapping[str, object]) -> Mapping[str, object]:
    raw_vision = _mapping_or_none(prediction.get("raw_vision"))
    if raw_vision is not None:
        return raw_vision
    return prediction


def _expected_derived_labels(
    golden: Mapping[str, object],
) -> Mapping[str, object] | None:
    for key in ("expected_derived_labels", "expected_derived", "derived_chart_data"):
        if key in golden:
            return _mapping_or_none(golden.get(key))
    return None


def _prediction_derived_chart_data(
    prediction: Mapping[str, object],
) -> Mapping[str, object] | None:
    for key in ("derived_chart_data", "derived", "chart_data"):
        if key in prediction:
            return _mapping_or_none(prediction.get(key))
    return prediction


def _include_derived_metrics(
    golden: Mapping[str, object],
    expected_derived: Mapping[str, object] | None,
) -> bool:
    scope = _mapping_or_none(golden.get("evaluation_scope"))
    if scope is not None and scope.get("include_derived_metrics") is False:
        return False
    return expected_derived is not None


def _activation_metrics(
    expected_raw: Mapping[str, object],
    predicted_raw: Mapping[str, object],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total_matches = 0
    total_eligible = 0

    for column_name, metric_name in (
        ("personality", "personality_activation_exact_match_rate"),
        ("design", "design_activation_exact_match_rate"),
    ):
        expected_column = _mapping_or_none(expected_raw.get(column_name))
        predicted_column = _mapping_or_none(predicted_raw.get(column_name))
        if expected_column is None:
            continue
        matches, eligible_count = _activation_match_counts(
            expected_column,
            predicted_column,
        )
        if eligible_count == 0:
            continue
        total_matches += matches
        total_eligible += eligible_count
        metrics[metric_name] = matches / eligible_count

    if total_eligible:
        metrics["activation_exact_match_rate"] = total_matches / total_eligible
    return metrics


def _activation_match_counts(
    expected_column: Mapping[str, object],
    predicted_column: Mapping[str, object] | None,
) -> tuple[int, int]:
    matches = 0
    eligible_count = 0
    predicted = predicted_column or {}

    for field_name in PLANETARY_FIELDS:
        if field_name not in expected_column or expected_column[field_name] is None:
            continue
        eligible_count += 1
        if activation_exact_match(expected_column[field_name], predicted.get(field_name)):
            matches += 1

    return matches, eligible_count


def _visual_metrics(
    expected_raw: Mapping[str, object],
    predicted_raw: Mapping[str, object],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    _add_set_metrics_if_available(
        metrics,
        "visually_defined_center",
        _collection_or_none(expected_raw, "visually_defined_centers"),
        _collection_or_none(predicted_raw, "visually_defined_centers"),
    )
    _add_set_metrics_if_available(
        metrics,
        "visually_active_gate",
        _collection_or_none(expected_raw, "visually_active_gates"),
        _collection_or_none(predicted_raw, "visually_active_gates"),
    )
    _add_set_metrics_if_available(
        metrics,
        "visible_channel",
        _collection_or_none(expected_raw, "visible_colored_channels"),
        _collection_or_none(predicted_raw, "visible_colored_channels"),
    )
    return metrics


def _derived_metrics(
    expected_derived: Mapping[str, object] | None,
    predicted_derived: Mapping[str, object],
) -> dict[str, float]:
    if expected_derived is None:
        return {}

    metrics: dict[str, float] = {}
    _add_required_set_metrics_if_expected_available(
        metrics,
        "derived_center",
        _collection_or_none(expected_derived, "defined_centers"),
        _collection_or_empty(predicted_derived, "defined_centers"),
    )
    _add_required_set_metrics_if_expected_available(
        metrics,
        "active_gate",
        _collection_or_none(expected_derived, "active_gates"),
        _collection_or_empty(predicted_derived, "active_gates"),
    )
    _add_required_set_metrics_if_expected_available(
        metrics,
        "active_channel",
        _collection_or_none(expected_derived, "active_channels"),
        _collection_or_empty(predicted_derived, "active_channels"),
    )
    return metrics


def _add_set_metrics_if_available(
    metrics: dict[str, float],
    prefix: str,
    expected_values: tuple[object, ...] | None,
    predicted_values: tuple[object, ...] | None,
) -> None:
    if expected_values is None or predicted_values is None:
        return
    precision, recall, f1 = precision_recall_f1(expected_values, predicted_values)
    metrics[f"{prefix}_precision"] = precision
    metrics[f"{prefix}_recall"] = recall
    metrics[f"{prefix}_f1"] = f1


def _add_required_set_metrics_if_expected_available(
    metrics: dict[str, float],
    prefix: str,
    expected_values: tuple[object, ...] | None,
    predicted_values: tuple[object, ...],
) -> None:
    if expected_values is None:
        return
    precision, recall, f1 = precision_recall_f1(expected_values, predicted_values)
    metrics[f"{prefix}_precision"] = precision
    metrics[f"{prefix}_recall"] = recall
    metrics[f"{prefix}_f1"] = f1


def _basic_info_metrics(
    expected_derived: Mapping[str, object] | None,
    predicted_derived: Mapping[str, object],
) -> dict[str, float]:
    if expected_derived is None:
        return {}

    expected_basic_info = _basic_info_mapping(expected_derived)
    predicted_basic_info = _basic_info_mapping(predicted_derived)
    if expected_basic_info is None:
        return {}
    predicted = predicted_basic_info or {}

    metrics: dict[str, float] = {}
    exact_matches: list[float] = []
    for field_name in BASIC_INFO_FIELDS:
        expected_value = _value(expected_basic_info, field_name)
        if expected_value is None:
            continue
        predicted_value = _value(predicted, field_name)
        metric_value = 1.0 if predicted_value == expected_value else 0.0
        metrics[f"{field_name}_exact_match"] = metric_value
        exact_matches.append(metric_value)

    if exact_matches:
        metrics["overall_basic_info_accuracy"] = macro_average(exact_matches)
    return metrics


def _basic_info_mapping(source: Mapping[str, object]) -> Mapping[str, object] | None:
    nested = _mapping_or_none(source.get("basic_info"))
    if nested is not None:
        return nested
    if any(field_name in source for field_name in BASIC_INFO_FIELDS):
        return source
    return None


def _expected_warnings(golden: Mapping[str, object]) -> tuple[object, ...]:
    validation_result = _mapping_or_none(golden.get("expected_validation_result"))
    if validation_result is not None and "warnings" in validation_result:
        return _tuple_or_empty(validation_result.get("warnings"))
    if "expected_warning_codes" in golden:
        return _tuple_or_empty(golden.get("expected_warning_codes"))
    return _tuple_or_empty(golden.get("warnings"))


def _prediction_warnings(prediction: Mapping[str, object]) -> tuple[object, ...]:
    for key in ("validation_result", "validation"):
        validation_result = _mapping_or_none(prediction.get(key))
        if validation_result is not None and "warnings" in validation_result:
            return _tuple_or_empty(validation_result.get("warnings"))
    return _tuple_or_empty(prediction.get("warnings"))


def _warning_codes(warnings: Iterable[object]) -> set[str]:
    return {
        code
        for warning in warnings
        if (code := _warning_code(warning)) is not None
    }


def _warning_code(warning: object) -> str | None:
    if isinstance(warning, str):
        return warning
    if isinstance(warning, Mapping):
        code = warning.get("code")
        return _string_value(code)
    code = getattr(warning, "code", None)
    return _string_value(code)


def _has_warning_metadata(warnings: Iterable[object]) -> bool:
    return any(_warning_metadata_key(warning) is not None for warning in warnings)


def _warning_metadata_match_rate(
    expected_warnings: Iterable[object],
    predicted_warnings: Iterable[object],
) -> float:
    expected = {
        metadata
        for warning in expected_warnings
        if (metadata := _warning_metadata_key(warning)) is not None
    }
    predicted = {
        metadata
        for warning in predicted_warnings
        if (metadata := _warning_metadata_key(warning)) is not None
    }
    if not expected and not predicted:
        return 1.0
    if not expected:
        return 0.0
    return len(expected & predicted) / len(expected)


def _warning_metadata_key(warning: object) -> tuple[str, str | None, bool | None, str | None] | None:
    if isinstance(warning, Mapping):
        code = _string_value(warning.get("code"))
        if code is None:
            return None
        return (
            code,
            _string_value(warning.get("severity")),
            _bool_or_none(warning.get("affects_validity")),
            _string_value(warning.get("source")),
        )

    code = _string_value(getattr(warning, "code", None))
    if code is None:
        return None
    return (
        code,
        _string_value(getattr(warning, "severity", None)),
        _bool_or_none(getattr(warning, "affects_validity", None)),
        _string_value(getattr(warning, "source", None)),
    )


def _aggregate_case_metrics(
    per_case: Sequence[EvaluationCaseResult],
) -> dict[str, float]:
    metric_names = sorted(
        {
            metric_name
            for case_result in per_case
            for metric_name in case_result.metrics
        }
    )
    return {
        metric_name: macro_average(
            case_result.metrics[metric_name]
            for case_result in per_case
            if metric_name in case_result.metrics
        )
        for metric_name in metric_names
    }


def _coerce_activation(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        gate_line = value.strip().split(".", maxsplit=1)
        if len(gate_line) != 2:
            return None
        gate, line = gate_line
        if not gate.isdecimal() or not line.isdecimal():
            return None
        return int(gate), int(line)
    if isinstance(value, Mapping):
        gate = value.get("gate")
        line = value.get("line")
        if isinstance(gate, bool) or isinstance(line, bool):
            return None
        if isinstance(gate, int) and isinstance(line, int):
            return gate, line
        return None

    gate = getattr(value, "gate", None)
    line = getattr(value, "line", None)
    if isinstance(gate, bool) or isinstance(line, bool):
        return None
    if isinstance(gate, int) and isinstance(line, int):
        return gate, line
    return None


def _collection_or_none(
    source: Mapping[str, object],
    field_name: str,
) -> tuple[object, ...] | None:
    if field_name not in source:
        return None
    value = source[field_name]
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _collection_or_empty(
    source: Mapping[str, object],
    field_name: str,
) -> tuple[object, ...]:
    return _collection_or_none(source, field_name) or ()


def _mapping_or_none(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _tuple_or_empty(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()


def _value(source: Mapping[str, object], field_name: str) -> object:
    return source.get(field_name)


def _string_value(value: object) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    if isinstance(value, str):
        return value
    return str(value)


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


__all__ = [
    "MetricResult",
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
