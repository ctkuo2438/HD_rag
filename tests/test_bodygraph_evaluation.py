import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from human_design.vision.constants import CANONICAL_CENTERS, PLANETARY_FIELDS
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.models import (
    ValidationCode,
    ValidationSeverity,
    ValidationSource,
    warning_defaults,
)
from human_design.vision.parser import parse_bodygraph_raw_extraction_json


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_LABELS_PATH = REPO_ROOT / "data/bodygraph_samples/golden_labels.example.json"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

BASIC_INFO_FIELDS = (
    "profile",
    "type",
    "authority",
    "definition",
    "strategy",
    "not_self_theme",
    "signature",
)

REQUIRED_CASE_KEYS = {
    "case_id",
    "image_filename",
    "label_source",
    "notes",
    "evaluation_scope",
    "expected_raw_labels",
    "expected_derived_labels",
    "expected_validation_result",
}

REQUIRED_RAW_LABEL_KEYS = {
    "personality",
    "design",
    "visually_active_gates",
    "visually_defined_centers",
    "visible_colored_channels",
    "uncertain_items",
}

REQUIRED_DERIVED_LABEL_KEYS = {
    "basic_info",
    "active_gates",
    "active_channels",
    "defined_centers",
}

DISALLOWED_MISSING_ACTIVATION_SENTINELS = ("", False, 0, {})


def _load_golden_labels() -> dict[str, Any]:
    assert GOLDEN_LABELS_PATH.exists()
    return json.loads(GOLDEN_LABELS_PATH.read_text(encoding="utf-8"))


def _cases() -> list[dict[str, Any]]:
    data = _load_golden_labels()
    cases = data["cases"]
    assert isinstance(cases, list)
    return cases


def _raw_parser_payload(raw_labels: dict[str, Any]) -> dict[str, object]:
    return {
        "personality": raw_labels["personality"],
        "design": raw_labels["design"],
        "visually_defined_centers": raw_labels.get("visually_defined_centers", []),
        "visually_active_gates": raw_labels.get("visually_active_gates", []),
        "visible_colored_channels": raw_labels.get("visible_colored_channels", []),
        "uncertain_items": raw_labels.get("uncertain_items", []),
    }


def _parse_raw_labels(raw_labels: dict[str, Any]):
    return parse_bodygraph_raw_extraction_json(json.dumps(_raw_parser_payload(raw_labels)))


def _full_raw_labels() -> dict[str, Any]:
    return {
        "personality": {
            "sun": "61.4",
            "earth": "62.4",
            "north_node": "3.1",
            "south_node": "60.2",
            "moon": "10.3",
            "mercury": "34.2",
            "venus": "1.1",
            "mars": "1.2",
            "jupiter": "1.3",
            "saturn": "1.4",
            "uranus": "1.5",
            "neptune": "1.6",
            "pluto": "1.1",
        },
        "design": {
            "sun": "32.6",
            "earth": "1.2",
            "north_node": "1.3",
            "south_node": "1.4",
            "moon": "1.5",
            "mercury": "1.6",
            "venus": "1.1",
            "mars": "1.2",
            "jupiter": "1.3",
            "saturn": "1.4",
            "uranus": "1.5",
            "neptune": "1.6",
            "pluto": "1.1",
        },
        "visually_active_gates": [3, 10, 32, 34, 60, 61, 62, 1],
        "visually_defined_centers": ["G", "Sacral", "Root"],
        "visible_colored_channels": ["3-60", "10-34"],
        "uncertain_items": [],
    }


def _basic_info() -> dict[str, str]:
    return {
        "profile": "4/6",
        "type": "Generator",
        "authority": "Sacral",
        "definition": "Single Definition",
        "strategy": "To Respond",
        "not_self_theme": "Frustration",
        "signature": "Satisfaction",
    }


def _derived_labels() -> dict[str, Any]:
    return {
        "active_gates": [1, 3, 10, 32, 34, 60, 61, 62],
        "active_channels": ["3-60", "10-34"],
        "defined_centers": ["G", "Sacral", "Root"],
        "basic_info": _basic_info(),
    }


def _warnings_are_valid(
    warnings: list[object] | None,
) -> bool:
    for warning in warnings or []:
        if isinstance(warning, str):
            _, affects_validity = warning_defaults(ValidationCode(warning))
            if affects_validity:
                return False
        elif isinstance(warning, dict) and warning.get("affects_validity") is True:
            return False
    return True


def _golden_case(
    case_id: str = "case_001",
    *,
    include_derived_metrics: bool = True,
    include_raw_visual_metrics: bool = True,
    expected_is_valid: bool | None = None,
    warning_entries: list[object] | None = None,
) -> dict[str, Any]:
    evaluation_scope: dict[str, object] = {
        "include_raw_visual_metrics": include_raw_visual_metrics,
        "include_derived_metrics": include_derived_metrics,
    }

    return {
        "case_id": case_id,
        "image_filename": "test1.png",
        "label_source": "sanitized synthetic fixture",
        "notes": "Synthetic evaluation case.",
        "evaluation_scope": evaluation_scope,
        "expected_raw_labels": _full_raw_labels(),
        "expected_derived_labels": _derived_labels()
        if include_derived_metrics
        else None,
        "expected_validation_result": {
            "is_valid": (
                _warnings_are_valid(warning_entries)
                if expected_is_valid is None
                else expected_is_valid
            ),
            "warnings": warning_entries or [],
        },
    }


def _prediction(
    *,
    personality: dict[str, str | None] | None = None,
    design: dict[str, str | None] | None = None,
    derived_chart_data: dict[str, Any] | None = None,
    warnings: list[dict[str, object]] | None = None,
    is_valid: bool | None = None,
    include_is_valid: bool = True,
) -> dict[str, Any]:
    raw_labels = _full_raw_labels()
    validation_result: dict[str, object] = {"warnings": warnings or []}
    if include_is_valid:
        validation_result["is_valid"] = (
            _warnings_are_valid(warnings) if is_valid is None else is_valid
        )

    return {
        "raw_vision": {
            "personality": personality if personality is not None else raw_labels["personality"],
            "design": design if design is not None else raw_labels["design"],
            "visually_active_gates": raw_labels["visually_active_gates"],
            "visually_defined_centers": raw_labels["visually_defined_centers"],
            "visible_colored_channels": raw_labels["visible_colored_channels"],
        },
        "derived_chart_data": derived_chart_data
        if derived_chart_data is not None
        else _derived_labels(),
        "validation_result": validation_result,
    }


def _metric_module():
    from human_design.vision import evaluation

    return evaluation


def _script_module():
    script_path = REPO_ROOT / "scripts/evaluate_bodygraph_extraction.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_bodygraph_extraction",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_golden_file(tmp_path: Path, cases: list[dict[str, Any]]) -> Path:
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "schema_version": "phase2_golden_labels_v2",
                "documentation": {},
                "recommended_sample_coverage": {},
                "cases": cases,
            }
        )
    )
    return golden_path


def _write_predictions_file(
    tmp_path: Path,
    predictions: list[dict[str, Any]],
) -> Path:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "schema_version": "phase2_predictions_v1",
                "predictions": predictions,
            }
        )
    )
    return predictions_path


def test_example_golden_labels_load_as_json() -> None:
    data = _load_golden_labels()

    assert set(data) == {
        "schema_version",
        "documentation",
        "recommended_sample_coverage",
        "cases",
    }
    assert isinstance(data["documentation"], dict)
    assert isinstance(data["recommended_sample_coverage"], dict)
    assert isinstance(data["cases"], list)
    assert data["cases"]


def test_every_case_has_required_top_level_keys() -> None:
    for case in _cases():
        assert set(case) == REQUIRED_CASE_KEYS
        assert isinstance(case["case_id"], str)
        assert isinstance(case["image_filename"], str)
        assert isinstance(case["label_source"], str)
        assert isinstance(case["notes"], str)
        assert isinstance(case["evaluation_scope"], dict)
        assert isinstance(case["expected_raw_labels"], dict)
        assert isinstance(case["expected_validation_result"], dict)


def test_golden_labels_reference_synthetic_fixture_without_loading_image() -> None:
    image_filenames = {case["image_filename"] for case in _cases()}

    assert image_filenames == {"test1.png"}
    assert all(filename.endswith(".png") for filename in image_filenames)


def test_full_derived_evaluation_cases_have_all_planetary_and_derived_labels() -> None:
    full_cases = [
        case
        for case in _cases()
        if case["evaluation_scope"]["include_derived_metrics"] is True
    ]
    assert full_cases

    for case in full_cases:
        raw_labels = case["expected_raw_labels"]
        assert set(raw_labels) == REQUIRED_RAW_LABEL_KEYS
        assert set(raw_labels["personality"]) == set(PLANETARY_FIELDS)
        assert set(raw_labels["design"]) == set(PLANETARY_FIELDS)

        activations = tuple(raw_labels["personality"].values()) + tuple(
            raw_labels["design"].values()
        )
        assert all(value is not None for value in activations)
        assert all(value not in DISALLOWED_MISSING_ACTIVATION_SENTINELS for value in activations)

        derived_labels = case["expected_derived_labels"]
        assert isinstance(derived_labels, dict)
        assert set(derived_labels) == REQUIRED_DERIVED_LABEL_KEYS
        assert set(derived_labels["defined_centers"]).issubset(CANONICAL_CENTERS)


def test_full_derived_golden_labels_are_consistent_with_interpreter() -> None:
    full_cases = [
        case
        for case in _cases()
        if case["evaluation_scope"]["include_derived_metrics"] is True
    ]
    assert full_cases

    for case in full_cases:
        parse_result = _parse_raw_labels(case["expected_raw_labels"])
        interpretation = interpret_bodygraph(parse_result.raw_vision)
        chart = interpretation.derived_chart_data
        expected = case["expected_derived_labels"]

        assert list(chart.active_gates) == expected["active_gates"]
        assert list(chart.active_channels) == expected["active_channels"]
        assert list(chart.defined_centers) == expected["defined_centers"]

        expected_basic_info = expected["basic_info"]
        for field_name in BASIC_INFO_FIELDS:
            assert getattr(chart.basic_info, field_name) == expected_basic_info[field_name]


def test_partial_raw_only_cases_exclude_derived_metrics_and_keep_all_planet_keys() -> None:
    partial_cases = [
        case
        for case in _cases()
        if case["evaluation_scope"]["include_derived_metrics"] is False
    ]
    assert partial_cases

    for case in partial_cases:
        assert case["evaluation_scope"]["include_derived_metrics"] is False
        assert case["expected_derived_labels"] is None

        raw_labels = case["expected_raw_labels"]
        assert set(raw_labels) == REQUIRED_RAW_LABEL_KEYS
        assert set(raw_labels["personality"]) == set(PLANETARY_FIELDS)
        assert set(raw_labels["design"]) == set(PLANETARY_FIELDS)

        activations = tuple(raw_labels["personality"].values()) + tuple(
            raw_labels["design"].values()
        )
        assert any(value is None for value in activations)
        assert all(value not in DISALLOWED_MISSING_ACTIVATION_SENTINELS for value in activations)


def test_recommended_sample_coverage_mentions_required_chart_types() -> None:
    coverage = _load_golden_labels()["recommended_sample_coverage"]
    coverage_text = json.dumps(coverage)

    for chart_type in (
        "Generator",
        "Manifesting Generator",
        "Projector",
        "Manifestor",
        "Reflector",
    ):
        assert chart_type in coverage_text

    assert "extra Reflector examples" in coverage_text


def test_gitignore_protects_private_bodygraph_artifacts() -> None:
    gitignore_lines = {
        line.strip()
        for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert {
        "data/bodygraph_samples/images/",
        "data/bodygraph_samples/private/",
        "data/bodygraph_samples/generated_responses/",
        "data/bodygraph_samples/vision_responses/",
    }.issubset(gitignore_lines)


def test_public_evaluation_api_and_activation_exact_match_metrics() -> None:
    evaluation = _metric_module()

    assert evaluation.activation_exact_match("61.4", "61.4") is True
    assert evaluation.activation_exact_match("61.4", "61.5") is False
    assert evaluation.activation_exact_match({"gate": 61, "line": 4}, "61.4") is True

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction=_prediction(),
    )

    assert result.metrics["personality_activation_exact_match_rate"] == 1.0
    assert result.metrics["design_activation_exact_match_rate"] == 1.0
    assert result.metrics["activation_exact_match_rate"] == 1.0


@pytest.mark.parametrize(
    ("expected_is_valid", "predicted_is_valid", "expected_metric"),
    [
        (True, True, 1.0),
        (False, False, 1.0),
        (False, True, 0.0),
    ],
)
def test_validation_is_valid_exact_match_metric(
    expected_is_valid: bool,
    predicted_is_valid: bool,
    expected_metric: float,
) -> None:
    evaluation = _metric_module()

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(expected_is_valid=expected_is_valid),
        prediction=_prediction(is_valid=predicted_is_valid),
    )

    assert result.metrics["validation_is_valid_exact_match"] == expected_metric


def test_fixture_validity_follows_explicit_true_metadata_with_override() -> None:
    # The fixture helpers mirror ValidationResult semantics: bare string
    # warning codes use canonical warning_defaults, literal dict
    # affects_validity=True invalidates, and explicit is_valid always wins
    # over warning metadata.
    assert _warnings_are_valid(None) is True
    assert _warnings_are_valid(["VISIBLE_CHANNEL_NOT_DERIVED"]) is True
    assert _warnings_are_valid(["MISSING_ACTIVATION"]) is False
    assert _warnings_are_valid([{"affects_validity": False}]) is True
    assert _warnings_are_valid([{"affects_validity": True}]) is False
    assert _warnings_are_valid([{"affects_validity": "true"}]) is True
    assert _warnings_are_valid([{"affects_validity": 1}]) is True

    nonfatal_warning = {
        "code": "VISIBLE_CHANNEL_NOT_DERIVED",
        "severity": ValidationSeverity.WARNING.value,
        "affects_validity": False,
        "source": ValidationSource.validation.value,
    }
    fatal_warning = {
        "code": "MISSING_ACTIVATION",
        "severity": ValidationSeverity.ERROR.value,
        "affects_validity": True,
        "source": ValidationSource.validation.value,
    }

    nonfatal_golden = _golden_case(warning_entries=[nonfatal_warning])
    fatal_prediction = _prediction(warnings=[fatal_warning])
    overridden_golden = _golden_case(
        expected_is_valid=False,
        warning_entries=[nonfatal_warning],
    )

    assert nonfatal_golden["expected_validation_result"]["is_valid"] is True
    assert fatal_prediction["validation_result"]["is_valid"] is False
    assert overridden_golden["expected_validation_result"]["is_valid"] is False


def test_missing_predicted_validation_is_valid_scores_zero() -> None:
    evaluation = _metric_module()
    prediction = _prediction(include_is_valid=False)

    validation_result = prediction["validation_result"]
    assert isinstance(validation_result, dict)
    assert "is_valid" not in validation_result

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(expected_is_valid=True),
        prediction=prediction,
    )

    assert result.metrics["validation_is_valid_exact_match"] == 0.0


def test_unlabeled_visual_collections_are_not_scored() -> None:
    # Scoping is label presence: a golden that does not label the visual
    # collections simply is not scored on them. There is no separate scope
    # flag inside evaluation any more.
    evaluation = _metric_module()
    golden = _golden_case()
    for field_name in (
        "visually_active_gates",
        "visually_defined_centers",
        "visible_colored_channels",
    ):
        del golden["expected_raw_labels"][field_name]

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=golden,
        prediction=_prediction(),
    )

    assert result.metrics["activation_exact_match_rate"] == 1.0
    assert not any(
        metric_name.startswith(
            ("visually_active_gate_", "visually_defined_center_", "visible_channel_")
        )
        for metric_name in result.metrics
    )


def test_raw_visual_metrics_run_when_labeled() -> None:
    evaluation = _metric_module()

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction=_prediction(),
    )

    assert result.metrics["visually_active_gate_f1"] == 1.0
    assert result.metrics["visually_defined_center_f1"] == 1.0
    assert result.metrics["visible_channel_f1"] == 1.0


@pytest.mark.parametrize(
    ("field_name", "metric_prefix"),
    [
        ("visually_active_gates", "visually_active_gate"),
        ("visually_defined_centers", "visually_defined_center"),
        ("visible_colored_channels", "visible_channel"),
    ],
)
def test_missing_predicted_visual_field_counts_as_empty_prediction(
    field_name: str,
    metric_prefix: str,
) -> None:
    evaluation = _metric_module()
    prediction = _prediction()
    raw_vision = prediction["raw_vision"]
    assert isinstance(raw_vision, dict)
    del raw_vision[field_name]

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction=prediction,
    )

    assert result.metrics[f"{metric_prefix}_precision"] == 0.0
    assert result.metrics[f"{metric_prefix}_recall"] == 0.0
    assert result.metrics[f"{metric_prefix}_f1"] == 0.0


@pytest.mark.parametrize(
    ("field_name", "malformed_value", "metric_prefix"),
    [
        ("visually_defined_centers", "G", "visually_defined_center"),
        ("visible_colored_channels", "3-60", "visible_channel"),
    ],
)
def test_malformed_raw_visual_collections_count_as_empty_predictions(
    field_name: str,
    malformed_value: object,
    metric_prefix: str,
) -> None:
    evaluation = _metric_module()
    prediction = _prediction()
    raw_vision = prediction["raw_vision"]
    assert isinstance(raw_vision, dict)
    raw_vision[field_name] = malformed_value

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction=prediction,
    )

    assert result.metrics[f"{metric_prefix}_precision"] == 0.0
    assert result.metrics[f"{metric_prefix}_recall"] == 0.0
    assert result.metrics[f"{metric_prefix}_f1"] == 0.0


@pytest.mark.parametrize("malformed_value", ["3-60", {"3-60": "ignored"}])
def test_malformed_derived_channel_collection_counts_as_empty_prediction(
    malformed_value: object,
) -> None:
    evaluation = _metric_module()
    prediction = _prediction()
    derived_chart_data = prediction["derived_chart_data"]
    assert isinstance(derived_chart_data, dict)
    derived_chart_data["active_channels"] = malformed_value

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction=prediction,
    )

    assert result.metrics["active_channel_precision"] == 0.0
    assert result.metrics["active_channel_recall"] == 0.0
    assert result.metrics["active_channel_f1"] == 0.0


def test_line_mismatch_and_missing_prediction_activation_count_as_failures() -> None:
    evaluation = _metric_module()
    personality = dict(_full_raw_labels()["personality"])
    personality["sun"] = "61.5"
    personality["moon"] = None

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction=_prediction(personality=personality),
    )

    assert result.metrics["personality_activation_exact_match_rate"] == 11 / 13
    assert result.metrics["design_activation_exact_match_rate"] == 1.0
    assert result.metrics["activation_exact_match_rate"] == 24 / 26


def test_hallucinated_activation_on_expected_null_lowers_null_respect_rate() -> None:
    # Golden says moon is unreadable (null); the prediction still emits a
    # value. Exact-match rates cannot see this, activation_null_respect_rate
    # is the metric that catches over-confident hallucination.
    evaluation = _metric_module()
    golden = _golden_case()
    golden["expected_raw_labels"]["personality"]["moon"] = None

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=golden,
        prediction=_prediction(),
    )

    assert result.metrics["activation_null_respect_rate"] == 0.0
    assert result.metrics["personality_activation_exact_match_rate"] == 1.0


def test_null_respect_rate_is_full_when_prediction_also_returns_null() -> None:
    evaluation = _metric_module()
    golden = _golden_case()
    golden["expected_raw_labels"]["personality"]["moon"] = None
    personality = dict(_full_raw_labels()["personality"])
    personality["moon"] = None

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=golden,
        prediction=_prediction(personality=personality),
    )

    assert result.metrics["activation_null_respect_rate"] == 1.0


def test_set_metrics_normalize_gate_types_and_channel_direction() -> None:
    # Label formatting must not silently zero a score: "34" == 34 and a
    # reversed "34-10" matches the canonical "10-34".
    evaluation = _metric_module()
    golden = _golden_case()
    golden["expected_derived_labels"]["active_gates"] = [
        str(gate) for gate in golden["expected_derived_labels"]["active_gates"]
    ]
    golden["expected_derived_labels"]["active_channels"] = ["3-60", "34-10"]

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=golden,
        prediction=_prediction(),
    )

    assert result.metrics["active_gate_f1"] == 1.0
    assert result.metrics["active_channel_f1"] == 1.0


def test_partial_raw_only_cases_keep_activation_metrics_and_skip_derived_metrics() -> None:
    evaluation = _metric_module()
    golden = _golden_case(include_derived_metrics=False)

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="partial_001",
        golden=golden,
        prediction=_prediction(derived_chart_data={}),
    )

    assert result.metrics["activation_exact_match_rate"] == 1.0
    assert "active_channel_f1" not in result.metrics
    assert "profile_exact_match" not in result.metrics


def test_missing_prediction_derived_values_count_as_failures_for_full_cases() -> None:
    evaluation = _metric_module()

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction=_prediction(derived_chart_data={}),
    )

    assert result.metrics["active_gate_precision"] == 0.0
    assert result.metrics["active_gate_recall"] == 0.0
    assert result.metrics["active_gate_f1"] == 0.0
    assert result.metrics["active_channel_f1"] == 0.0
    assert result.metrics["derived_center_f1"] == 0.0
    assert result.metrics["overall_basic_info_accuracy"] == 0.0


def test_evaluation_does_not_read_removed_prediction_aliases() -> None:
    evaluation = _metric_module()
    raw_labels = _full_raw_labels()

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction={
            "personality": raw_labels["personality"],
            "design": raw_labels["design"],
            "derived": _derived_labels(),
            "validation": {"is_valid": True, "warnings": []},
        },
    )

    assert result.metrics["activation_exact_match_rate"] == 0.0
    assert result.metrics["overall_basic_info_accuracy"] == 0.0
    assert result.metrics["validation_is_valid_exact_match"] == 0.0


def test_set_valued_and_basic_info_metrics_are_reported() -> None:
    evaluation = _metric_module()
    prediction = _prediction(
        derived_chart_data={
            "active_gates": [3, 10, 34, 60, 99],
            "active_channels": ["3-60"],
            "defined_centers": ["G", "Sacral"],
            "basic_info": {
                **_basic_info(),
                "authority": "Emotional",
            },
        }
    )

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(),
        prediction=prediction,
    )

    assert result.metrics["derived_center_precision"] == 1.0
    assert result.metrics["derived_center_recall"] == 2 / 3
    assert result.metrics["active_gate_precision"] == 4 / 5
    assert result.metrics["active_gate_recall"] == 4 / 8
    assert result.metrics["active_channel_precision"] == 1.0
    assert result.metrics["active_channel_recall"] == 1 / 2
    assert result.metrics["visually_defined_center_f1"] == 1.0
    assert result.metrics["visually_active_gate_f1"] == 1.0
    assert result.metrics["visible_channel_f1"] == 1.0
    assert result.metrics["authority_exact_match"] == 0.0
    assert result.metrics["profile_exact_match"] == 1.0
    assert result.metrics["overall_basic_info_accuracy"] == 6 / 7


def test_precision_recall_f1_empty_set_conventions() -> None:
    evaluation = _metric_module()

    assert evaluation.precision_recall_f1(set(), set()) == (1.0, 1.0, 1.0)
    assert evaluation.precision_recall_f1({"3-60"}, set()) == (0.0, 0.0, 0.0)
    assert evaluation.precision_recall_f1(set(), {"3-60"}) == (0.0, 1.0, 0.0)


def test_macro_aggregate_metrics_and_threshold_checks() -> None:
    evaluation = _metric_module()
    personality = dict(_full_raw_labels()["personality"])
    personality["sun"] = "61.5"

    summary = evaluation.evaluate_bodygraph_predictions(
        golden_cases=[
            _golden_case("case_001"),
            _golden_case("case_002"),
        ],
        predictions={
            "case_001": _prediction(),
            "case_002": _prediction(personality=personality),
        },
        thresholds={"activation_exact_match_rate": 0.99},
    )

    assert summary.aggregate_metrics["activation_exact_match_rate"] == (1.0 + 25 / 26) / 2
    assert summary.passed_thresholds is False
    assert evaluation.check_thresholds({"metric": 1.0}, {"metric": 0.9}) is True
    assert evaluation.check_thresholds({"metric": 0.5}, {"metric": 0.9}) is False
    assert not hasattr(summary.per_case[0], "passed_thresholds")


def test_missing_prediction_adds_case_warning_and_scores_zero() -> None:
    evaluation = _metric_module()

    summary = evaluation.evaluate_bodygraph_predictions(
        golden_cases=[_golden_case("case_001")],
        predictions={},
        thresholds={"activation_exact_match_rate": 0.5},
    )

    case_result = summary.per_case[0]
    assert "Prediction missing for this case." in case_result.warnings
    assert case_result.metrics["activation_exact_match_rate"] == 0.0
    assert summary.passed_thresholds is False


def test_warning_code_metrics_ignore_message_and_metadata() -> None:
    evaluation = _metric_module()
    expected_warning = {
        "code": "VISIBLE_CHANNEL_NOT_DERIVED",
        "severity": ValidationSeverity.WARNING.value,
        "affects_validity": False,
        "source": ValidationSource.validation.value,
        "message": "Ignored by evaluation.",
    }
    predicted_warning = {
        "code": "VISIBLE_CHANNEL_NOT_DERIVED",
        "severity": ValidationSeverity.WARNING.value,
        "affects_validity": False,
        "source": ValidationSource.validation.value,
        "message": "A different message is fine.",
    }

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(
            warning_entries=[expected_warning],
        ),
        prediction=_prediction(warnings=[predicted_warning]),
    )

    assert result.metrics["warning_code_precision"] == 1.0
    assert result.metrics["warning_code_recall"] == 1.0
    assert result.metrics["warning_code_f1"] == 1.0
    # Metadata matching was removed: severity / affects_validity are a
    # deterministic function of the code (models.warning_defaults), so
    # matching codes already implies matching metadata.
    assert "warning_metadata_exact_match_rate" not in result.metrics


def test_bare_string_warning_labels_match_full_warning_objects() -> None:
    # Golden labels may record warnings as bare code strings; a prediction
    # carrying the full warning object for the same code is a perfect match.
    evaluation = _metric_module()
    predicted_warning = {
        "code": "MISSING_ACTIVATION",
        "severity": ValidationSeverity.ERROR.value,
        "affects_validity": True,
        "source": ValidationSource.validation.value,
        "message": "Required activation is missing.",
        "field_path": "personality.moon",
    }

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(
            expected_is_valid=False,
            warning_entries=["MISSING_ACTIVATION"],
        ),
        prediction=_prediction(is_valid=False, warnings=[predicted_warning]),
    )

    assert result.metrics["warning_code_precision"] == 1.0
    assert result.metrics["warning_code_recall"] == 1.0
    assert result.metrics["warning_code_f1"] == 1.0
    assert result.metrics["validation_is_valid_exact_match"] == 1.0


def test_non_list_prediction_warnings_score_as_empty() -> None:
    # A warnings value that is not a list/tuple cannot carry warning entries;
    # it is treated as an empty prediction, so every expected code is missed.
    evaluation = _metric_module()
    expected_warning = {
        "code": "MISSING_ACTIVATION",
        "severity": ValidationSeverity.ERROR.value,
        "affects_validity": True,
        "source": ValidationSource.parser.value,
        "message": "Required activation is missing.",
    }
    prediction = _prediction()
    validation_result = prediction["validation_result"]
    assert isinstance(validation_result, dict)
    validation_result["warnings"] = "MISSING_ACTIVATION"

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(warning_entries=[expected_warning]),
        prediction=prediction,
    )

    assert result.metrics["warning_code_precision"] == 0.0
    assert result.metrics["warning_code_recall"] == 0.0
    assert result.metrics["warning_code_f1"] == 0.0


@pytest.mark.parametrize(
    "bad_entry",
    [
        "MISSING_ACTIVATON",  # typo of a known code
        "STALE_REMOVED_CODE",
        42,
        {"message": "warning without a code"},
    ],
)
def test_unknown_or_malformed_warning_entries_fail_loudly(
    bad_entry: object,
) -> None:
    # Warning codes are a closed contract: anything that is not a known code
    # string or a mapping with a known "code" must raise instead of silently
    # scoring zero (or, worse, silently passing when both sides are garbage).
    evaluation = _metric_module()
    expected_warning = {
        "code": "MISSING_ACTIVATION",
        "severity": ValidationSeverity.ERROR.value,
        "affects_validity": True,
        "source": ValidationSource.parser.value,
        "message": "Required activation is missing.",
    }
    prediction = _prediction()
    validation_result = prediction["validation_result"]
    assert isinstance(validation_result, dict)
    validation_result["warnings"] = [bad_entry]

    with pytest.raises(ValueError, match="warning"):
        evaluation.evaluate_bodygraph_prediction(
            case_id="case_001",
            golden=_golden_case(warning_entries=[expected_warning]),
            prediction=prediction,
        )


def test_script_main_reads_json_files_and_returns_nonzero_on_threshold_failure(
    tmp_path,
    capsys,
) -> None:
    script = _script_module()
    golden_path = _write_golden_file(tmp_path, [_golden_case("case_001")])
    predictions_path = _write_predictions_file(
        tmp_path,
        [{"case_id": "case_001", **_prediction()}],
    )

    passing_status = script.main(
        [
            "--golden",
            str(golden_path),
            "--predictions",
            str(predictions_path),
            "--threshold",
            "activation_exact_match_rate=1.0",
        ]
    )
    passing_output = capsys.readouterr().out

    assert passing_status == 0
    assert "Per-case metrics" in passing_output
    assert "Aggregate metrics" in passing_output

    failing_status = script.main(
        [
            "--golden",
            str(golden_path),
            "--predictions",
            str(predictions_path),
            "--threshold",
            "activation_exact_match_rate=1.1",
        ]
    )

    assert failing_status == 1


def test_script_json_keeps_threshold_state_only_at_summary_level(
    tmp_path,
    capsys,
) -> None:
    script = _script_module()
    golden_path = _write_golden_file(tmp_path, [_golden_case("case_001")])
    predictions_path = _write_predictions_file(
        tmp_path,
        [{"case_id": "case_001", **_prediction()}],
    )

    status = script.main(
        [
            "--golden",
            str(golden_path),
            "--predictions",
            str(predictions_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert "passed_thresholds" not in payload["per_case"][0]
    assert payload["passed_thresholds"] is True


def test_script_loaders_accept_only_canonical_versioned_file_shapes(tmp_path) -> None:
    script = _script_module()
    golden_path = _write_golden_file(tmp_path, [_golden_case("case_001")])
    predictions_path = _write_predictions_file(
        tmp_path,
        [{"case_id": "case_001", **_prediction()}],
    )

    assert script._load_golden_cases(golden_path)[0]["case_id"] == "case_001"
    assert set(script._load_predictions(predictions_path)) == {"case_001"}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"case_001": {}},
        {"cases": []},
        {"case_id": "case_001", "raw_vision": {}},
    ],
)
def test_prediction_loader_rejects_legacy_unwrapped_shapes(
    tmp_path,
    payload: object,
) -> None:
    script = _script_module()
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="predictions|schema_version"):
        script._load_predictions(path)


def test_prediction_loader_rejects_duplicate_case_ids(tmp_path) -> None:
    script = _script_module()
    prediction = {"case_id": "case_001", **_prediction()}
    path = _write_predictions_file(tmp_path, [prediction, prediction])

    with pytest.raises(ValueError, match="duplicate prediction case_id"):
        script._load_predictions(path)


def test_golden_loader_rejects_non_boolean_scope_flags(tmp_path) -> None:
    script = _script_module()
    golden_case = _golden_case()
    golden_case["evaluation_scope"]["include_derived_metrics"] = 1
    path = _write_golden_file(tmp_path, [golden_case])

    with pytest.raises(ValueError, match="include_derived_metrics must be a bool"):
        script._load_golden_cases(path)


@pytest.mark.parametrize(
    ("include_derived_metrics", "expected_derived_labels"),
    [
        (True, None),
        (False, _derived_labels()),
    ],
)
def test_golden_loader_requires_scope_and_derived_labels_to_agree(
    tmp_path,
    include_derived_metrics: bool,
    expected_derived_labels: object,
) -> None:
    script = _script_module()
    golden_case = _golden_case(
        include_derived_metrics=include_derived_metrics,
    )
    golden_case["expected_derived_labels"] = expected_derived_labels
    path = _write_golden_file(tmp_path, [golden_case])

    with pytest.raises(
        ValueError,
        match="include_derived_metrics.*expected_derived_labels|"
        "expected_derived_labels.*include_derived_metrics",
    ):
        script._load_golden_cases(path)


def test_golden_loader_accepts_warning_entries_with_field_path(tmp_path) -> None:
    # Golden warnings are naturally copied from pipeline --json output,
    # whose warnings carry field_path; the loader must accept them.
    script = _script_module()
    warning = {
        "code": "MISSING_ACTIVATION",
        "message": "MISSING_ACTIVATION at personality.moon",
        "severity": ValidationSeverity.ERROR.value,
        "affects_validity": True,
        "source": ValidationSource.validation.value,
        "field_path": "personality.moon",
    }
    golden_case = _golden_case(
        expected_is_valid=False,
        warning_entries=[warning],
    )
    path = _write_golden_file(tmp_path, [golden_case])

    assert script._load_golden_cases(path)[0]["case_id"] == "case_001"


def test_golden_loader_accepts_bare_string_warning_codes(tmp_path) -> None:
    # Hand-written golden labels may record just the expected code strings.
    script = _script_module()
    golden_case = _golden_case(
        expected_is_valid=False,
        warning_entries=["MISSING_ACTIVATION", "VISIBLE_CHANNEL_NOT_DERIVED"],
    )
    path = _write_golden_file(tmp_path, [golden_case])

    assert script._load_golden_cases(path)[0]["case_id"] == "case_001"


def test_golden_loader_rejects_unknown_bare_string_warning_code(tmp_path) -> None:
    script = _script_module()
    golden_case = _golden_case(
        expected_is_valid=False,
        warning_entries=["MISSING_ACTIVATON"],
    )
    path = _write_golden_file(tmp_path, [golden_case])

    with pytest.raises(ValueError, match="warning code"):
        script._load_golden_cases(path)


def test_script_exits_with_input_error_code_on_unknown_warning_codes(
    tmp_path,
    capsys,
) -> None:
    # Unknown codes are input errors (exit 2), whether they appear in the
    # golden file (caught at load) or the predictions (caught at evaluate).
    script = _script_module()
    good_golden_path = _write_golden_file(
        tmp_path,
        [
            _golden_case(
                "case_001",
                expected_is_valid=False,
                warning_entries=["MISSING_ACTIVATION"],
            )
        ],
    )
    bad_golden_path = tmp_path / "bad_golden.json"
    bad_golden_case = _golden_case(
        "case_001",
        expected_is_valid=False,
        warning_entries=["MISSING_ACTIVATON"],
    )
    bad_golden_path.write_text(
        json.dumps(
            {
                "schema_version": "phase2_golden_labels_v2",
                "documentation": {},
                "recommended_sample_coverage": {},
                "cases": [bad_golden_case],
            }
        )
    )
    stale_prediction = _prediction(is_valid=False)
    stale_prediction["validation_result"]["warnings"] = [
        {
            "code": "STALE_REMOVED_CODE",
            "severity": ValidationSeverity.WARNING.value,
            "affects_validity": False,
            "source": ValidationSource.validation.value,
            "message": "This code no longer exists.",
        }
    ]
    predictions_path = _write_predictions_file(
        tmp_path,
        [{"case_id": "case_001", **stale_prediction}],
    )

    bad_golden_status = script.main(
        [
            "--golden",
            str(bad_golden_path),
            "--predictions",
            str(predictions_path),
        ]
    )
    bad_prediction_status = script.main(
        [
            "--golden",
            str(good_golden_path),
            "--predictions",
            str(predictions_path),
        ]
    )
    stderr_output = capsys.readouterr().err

    assert bad_golden_status == 2
    assert bad_prediction_status == 2
    assert "warning code" in stderr_output


def test_golden_loader_rejects_duplicate_case_ids(tmp_path) -> None:
    script = _script_module()
    golden_case = _golden_case("case_001")
    path = _write_golden_file(tmp_path, [golden_case, golden_case])

    with pytest.raises(ValueError, match="duplicate golden case_id"):
        script._load_golden_cases(path)