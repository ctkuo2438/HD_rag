import importlib.util
import json
import subprocess
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
NONCANONICAL_WARNING_METADATA = (
    {"severity": "INFO"},
    {"affects_validity": False},
    {"severity": "INFO", "affects_validity": False},
    {"affects_validity": 1},
    {"affects_validity": "true"},
    {"severity": None},
    {"affects_validity": None},
)
PRIVATE_WARNING_MARKERS = (
    "SYNTHETIC_PRIVATE_CHART_CONTENT",
    "/synthetic/private/chart.png",
    "data:image/png;base64,U1lOVEhFVElDLU9OTFk=",
    "sk-synthetic-credential-do-not-print",
)
RAW_METRIC_NAMES = {
    "personality_activation_exact_match_rate",
    "design_activation_exact_match_rate",
    "activation_exact_match_rate",
} | {
    f"{prefix}_{metric}"
    for prefix in ("visually_defined_center", "visually_active_gate", "visible_channel")
    for metric in ("precision", "recall", "f1")
}
DERIVED_METRIC_NAMES = {
    f"{field}_exact_match" for field in BASIC_INFO_FIELDS
} | {"overall_basic_info_accuracy"} | {
    f"{prefix}_{metric}"
    for prefix in ("derived_center", "active_gate", "active_channel")
    for metric in ("precision", "recall", "f1")
}
VALIDATION_METRIC_NAMES = {
    "validation_is_valid_exact_match",
    "warning_code_precision",
    "warning_code_recall",
    "warning_code_f1",
}
PREDICTION_COLLECTIONS = (
    ("raw_vision", "expected_raw_labels", "visually_active_gates", "visually_active_gate"),
    ("raw_vision", "expected_raw_labels", "visually_defined_centers", "visually_defined_center"),
    ("raw_vision", "expected_raw_labels", "visible_colored_channels", "visible_channel"),
    ("derived_chart_data", "expected_derived_labels", "active_gates", "active_gate"),
    ("derived_chart_data", "expected_derived_labels", "active_channels", "active_channel"),
    ("derived_chart_data", "expected_derived_labels", "defined_centers", "derived_center"),
    ("validation_result", "expected_validation_result", "warnings", "warning_code"),
)


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


def _warning_payload(
    code: ValidationCode = ValidationCode.MISSING_ACTIVATION,
) -> dict[str, object]:
    severity, affects_validity = warning_defaults(code)
    return {
        "code": code.value,
        "severity": severity.value,
        "affects_validity": affects_validity,
        "source": ValidationSource.validation.value,
        "message": "Synthetic warning.",
        "field_path": "personality.moon",
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


@pytest.fixture(params=["loader", "evaluator"])
def check_scope(request, tmp_path):
    def check(golden):
        if request.param == "loader":
            _script_module()._load_golden_cases(_write_golden_file(tmp_path, [golden]))
        else:
            _metric_module().evaluate_bodygraph_prediction(
                case_id="case_001", golden=golden, prediction=_prediction(),
            )

    return check


@pytest.fixture(params=["loader", "evaluator", "batch_evaluator"])
def check_prediction(request, tmp_path):
    def check(prediction, *, include_metrics=True):
        if request.param == "loader":
            path = _write_predictions_file(tmp_path, [{"case_id": "case_001", **prediction}])
            return _script_module()._load_predictions(path)
        golden = _golden_case(
            include_raw_visual_metrics=include_metrics,
            include_derived_metrics=include_metrics,
        )
        if request.param == "batch_evaluator":
            return _metric_module().evaluate_bodygraph_predictions(
                golden_cases=[golden], predictions={"case_001": prediction},
            )
        return _metric_module().evaluate_bodygraph_prediction(
            case_id="case_001", golden=golden, prediction=prediction,
        )

    return check


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


@pytest.mark.parametrize("directory", ["", "nested/artifacts/"])
@pytest.mark.parametrize(
    "suffix",
    ["vision_response.json", "bodygraph_prediction.json", "base64", "image_payload.json"],
)
def test_gitignore_protects_generated_artifact_suffixes(
    directory: str, suffix: str,
) -> None:
    artifact_path = f"{directory}sample.{suffix}"
    result = subprocess.run(
        [
            "git", "-c", "core.excludesFile=/dev/null", "check-ignore",
            "--no-index", "--verbose", "--", artifact_path,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(".gitignore:")
    assert result.stdout.rstrip().endswith(f":*.{suffix}\t{artifact_path}")


def test_gitignore_preserves_sanitized_fixtures() -> None:
    result = subprocess.run(
        [
            "git", "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "--",
            "tests/fixtures/bodygraph/test1.png",
            "tests/fixtures/bodygraph/test1_raw_response.json",
            "data/bodygraph_samples/golden_labels.example.json",
            "data/pdfs/.gitkeep",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert result.stdout == ""


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


@pytest.mark.parametrize(
    "field_name",
    ["visually_active_gates", "visually_defined_centers", "visible_colored_channels"],
)
def test_enabled_raw_scope_requires_visual_collections(field_name: str) -> None:
    golden = _golden_case()
    del golden["expected_raw_labels"][field_name]

    with pytest.raises(ValueError, match="expected_raw_labels"):
        _metric_module().evaluate_bodygraph_prediction(
            case_id="case_001", golden=golden, prediction=_prediction(),
        )


@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize("include_derived", [False, True])
@pytest.mark.parametrize("missing_prediction", [False, True])
def test_scope_gates_metric_families(
    tmp_path, include_raw, include_derived, missing_prediction,
) -> None:
    golden = _golden_case(
        include_raw_visual_metrics=include_raw,
        include_derived_metrics=include_derived,
    )
    loaded = _script_module()._load_golden_cases(_write_golden_file(tmp_path, [golden]))
    result = _metric_module().evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=loaded[0],
        prediction={} if missing_prediction else _prediction(),
    )
    expected_names = set(VALIDATION_METRIC_NAMES)
    if include_raw:
        expected_names.update(RAW_METRIC_NAMES)
    if include_derived:
        expected_names.update(DERIVED_METRIC_NAMES)

    assert set(result.metrics) == expected_names
    assert ("Prediction is missing derived_chart_data." in result.warnings) is (
        include_derived and missing_prediction
    )
    if not missing_prediction:
        assert all(value == 1.0 for value in result.metrics.values())


@pytest.mark.parametrize("include_raw", [False, True])
def test_raw_scope_also_gates_null_respect(include_raw: bool) -> None:
    golden = _golden_case(
        include_raw_visual_metrics=include_raw, include_derived_metrics=False,
    )
    golden["expected_raw_labels"]["personality"]["moon"] = None
    result = _metric_module().evaluate_bodygraph_prediction(
        case_id="case_001", golden=golden, prediction=_prediction(),
    )

    if include_raw:
        assert result.metrics["activation_null_respect_rate"] == 0.0
        assert set(result.metrics) == (
            RAW_METRIC_NAMES | VALIDATION_METRIC_NAMES | {"activation_null_respect_rate"}
        )
    else:
        assert set(result.metrics) == VALIDATION_METRIC_NAMES


def test_scope_excludes_disabled_cases_from_aggregate_metrics() -> None:
    raw_only = _golden_case("raw", include_derived_metrics=False)
    derived_only = _golden_case("derived", include_raw_visual_metrics=False)
    combined = _golden_case("combined")
    validation_only = _golden_case(
        "validation", include_raw_visual_metrics=False, include_derived_metrics=False,
    )
    derived_prediction = _prediction()
    derived_prediction["raw_vision"] = {}
    validation_prediction = _prediction(derived_chart_data={})
    validation_prediction["raw_vision"] = {}

    summary = _metric_module().evaluate_bodygraph_predictions(
        golden_cases=[raw_only, derived_only, combined, validation_only],
        predictions={
            "raw": _prediction(derived_chart_data={}),
            "derived": derived_prediction,
            "combined": _prediction(),
            "validation": validation_prediction,
        },
        thresholds={"activation_exact_match_rate": 1.0, "overall_basic_info_accuracy": 1.0},
    )

    assert set(summary.aggregate_metrics) == (
        RAW_METRIC_NAMES | DERIVED_METRIC_NAMES | VALIDATION_METRIC_NAMES
    )
    assert all(value == 1.0 for value in summary.aggregate_metrics.values())
    assert summary.passed_thresholds is True


def test_mixed_scope_aggregation_does_not_inflate_nonperfect_scores() -> None:
    summary = _metric_module().evaluate_bodygraph_predictions(
        golden_cases=[
            _golden_case("combined"),
            _golden_case("raw", include_derived_metrics=False),
            _golden_case("derived", include_raw_visual_metrics=False),
            _golden_case(
                "validation", include_raw_visual_metrics=False, include_derived_metrics=False,
            ),
        ],
        predictions={
            "combined": _prediction(),
            "raw": _prediction(personality={}, design={}),
            "derived": _prediction(derived_chart_data={}),
            "validation": _prediction(),
        },
    )
    per_case = {result.case_id: result.metrics for result in summary.per_case}

    # Disabled families have perfect predictions, but only eligible 0/1 scores count.
    for metric, zero_case, excluded_cases in (
        ("activation_exact_match_rate", "raw", ("derived", "validation")),
        ("overall_basic_info_accuracy", "derived", ("raw", "validation")),
    ):
        assert per_case["combined"][metric] == 1.0
        assert per_case[zero_case][metric] == 0.0
        assert all(metric not in per_case[case_id] for case_id in excluded_cases)
        assert summary.aggregate_metrics[metric] == 0.5


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
def test_missing_predicted_visual_field_counts_as_failure(
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
    ("field_name", "malformed_value"),
    [
        ("visually_defined_centers", "G"),
        ("visible_colored_channels", "3-60"),
    ],
)
def test_malformed_raw_visual_collections_raise_input_errors(
    field_name: str,
    malformed_value: object,
) -> None:
    evaluation = _metric_module()
    prediction = _prediction()
    raw_vision = prediction["raw_vision"]
    assert isinstance(raw_vision, dict)
    raw_vision[field_name] = malformed_value

    with pytest.raises(ValueError, match=field_name):
        evaluation.evaluate_bodygraph_prediction(
            case_id="case_001", golden=_golden_case(), prediction=prediction,
        )


@pytest.mark.parametrize("malformed_value", ["3-60", {"3-60": "ignored"}])
def test_malformed_derived_channel_collection_raises_input_error(
    malformed_value: object,
) -> None:
    evaluation = _metric_module()
    prediction = _prediction()
    derived_chart_data = prediction["derived_chart_data"]
    assert isinstance(derived_chart_data, dict)
    derived_chart_data["active_channels"] = malformed_value

    with pytest.raises(ValueError, match="active_channels"):
        evaluation.evaluate_bodygraph_prediction(
            case_id="case_001", golden=_golden_case(), prediction=prediction,
        )


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
    golden = _golden_case(include_derived_metrics=False)
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
    golden = _golden_case(include_derived_metrics=False)
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
            "active_gates": [3, 10, 34, 60, 64],
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


@pytest.mark.parametrize("column", ["personality", "design"])
@pytest.mark.parametrize("missing_level", ["field", "column", "section"])
def test_missing_activation_never_matches_expected_null(column, missing_level) -> None:
    golden = _golden_case(include_derived_metrics=False)
    golden["expected_raw_labels"][column]["moon"] = None
    prediction = _prediction()
    prediction["raw_vision"][column]["moon"] = None
    if missing_level == "field":
        del prediction["raw_vision"][column]["moon"]
    elif missing_level == "column":
        del prediction["raw_vision"][column]
    else:
        del prediction["raw_vision"]

    result = _metric_module().evaluate_bodygraph_prediction(
        case_id="case_001", golden=golden, prediction=prediction,
    )

    assert result.metrics["activation_null_respect_rate"] == 0.0


@pytest.mark.parametrize("section,labels,field,prefix", PREDICTION_COLLECTIONS)
@pytest.mark.parametrize("state", ["missing", "empty", "matching"])
@pytest.mark.parametrize("expected_empty", [False, True])
def test_prediction_collection_presence_controls_credit(
    section, labels, field, prefix, state, expected_empty,
) -> None:
    golden = _golden_case(warning_entries=["MISSING_ACTIVATION"])
    if expected_empty:
        golden[labels][field] = []
    prediction = _prediction()
    if state == "missing":
        del prediction[section][field]
    else:
        prediction[section][field] = [] if state == "empty" else golden[labels][field]

    result = _metric_module().evaluate_bodygraph_prediction(
        case_id="case_001", golden=golden, prediction=prediction,
    )

    expected_score = float(state == "matching" or (state == "empty" and expected_empty))
    for metric in ("precision", "recall", "f1"):
        assert result.metrics[f"{prefix}_{metric}"] == expected_score


@pytest.mark.parametrize("section,labels,field,prefix", PREDICTION_COLLECTIONS)
@pytest.mark.parametrize("value", [None, "", {}, 1, False])
def test_malformed_prediction_collection_containers_are_rejected(
    check_prediction, section, labels, field, prefix, value,
) -> None:
    prediction = _prediction()
    prediction[section][field] = value

    with pytest.raises(ValueError, match=rf"{section}\.{field}"):
        check_prediction(prediction, include_metrics=False)


@pytest.mark.parametrize("column", ["personality", "design"])
@pytest.mark.parametrize(
    "value",
    [
        "", "bad", "61.x", "0.1", "65.1", "1.0", "1.7", 61.4, True, False, [], {},
        {"gate": 61}, {"line": 4}, {"gate": True, "line": 4},
        {"gate": 61, "line": False}, {"gate": 61.0, "line": 4},
        {"gate": "61", "line": 4}, {"gate": 0, "line": 1}, {"gate": 64, "line": 7},
    ],
)
def test_malformed_prediction_activations_are_rejected(check_prediction, column, value) -> None:
    prediction = _prediction()
    prediction["raw_vision"][column]["sun"] = value

    with pytest.raises(ValueError, match=rf"raw_vision\.{column}\.sun"):
        check_prediction(prediction, include_metrics=False)


@pytest.mark.parametrize(
    "path,value",
    [
        (f"{section}.{field}", value)
        for section, field in (
            ("raw_vision", "visually_active_gates"),
            ("derived_chart_data", "active_gates"),
        )
        for value in (True, False, 1.0, 0, 65, "65", "one", None, {}, [])
    ] + [
        (f"{section}.{field}", value)
        for section, field in (
            ("raw_vision", "visually_defined_centers"),
            ("derived_chart_data", "defined_centers"),
            ("raw_vision", "visible_colored_channels"),
            ("derived_chart_data", "active_channels"),
        )
        for value in (True, 1, None, {}, [], "invalid")
    ],
)
def test_malformed_prediction_set_elements_are_rejected(check_prediction, path, value) -> None:
    section, field = path.split(".")
    prediction = _prediction()
    prediction[section][field] = [value]

    with pytest.raises(ValueError, match=rf"{section}\.{field}\[0\]"):
        check_prediction(prediction, include_metrics=False)


@pytest.mark.parametrize(
    "path",
    [
        "raw_vision", "raw_vision.personality", "raw_vision.design",
        "derived_chart_data", "derived_chart_data.basic_info", "validation_result",
    ],
)
@pytest.mark.parametrize("value", [None, 0, 1, True, False, "false", []])
def test_malformed_prediction_objects_are_rejected(check_prediction, path, value) -> None:
    prediction = _prediction()
    parts = path.split(".")
    parent = prediction
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        check_prediction(prediction, include_metrics=False)


@pytest.mark.parametrize("field", BASIC_INFO_FIELDS)
@pytest.mark.parametrize("value", [None, 0, 1, True, False, [], {}])
def test_malformed_basic_info_scalars_are_rejected(check_prediction, field, value) -> None:
    prediction = _prediction()
    prediction["derived_chart_data"]["basic_info"][field] = value

    with pytest.raises(ValueError, match=rf"derived_chart_data\.basic_info\.{field}"):
        check_prediction(prediction, include_metrics=False)


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_predicted_validity_requires_a_real_boolean(check_prediction, value) -> None:
    prediction = _prediction()
    prediction["validation_result"]["is_valid"] = value

    with pytest.raises(ValueError, match=r"validation_result\.is_valid"):
        check_prediction(prediction, include_metrics=False)


@pytest.mark.parametrize(
    "entry",
    [
        {}, {"case_id": None}, {"case_id": False}, {"case_id": 1}, {"case_id": ""},
        {"case_id": "case_001", "derived": {}},
        {"case_id": "case_001", "validation": {}},
        {"case_id": "case_001", "extra": {}},
    ],
)
def test_partial_prediction_entries_still_require_identity_and_supported_keys(tmp_path, entry):
    path = _write_predictions_file(tmp_path, [entry])

    with pytest.raises(ValueError, match="case_id|unsupported"):
        _script_module()._load_predictions(path)


@pytest.mark.parametrize("value", [None, "1.1", "64.6", " 61.4 ", {"gate": 61, "line": 4}])
def test_valid_prediction_activations_are_accepted(check_prediction, value) -> None:
    prediction = _prediction()
    prediction["raw_vision"]["personality"]["sun"] = value

    check_prediction(prediction)


@pytest.mark.parametrize("value", [None, False, [], "missing"])
def test_malformed_entire_prediction_is_not_missing(value) -> None:
    evaluation = _metric_module()
    with pytest.raises(ValueError, match="prediction"):
        evaluation.evaluate_bodygraph_prediction(
            case_id="case_001", golden=_golden_case(), prediction=value,
        )
    with pytest.raises(ValueError, match="prediction"):
        evaluation.evaluate_bodygraph_predictions(
            golden_cases=[_golden_case()], predictions={"case_001": value},
        )


@pytest.mark.parametrize("predictions", [None, [], False, "case_001"])
def test_batch_predictions_require_a_mapping(predictions) -> None:
    with pytest.raises(ValueError, match="predictions"):
        _metric_module().evaluate_bodygraph_predictions(
            golden_cases=[_golden_case()], predictions=predictions,
        )


@pytest.mark.parametrize("value", [None, [], {"raw_vision": None}])
def test_batch_rejects_malformed_entries_without_matching_golden_cases(value) -> None:
    with pytest.raises(ValueError, match="prediction"):
        _metric_module().evaluate_bodygraph_predictions(
            golden_cases=[_golden_case()],
            predictions={"case_001": _prediction(), "unmatched": value},
        )


@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize("include_derived", [False, True])
@pytest.mark.parametrize("present", [False, True])
def test_missing_prediction_scores_all_applicable_metrics_as_failures(
    tmp_path, include_raw, include_derived, present,
) -> None:
    golden = _golden_case(
        include_raw_visual_metrics=include_raw, include_derived_metrics=include_derived,
    )
    for _, labels, field, _ in PREDICTION_COLLECTIONS:
        if golden[labels] is not None:
            golden[labels][field] = []
    if not include_derived:
        golden["expected_raw_labels"]["personality"]["moon"] = None
    path = _write_predictions_file(tmp_path, [{"case_id": "case_001"}] if present else [])
    predictions = _script_module()._load_predictions(path)

    summary = _metric_module().evaluate_bodygraph_predictions(
        golden_cases=[golden], predictions=predictions,
    )

    expected_names = VALIDATION_METRIC_NAMES.copy()
    if include_raw:
        expected_names |= RAW_METRIC_NAMES
        if not include_derived:
            expected_names.add("activation_null_respect_rate")
    if include_derived:
        expected_names |= DERIVED_METRIC_NAMES
    assert summary.per_case[0].metrics == dict.fromkeys(expected_names, 0.0)
    assert summary.aggregate_metrics == dict.fromkeys(expected_names, 0.0)
    assert ("Prediction missing for this case." in summary.per_case[0].warnings) is not present


@pytest.mark.parametrize("section", ["raw_vision", "derived_chart_data", "validation_result"])
def test_loader_preserves_omitted_prediction_sections(tmp_path, section) -> None:
    prediction = {"case_id": "case_001", **_prediction()}
    del prediction[section]
    path = _write_predictions_file(tmp_path, [prediction])

    loaded = _script_module()._load_predictions(path)["case_001"]

    assert loaded == prediction
    assert section not in loaded


@pytest.mark.parametrize("field", BASIC_INFO_FIELDS)
def test_missing_basic_info_field_stays_in_accuracy_denominator(field) -> None:
    prediction = _prediction()
    del prediction["derived_chart_data"]["basic_info"][field]

    result = _metric_module().evaluate_bodygraph_prediction(
        case_id="case_001", golden=_golden_case(), prediction=prediction,
    )

    assert result.metrics[f"{field}_exact_match"] == 0.0
    assert result.metrics["overall_basic_info_accuracy"] == 6 / 7


def test_missing_outputs_stay_in_mixed_scope_aggregation_denominators() -> None:
    cases = []
    for case_id in ("perfect", "missing", "disabled"):
        golden = _golden_case(
            case_id, include_raw_visual_metrics=case_id != "disabled",
            include_derived_metrics=case_id != "disabled",
        )
        golden["expected_raw_labels"]["visually_active_gates"] = []
        if case_id != "disabled":
            golden["expected_derived_labels"]["active_channels"] = []
        cases.append(golden)
    prediction = _prediction()
    prediction["raw_vision"]["visually_active_gates"] = []
    prediction["derived_chart_data"]["active_channels"] = []

    summary = _metric_module().evaluate_bodygraph_predictions(
        golden_cases=cases, predictions={"perfect": prediction, "disabled": _prediction()},
        thresholds={"visually_active_gate_f1": 0.75, "active_channel_f1": 0.75},
    )

    for metric in ("visually_active_gate_f1", "active_channel_f1"):
        assert [case.metrics[metric] for case in summary.per_case[:2]] == [1.0, 0.0]
        assert metric not in summary.per_case[2].metrics
        assert summary.aggregate_metrics[metric] == 0.5
    assert summary.aggregate_metrics["warning_code_f1"] == 2 / 3
    assert summary.passed_thresholds is False


def test_missing_null_activations_stay_in_aggregation_denominator() -> None:
    cases = [_golden_case(name, include_derived_metrics=False) for name in ("good", "missing")]
    for golden in cases:
        golden["expected_raw_labels"]["personality"]["moon"] = None
    prediction = _prediction()
    prediction["raw_vision"]["personality"]["moon"] = None

    summary = _metric_module().evaluate_bodygraph_predictions(
        golden_cases=cases, predictions={"good": prediction},
    )

    assert summary.aggregate_metrics["activation_null_respect_rate"] == 0.5


@pytest.mark.parametrize("collection_type", [list, tuple, set, frozenset])
def test_valid_prediction_representations_keep_existing_scores(collection_type) -> None:
    golden = _golden_case(warning_entries=["MISSING_ACTIVATION"])
    prediction = _prediction()
    prediction["raw_vision"]["personality"]["sun"] = {"gate": 61, "line": 4}
    prediction["raw_vision"]["design"]["sun"] = " 32.6 "
    for section, _, field, _ in PREDICTION_COLLECTIONS[:-1]:
        values = prediction[section][field]
        if "gates" in field:
            values = [f" {value} " for value in values]
        elif "channels" in field:
            values = ["60-3", "34-10"]
        else:
            values = [f" {value} " for value in values]
        prediction[section][field] = collection_type(values)
    prediction["validation_result"] = {
        "is_valid": False,
        "warnings": ("MISSING_ACTIVATION", _warning_payload()),
    }

    result = _metric_module().evaluate_bodygraph_prediction(
        case_id="case_001", golden=golden, prediction=prediction,
    )

    assert set(result.metrics.values()) == {1.0}


@pytest.mark.parametrize("state,expected_status", [("missing", 1), ("malformed", 2)])
def test_cli_distinguishes_missing_output_from_malformed_input(
    tmp_path, capsys, state, expected_status,
) -> None:
    golden_path = _write_golden_file(tmp_path, [_golden_case()])
    prediction = {"case_id": "case_001"}
    if state == "malformed":
        prediction["validation_result"] = {"warnings": "synthetic private payload"}
    predictions_path = _write_predictions_file(tmp_path, [prediction])

    status = _script_module().main([
        "--golden", str(golden_path), "--predictions", str(predictions_path),
        "--threshold", "warning_code_f1=1.0", "--json",
    ])

    output = capsys.readouterr()
    assert status == expected_status
    if state == "missing":
        assert json.loads(output.out)["aggregate_metrics"]["warning_code_f1"] == 0.0
        assert output.err == ""
    else:
        assert output.out == ""
        assert "Invalid evaluation input:" in output.err
        assert "validation_result.warnings" in output.err
        assert "synthetic private payload" not in output.err


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


@pytest.mark.parametrize("code", list(ValidationCode))
def test_bare_string_warning_labels_match_full_warning_objects(
    code: ValidationCode,
) -> None:
    # Golden labels may record warnings as bare code strings; a prediction
    # carrying the full warning object for the same code is a perfect match.
    evaluation = _metric_module()
    predicted_warning = _warning_payload(code)
    _, affects_validity = warning_defaults(code)

    result = evaluation.evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(
            expected_is_valid=not affects_validity,
            warning_entries=[code.value],
        ),
        prediction=_prediction(
            is_valid=not affects_validity, warnings=[predicted_warning],
        ),
    )

    assert result.metrics["warning_code_precision"] == 1.0
    assert result.metrics["warning_code_recall"] == 1.0
    assert result.metrics["warning_code_f1"] == 1.0
    assert result.metrics["validation_is_valid_exact_match"] == 1.0


@pytest.mark.parametrize("side", ["golden", "prediction"])
@pytest.mark.parametrize("metadata", NONCANONICAL_WARNING_METADATA)
def test_evaluation_rejects_noncanonical_warning_metadata(side, metadata) -> None:
    canonical = _warning_payload()
    conflicting = {**canonical, **metadata}

    with pytest.raises(ValueError, match="canonical"):
        _metric_module().evaluate_bodygraph_prediction(
            case_id="case_001",
            golden=_golden_case(
                warning_entries=[conflicting if side == "golden" else canonical],
            ),
            prediction=_prediction(
                warnings=[conflicting if side == "prediction" else canonical],
            ),
        )


@pytest.mark.parametrize("side", ["golden", "prediction"])
@pytest.mark.parametrize("metadata", NONCANONICAL_WARNING_METADATA)
def test_evaluation_loaders_reject_noncanonical_warning_metadata(
    tmp_path, capsys, side, metadata,
) -> None:
    script = _script_module()
    canonical = _warning_payload()
    conflicting = {**canonical, **metadata}
    golden_path = _write_golden_file(
        tmp_path,
        [_golden_case(
            warning_entries=[conflicting if side == "golden" else canonical],
        )],
    )
    predictions_path = _write_predictions_file(
        tmp_path,
        [{"case_id": "case_001", **_prediction(
            warnings=[conflicting if side == "prediction" else canonical],
        )}],
    )

    with pytest.raises(ValueError, match="canonical"):
        if side == "golden":
            script._load_golden_cases(golden_path)
        else:
            script._load_predictions(predictions_path)

    assert script.main([
        "--golden", str(golden_path), "--predictions", str(predictions_path),
    ]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "canonical" in output.err


@pytest.mark.parametrize("marker", PRIVATE_WARNING_MARKERS)
@pytest.mark.parametrize("as_object", [False, True])
def test_unknown_warning_code_errors_do_not_expose_input(marker, as_object) -> None:
    warning = {**_warning_payload(), "code": marker} if as_object else marker

    with pytest.raises(ValueError) as exc_info:
        _metric_module().warning_codes(["VISIBLE_CHANNEL_NORMALIZED", warning])

    assert str(exc_info.value) == "warnings[1] has unknown warning code"
    assert marker not in str(exc_info.value)


@pytest.mark.parametrize("marker", PRIVATE_WARNING_MARKERS)
@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize("side", ["golden", "prediction", "unmatched_prediction"])
@pytest.mark.parametrize("json_mode", [False, True])
def test_unknown_warning_codes_are_redacted_in_cli_errors(
    tmp_path, capsys, marker, as_object, side, json_mode,
) -> None:
    warning = {**_warning_payload(), "code": marker} if as_object else marker
    golden = _golden_case()
    predictions = [{"case_id": "case_001", **_prediction()}]
    if side == "golden":
        golden["expected_validation_result"]["warnings"] = [warning]
    elif side == "prediction":
        predictions[0]["validation_result"]["warnings"] = [warning]
    else:
        predictions.append({
            "case_id": "unmatched", "validation_result": {"warnings": [warning]},
        })
    golden_path = _write_golden_file(tmp_path, [golden])
    predictions_path = _write_predictions_file(tmp_path, predictions)

    status = _script_module().main([
        "--golden", str(golden_path), "--predictions", str(predictions_path),
        *(["--json"] if json_mode else []),
    ])

    output = capsys.readouterr()
    assert status == 2
    assert output.out == ""
    assert output.err == "Invalid evaluation input: warnings[0] has unknown warning code\n"
    assert marker not in output.err


@pytest.mark.parametrize("marker", PRIVATE_WARNING_MARKERS)
def test_unexpected_field_names_are_redacted_in_shared_schema_errors(marker) -> None:
    value = {"schema_version": "phase2_predictions_v1", "predictions": [], marker: "private value"}

    with pytest.raises(ValueError) as exc_info:
        _script_module()._require_exact_keys(
            value, {"schema_version", "predictions"}, "predictions",
        )

    assert marker not in str(exc_info.value)
    assert "private value" not in str(exc_info.value)
    assert str(exc_info.value) == "predictions contains unexpected fields"


@pytest.mark.parametrize("marker", PRIVATE_WARNING_MARKERS)
@pytest.mark.parametrize("location", ["warning", "prediction_envelope"])
@pytest.mark.parametrize("json_mode", [False, True])
def test_unexpected_field_names_are_redacted_in_cli_errors(
    tmp_path, capsys, marker, location, json_mode,
) -> None:
    warning = _warning_payload()
    if location == "warning":
        warning[marker] = "synthetic private payload"
    golden_path = _write_golden_file(tmp_path, [_golden_case(warning_entries=[warning])])
    predictions_path = _write_predictions_file(tmp_path, [{"case_id": "case_001"}])
    if location == "prediction_envelope":
        payload = json.loads(predictions_path.read_text())
        payload[marker] = "synthetic private payload"
        predictions_path.write_text(json.dumps(payload))

    status = _script_module().main([
        "--golden", str(golden_path), "--predictions", str(predictions_path),
        *(["--json"] if json_mode else []),
    ])

    output = capsys.readouterr()
    label = (
        "golden labels cases[0].expected_validation_result.warnings[0]"
        if location == "warning" else "predictions"
    )
    assert status == 2
    assert output.out == ""
    assert marker not in output.err
    assert "synthetic private payload" not in output.err
    assert output.err == f"Invalid evaluation input: {label} contains unexpected fields\n"


@pytest.mark.parametrize("include_metrics", [False, True])
@pytest.mark.parametrize(
    "prediction_state",
    ["missing_case", "missing_validation", "missing_warnings", "empty_warnings", "present_warnings"],
)
@pytest.mark.parametrize(
    "warning",
    [
        "UNKNOWN_SYNTHETIC_CODE",
        {**_warning_payload(), "code": "UNKNOWN_SYNTHETIC_CODE"},
        *({**_warning_payload(), **metadata} for metadata in NONCANONICAL_WARNING_METADATA),
    ],
)
def test_golden_warning_validation_is_independent_of_prediction_presence(
    include_metrics, prediction_state, warning,
) -> None:
    evaluation = _metric_module()
    golden = _golden_case(
        include_raw_visual_metrics=include_metrics,
        include_derived_metrics=include_metrics,
        expected_is_valid=False,
        warning_entries=[warning],
    )
    prediction = _prediction(warnings=[_warning_payload()])
    if prediction_state == "missing_validation":
        del prediction["validation_result"]
    elif prediction_state == "missing_warnings":
        del prediction["validation_result"]["warnings"]
    elif prediction_state == "empty_warnings":
        prediction["validation_result"]["warnings"] = []
    missing_case = prediction_state == "missing_case"

    with pytest.raises(ValueError, match="unknown warning code|canonical"):
        evaluation.evaluate_bodygraph_prediction(
            case_id="case_001", golden=golden, prediction={} if missing_case else prediction,
        )
    with pytest.raises(ValueError, match="unknown warning code|canonical"):
        evaluation.evaluate_bodygraph_predictions(
            golden_cases=[golden], predictions={} if missing_case else {"case_001": prediction},
        )


@pytest.mark.parametrize(
    "expected_warnings,prediction,expected_scores",
    [
        ([], {}, (0.0, 0.0, 0.0)),
        ([], {"validation_result": {"warnings": []}}, (1.0, 1.0, 1.0)),
        ([], {"validation_result": {"warnings": ["MISSING_ACTIVATION"]}}, (0.0, 1.0, 0.0)),
        (["MISSING_ACTIVATION"], {}, (0.0, 0.0, 0.0)),
        (["MISSING_ACTIVATION"], {"validation_result": {"warnings": []}}, (0.0, 0.0, 0.0)),
        (
            ["MISSING_ACTIVATION"],
            {"validation_result": {"warnings": ["MISSING_ACTIVATION"]}},
            (1.0, 1.0, 1.0),
        ),
    ],
)
def test_golden_warning_validation_preserves_missing_and_empty_scoring(
    expected_warnings, prediction, expected_scores,
) -> None:
    result = _metric_module().evaluate_bodygraph_prediction(
        case_id="case_001",
        golden=_golden_case(
            include_raw_visual_metrics=False, include_derived_metrics=False,
            warning_entries=expected_warnings,
        ),
        prediction=prediction,
    )

    assert set(result.metrics) == VALIDATION_METRIC_NAMES
    assert tuple(
        result.metrics[f"warning_code_{metric}"] for metric in ("precision", "recall", "f1")
    ) == expected_scores


def test_non_list_prediction_warnings_raise_input_error() -> None:
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

    with pytest.raises(ValueError, match="validation_result.warnings"):
        evaluation.evaluate_bodygraph_prediction(
            case_id="case_001",
            golden=_golden_case(warning_entries=[expected_warning]),
            prediction=prediction,
        )


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


@pytest.mark.parametrize("flag", ["include_raw_visual_metrics", "include_derived_metrics"])
@pytest.mark.parametrize(
    "value", [0, 1, -1, 0.0, 1.0, "true", "false", "", None, [], {}, ["truthy"]],
)
def test_scope_flags_require_strict_booleans(check_scope, flag, value) -> None:
    golden_case = _golden_case()
    golden_case["evaluation_scope"][flag] = value

    with pytest.raises(ValueError, match=f"{flag} must be a bool"):
        check_scope(golden_case)


@pytest.mark.parametrize(
    "scope",
    [
        "missing", None, [], {},
        {"include_raw_visual_metrics": True},
        {"include_derived_metrics": True},
        {"include_raw_visual_metrics": True, "include_derived_metrics": True, "extra": False},
    ],
)
def test_scope_must_be_explicit_and_complete(check_scope, scope) -> None:
    golden = _golden_case()
    if scope == "missing":
        del golden["evaluation_scope"]
    else:
        golden["evaluation_scope"] = scope

    with pytest.raises(ValueError, match="evaluation_scope"):
        check_scope(golden)


@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize("column", ["personality", "design"])
@pytest.mark.parametrize("field", PLANETARY_FIELDS)
def test_derived_scope_requires_every_source_activation(
    check_scope, include_raw, column, field,
) -> None:
    golden = _golden_case(include_raw_visual_metrics=include_raw)
    golden["expected_raw_labels"][column][field] = None

    with pytest.raises(ValueError, match=f"{column}.{field}"):
        check_scope(golden)


@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize(
    "activation",
    [
        "", "   ", False, True, 0, 61.4, "unknown", "61.x",
        "0.1", "65.1", "1.0", "1.7", {},
        {"gate": 61}, {"line": 4},
        {"gate": True, "line": 4}, {"gate": 61, "line": False},
        {"gate": 61.0, "line": 4}, {"gate": "61", "line": 4},
        {"gate": 0, "line": 1}, {"gate": 1, "line": 7},
    ],
)
def test_derived_scope_rejects_invalid_source_activations(
    check_scope, include_raw, activation,
) -> None:
    golden = _golden_case(include_raw_visual_metrics=include_raw)
    golden["expected_raw_labels"]["design"]["pluto"] = activation

    with pytest.raises(ValueError, match="design.pluto"):
        check_scope(golden)


@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize(
    "activation", ["1.1", "64.6", " 61.4 ", {"gate": 1, "line": 1}, {"gate": 64, "line": 6}],
)
def test_derived_scope_accepts_valid_source_activation_representations(
    check_scope, include_raw, activation,
) -> None:
    golden = _golden_case(include_raw_visual_metrics=include_raw)
    for column in ("personality", "design"):
        golden["expected_raw_labels"][column] = dict.fromkeys(PLANETARY_FIELDS, activation)

    check_scope(golden)


@pytest.mark.parametrize(
    "path",
    [
        "expected_raw_labels",
        "expected_raw_labels.personality",
        "expected_raw_labels.design.pluto",
        "expected_derived_labels",
        "expected_derived_labels.active_gates",
        "expected_derived_labels.active_channels",
        "expected_derived_labels.defined_centers",
        "expected_derived_labels.basic_info",
        "expected_derived_labels.basic_info.profile",
    ],
)
def test_scope_rejects_missing_required_label_fields(check_scope, path) -> None:
    golden = _golden_case()
    parts = path.split(".")
    parent = golden
    for part in parts[:-1]:
        parent = parent[part]
    del parent[parts[-1]]

    with pytest.raises(ValueError, match=parts[0]):
        check_scope(golden)


def test_disabled_derived_scope_requires_explicit_null_labels(check_scope) -> None:
    golden = _golden_case(include_derived_metrics=False)
    del golden["expected_derived_labels"]

    with pytest.raises(ValueError, match="expected_derived_labels"):
        check_scope(golden)


@pytest.mark.parametrize("include_raw", [False, True])
def test_scope_accepts_all_null_sources_without_derived_metrics(check_scope, include_raw) -> None:
    golden = _golden_case(
        include_raw_visual_metrics=include_raw, include_derived_metrics=False,
    )
    for column in ("personality", "design"):
        golden["expected_raw_labels"][column] = dict.fromkeys(PLANETARY_FIELDS)

    check_scope(golden)


@pytest.mark.parametrize("invalid_input", ["raw_flag", "derived_flag", "activation"])
def test_invalid_scope_or_eligibility_is_a_cli_input_error(tmp_path, capsys, invalid_input) -> None:
    golden = _golden_case(include_raw_visual_metrics=False)
    if invalid_input == "activation":
        golden["expected_raw_labels"]["design"]["pluto"] = None
    else:
        flag = "include_raw_visual_metrics" if invalid_input == "raw_flag" else "include_derived_metrics"
        golden["evaluation_scope"][flag] = 1
    golden_path = _write_golden_file(tmp_path, [golden])
    predictions_path = _write_predictions_file(
        tmp_path, [{"case_id": "case_001", **_prediction()}],
    )

    status = _script_module().main([
        "--golden", str(golden_path), "--predictions", str(predictions_path), "--json",
    ])

    assert status == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "Invalid evaluation input" in output.err


@pytest.mark.parametrize(
    ("include_derived_metrics", "expected_derived_labels"),
    [
        (True, None),
        (False, _derived_labels()),
    ],
)
def test_scope_and_derived_labels_must_agree(
    check_scope,
    include_derived_metrics: bool,
    expected_derived_labels: object,
) -> None:
    golden_case = _golden_case(
        include_derived_metrics=include_derived_metrics,
    )
    golden_case["expected_derived_labels"] = expected_derived_labels

    with pytest.raises(
        ValueError,
        match="include_derived_metrics.*expected_derived_labels|"
        "expected_derived_labels.*include_derived_metrics",
    ):
        check_scope(golden_case)


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
    # Unknown codes are input errors (exit 2) in both golden and prediction files.
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
