import importlib.util
import json
from pathlib import Path
from typing import Any

from human_design.vision.constants import CANONICAL_CENTERS
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.models import ValidationSeverity, ValidationSource
from human_design.vision.parser import parse_bodygraph_raw_extraction_json


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_LABELS_PATH = REPO_ROOT / "data/bodygraph_samples/golden_labels.example.json"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

PLANETARY_FIELDS = {
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
}

PLANETARY_FIELD_ORDER = (
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
    "expected_warning_codes",
}

REQUIRED_RAW_LABEL_KEYS = {
    "personality",
    "design",
    "visually_active_gates",
    "visually_defined_centers",
    "visually_undefined_centers",
    "visible_colored_channels",
    "confidence",
    "uncertain_items",
}

REQUIRED_DERIVED_LABEL_KEYS = {
    "active_gates",
    "active_channels",
    "defined_centers",
    "type",
    "authority",
    "profile",
    "definition",
    "strategy",
    "not_self_theme",
    "signature",
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


def _confidence_payload(value: float = 1.0) -> dict[str, object]:
    return {
        "personality": {field_name: value for field_name in PLANETARY_FIELD_ORDER},
        "design": {field_name: value for field_name in PLANETARY_FIELD_ORDER},
        "visually_defined_centers": value,
        "visually_undefined_centers": value,
        "visually_active_gates": value,
        "visible_colored_channels": value,
    }


def _raw_parser_payload(raw_labels: dict[str, Any]) -> dict[str, object]:
    return {
        "personality": raw_labels["personality"],
        "design": raw_labels["design"],
        "visually_defined_centers": raw_labels.get("visually_defined_centers", []),
        "visually_undefined_centers": raw_labels.get("visually_undefined_centers", []),
        "visually_active_gates": raw_labels.get("visually_active_gates", []),
        "visible_colored_channels": raw_labels.get("visible_colored_channels", []),
        "confidence": raw_labels.get("confidence", _confidence_payload()),
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
        "visually_undefined_centers": [
            "Head",
            "Ajna",
            "Throat",
            "Ego",
            "Spleen",
            "Solar Plexus",
        ],
        "visible_colored_channels": ["3-60", "10-34"],
        "confidence": _confidence_payload(),
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


def _golden_case(
    case_id: str = "case_001",
    *,
    include_derived_metrics: bool = True,
    warning_codes: list[str] | None = None,
    warning_entries: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "evaluation_scope": {
            "mode": "full_derived_chart"
            if include_derived_metrics
            else "raw_visual_evidence_only",
            "include_derived_metrics": include_derived_metrics,
            "include_raw_visual_metrics": True,
        },
        "expected_raw_labels": _full_raw_labels(),
        "expected_derived_labels": _derived_labels()
        if include_derived_metrics
        else None,
        "expected_validation_result": {
            "is_valid": not warning_entries,
            "warnings": warning_entries or [],
        },
        "expected_warning_codes": warning_codes or [],
    }


def _prediction(
    *,
    personality: dict[str, str | None] | None = None,
    design: dict[str, str | None] | None = None,
    derived_chart_data: dict[str, Any] | None = None,
    warnings: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    raw_labels = _full_raw_labels()
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
        "validation_result": {"warnings": warnings or []},
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


def test_example_golden_labels_load_as_json() -> None:
    data = _load_golden_labels()

    assert set(data) >= {
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
        assert isinstance(case["expected_warning_codes"], list)


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
        assert set(raw_labels["personality"]) == PLANETARY_FIELDS
        assert set(raw_labels["design"]) == PLANETARY_FIELDS

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

        expected_basic_info = expected.get("basic_info", expected)
        for field_name in BASIC_INFO_FIELDS:
            assert getattr(chart.basic_info, field_name) == expected_basic_info[field_name]


def test_partial_raw_only_cases_exclude_derived_metrics_and_keep_all_planet_keys() -> None:
    partial_cases = [
        case
        for case in _cases()
        if case["evaluation_scope"]["mode"] == "raw_visual_evidence_only"
    ]
    assert partial_cases

    for case in partial_cases:
        assert case["evaluation_scope"]["include_derived_metrics"] is False
        assert case["expected_derived_labels"] is None

        raw_labels = case["expected_raw_labels"]
        assert set(raw_labels) == REQUIRED_RAW_LABEL_KEYS
        assert set(raw_labels["personality"]) == PLANETARY_FIELDS
        assert set(raw_labels["design"]) == PLANETARY_FIELDS

        activations = tuple(raw_labels["personality"].values()) + tuple(
            raw_labels["design"].values()
        )
        assert any(value is None for value in activations)
        assert all(value not in DISALLOWED_MISSING_ACTIVATION_SENTINELS for value in activations)


def test_warning_codes_match_validation_warning_entries() -> None:
    for case in _cases():
        expected_warning_codes = case["expected_warning_codes"]
        validation = case["expected_validation_result"]
        warnings = validation["warnings"]

        assert isinstance(expected_warning_codes, list)
        assert isinstance(warnings, list)
        assert [warning["code"] for warning in warnings] == expected_warning_codes
        assert all(isinstance(code, str) and code for code in expected_warning_codes)


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


def test_warning_code_metrics_and_metadata_match_rate() -> None:
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
            warning_codes=["VISIBLE_CHANNEL_NOT_DERIVED"],
            warning_entries=[expected_warning],
        ),
        prediction=_prediction(warnings=[predicted_warning]),
    )

    assert result.metrics["warning_code_precision"] == 1.0
    assert result.metrics["warning_code_recall"] == 1.0
    assert result.metrics["warning_code_f1"] == 1.0
    assert result.metrics["warning_metadata_exact_match_rate"] == 1.0


def test_script_main_reads_json_files_and_returns_nonzero_on_threshold_failure(
    tmp_path,
    capsys,
) -> None:
    script = _script_module()
    golden_path = tmp_path / "golden.json"
    predictions_path = tmp_path / "predictions.json"
    golden_path.write_text(json.dumps({"cases": [_golden_case("case_001")]}))
    predictions_path.write_text(json.dumps({"case_001": _prediction()}))

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
