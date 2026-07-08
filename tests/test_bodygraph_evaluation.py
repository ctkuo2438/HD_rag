import json
from pathlib import Path
from typing import Any

from human_design.vision.constants import CANONICAL_CENTERS


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_LABELS_PATH = REPO_ROOT / "data/bodygraph_samples/golden_labels.example.json"
FIXTURE_IMAGE_PATH = REPO_ROOT / "tests/fixtures/bodygraph/test1.png"
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


def test_safe_fixture_image_exists_and_is_small_png() -> None:
    assert FIXTURE_IMAGE_PATH.exists()
    assert FIXTURE_IMAGE_PATH.is_file()
    assert FIXTURE_IMAGE_PATH.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert FIXTURE_IMAGE_PATH.stat().st_size < 10 * 1024


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
