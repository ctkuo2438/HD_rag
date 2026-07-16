import json
from pathlib import Path
from typing import Any

import pytest

from human_design.vision.models import (
    Activation,
    RawVisionExtraction,
    ValidationCode,
    ValidationSeverity,
    ValidationSource,
    ValidationWarning,
)
from human_design.vision.parser import (
    BodyGraphParseError,
    parse_bodygraph_raw_extraction_json,
)
from human_design.vision.prompt import load_bodygraph_raw_extraction_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = REPO_ROOT / "prompts/bodygraph_raw_extraction.txt"

PLANET_FIELDS = (
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


def _activation_values() -> dict[str, str]:
    return {
        "sun": "61.4",
        "earth": "62.4",
        "north_node": "10.2",
        "south_node": "15.2",
        "moon": "34.1",
        "mercury": "20.3",
        "venus": "57.5",
        "mars": "3.1",
        "jupiter": "60.1",
        "saturn": "14.6",
        "uranus": "2.6",
        "neptune": "45.2",
        "pluto": "50.4",
    }


def _valid_payload() -> dict[str, Any]:
    return {
        "personality": _activation_values(),
        "design": {
            **_activation_values(),
            "sun": "32.6",
            "earth": "42.6",
        },
        "visually_defined_centers": ["Throat", "G Center", "Sacral"],
        "visually_active_gates": [10, "20", 34, 57, "60"],
        "visible_colored_channels": ["10-34", "3 - 60"],
        "uncertain_items": [
            {
                "field_path": "personality.sun",
                "observed_value": "61.4",
                "reason": "Synthetic uncertainty item.",
                "confidence": 0.5,
            }
        ],
    }


def test_simplified_six_field_raw_schema_parses_without_fixed_confidence() -> None:
    result = _parse(_valid_payload())

    assert not hasattr(result.raw_vision, "confidence")


def test_removed_fixed_confidence_is_rejected_as_a_top_level_extra() -> None:
    payload = _valid_payload()
    payload["confidence"] = {}

    with pytest.raises(BodyGraphParseError, match="confidence"):
        _parse(payload)


def _parse(payload: dict[str, Any]):
    return parse_bodygraph_raw_extraction_json(json.dumps(payload))


def _warning_codes(warnings: tuple[ValidationWarning, ...]) -> tuple[ValidationCode, ...]:
    return tuple(warning.code for warning in warnings)


def _warnings_by_code(
    warnings: tuple[ValidationWarning, ...],
    code: ValidationCode,
) -> tuple[ValidationWarning, ...]:
    return tuple(warning for warning in warnings if warning.code is code)


def test_prompt_file_and_loader_return_raw_fact_instructions() -> None:
    assert PROMPT_PATH.exists()

    prompt = load_bodygraph_raw_extraction_prompt()
    prompt_lower = prompt.lower()

    assert "extract only raw visible facts" in prompt_lower
    assert "strict json only" in prompt_lower
    assert "json null" in prompt_lower
    assert "each uncertain item confidence" in prompt_lower
    assert "0.0 through 1.0" in prompt_lower
    assert '"confidence": {' not in prompt_lower

    for field_name in PLANET_FIELDS:
        assert field_name in prompt

    for forbidden_inference in (
        "do not infer type",
        "do not infer authority",
        "do not infer profile",
        "do not infer definition",
        "do not infer strategy",
        "do not infer not_self_theme",
        "do not infer signature",
    ):
        assert forbidden_inference in prompt_lower


def test_parse_valid_strict_json_into_typed_raw_vision_models() -> None:
    result = _parse(_valid_payload())

    assert isinstance(result.raw_vision, RawVisionExtraction)
    assert result.raw_vision.personality.sun == Activation(gate=61, line=4)
    assert result.raw_vision.design.sun == Activation(gate=32, line=6)
    assert result.raw_vision.personality.jupiter == Activation(gate=60, line=1)
    assert result.raw_vision.personality.pluto == Activation(gate=50, line=4)
    assert result.raw_vision.visually_defined_centers == ("Throat", "G", "Sacral")
    assert result.raw_vision.visually_active_gates == (10, 20, 34, 57, 60)
    assert result.raw_vision.visible_colored_channels == ("10-34", "3-60")
    assert result.raw_vision.uncertain_items[0].field_path == "personality.sun"
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("raw_alias", "canonical_center"),
    [
        ("head", "Head"),
        ("head center", "Head"),
        ("ajna", "Ajna"),
        ("ajna center", "Ajna"),
        ("throat", "Throat"),
        ("throat center", "Throat"),
        ("g", "G"),
        ("g center", "G"),
        ("g centre", "G"),
        ("g_center", "G"),
        ("self", "G"),
        ("self center", "G"),
        ("identity", "G"),
        ("identity center", "G"),
        ("ego", "Ego"),
        ("ego center", "Ego"),
        ("heart", "Ego"),
        ("heart center", "Ego"),
        ("will", "Ego"),
        ("will center", "Ego"),
        ("sacral", "Sacral"),
        ("sacral center", "Sacral"),
        ("spleen", "Spleen"),
        ("spleen center", "Spleen"),
        ("splenic", "Spleen"),
        ("splenic center", "Spleen"),
        ("solar plexus", "Solar Plexus"),
        ("solar plexus center", "Solar Plexus"),
        ("solar_plexus", "Solar Plexus"),
        ("solar_plexus_center", "Solar Plexus"),
        ("emotional", "Solar Plexus"),
        ("emotional center", "Solar Plexus"),
        ("heart_center", "Ego"),
        ("ego_center", "Ego"),
        ("root", "Root"),
        ("root center", "Root"),
        ("root_center", "Root"),
        ("throat_center", "Throat"),
        ("spleen_center", "Spleen"),
        ("sacral_center", "Sacral"),
        ("head_center", "Head"),
        ("ajna_center", "Ajna"),
    ],
)
def test_real_vision_center_aliases_normalize_to_canonical_names(
    raw_alias: str,
    canonical_center: str,
) -> None:
    payload = _valid_payload()
    payload["visually_defined_centers"] = [raw_alias]

    result = _parse(payload)

    assert result.raw_vision.visually_defined_centers == (canonical_center,)
    assert result.warnings == ()


def test_center_normalization_handles_case_whitespace_and_separators(
) -> None:
    payload = _valid_payload()
    payload["visually_defined_centers"] = ["  SoLaR___PlExUs---CeNtEr  "]

    result = _parse(payload)

    assert result.raw_vision.visually_defined_centers == ("Solar Plexus",)
    assert result.warnings == ()


def test_invalid_visual_centers_are_skipped_with_indexed_nonfatal_warnings() -> None:
    payload = _valid_payload()
    payload["visually_defined_centers"] = [
        "Sacral",
        "mystery_center",
        123,
        "Root",
    ]

    result = _parse(payload)

    assert result.raw_vision.visually_defined_centers == ("Sacral", "Root")
    assert _warning_codes(result.warnings) == (
        ValidationCode.INVALID_VISUAL_CENTER,
        ValidationCode.INVALID_VISUAL_CENTER,
    )
    assert "visually_defined_centers[1]" in result.warnings[0].message
    assert "visually_defined_centers[2]" in result.warnings[1].message
    assert all(
        warning.severity is ValidationSeverity.WARNING
        and warning.affects_validity is False
        and warning.source is ValidationSource.parser
        for warning in result.warnings
    )


@pytest.mark.parametrize(
    ("field_name", "raw_value"),
    [
        ("visually_defined_centers", "Sacral"),
    ],
)
def test_visual_center_containers_must_remain_lists(
    field_name: str,
    raw_value: object,
) -> None:
    payload = _valid_payload()
    payload[field_name] = raw_value

    with pytest.raises(BodyGraphParseError, match=field_name):
        _parse(payload)


def test_malformed_json_raises_parse_error() -> None:
    with pytest.raises(BodyGraphParseError, match="Invalid JSON"):
        parse_bodygraph_raw_extraction_json('{"personality":')


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_numeric_constants_are_rejected(constant: str) -> None:
    with pytest.raises(BodyGraphParseError, match="Invalid JSON|non-standard"):
        parse_bodygraph_raw_extraction_json(f'{{"value": {constant}}}')


def test_nested_nonfinite_uncertain_observed_value_is_rejected_at_json_decode() -> None:
    payload = json.dumps(_valid_payload()).replace(
        '"observed_value": "61.4"',
        '"observed_value": NaN',
    )

    with pytest.raises(BodyGraphParseError, match="Invalid JSON|non-standard"):
        parse_bodygraph_raw_extraction_json(payload)


@pytest.mark.parametrize(
    "missing_field",
    [
        "personality",
        "design",
        "visually_defined_centers",
        "visually_active_gates",
        "visible_colored_channels",
        "uncertain_items",
    ],
)
def test_missing_required_top_level_fields_raise_parse_error(
    missing_field: str,
) -> None:
    payload = _valid_payload()
    del payload[missing_field]

    with pytest.raises(BodyGraphParseError, match=missing_field):
        _parse(payload)


def test_unexpected_top_level_field_remains_rejected() -> None:
    payload = _valid_payload()
    payload["provider_metadata"] = {}

    with pytest.raises(BodyGraphParseError, match="provider_metadata"):
        _parse(payload)


@pytest.mark.parametrize("container", ["personality", "design"])
def test_personality_and_design_containers_must_be_objects(container: str) -> None:
    payload = _valid_payload()
    payload[container] = []

    with pytest.raises(BodyGraphParseError, match=container):
        _parse(payload)


@pytest.mark.parametrize("column_name", ["personality", "design"])
def test_activation_columns_reject_extra_keys_with_full_path(
    column_name: str,
) -> None:
    payload = _valid_payload()
    payload[column_name]["ascendant"] = "12.3"

    with pytest.raises(BodyGraphParseError, match=rf"{column_name}\.ascendant"):
        _parse(payload)


def test_uncertain_items_reject_extra_keys_with_indexed_full_path() -> None:
    payload = _valid_payload()
    payload["uncertain_items"][0]["type"] = "Generator"

    with pytest.raises(BodyGraphParseError, match=r"uncertain_items\[0\]\.type"):
        _parse(payload)


def test_null_activation_value_becomes_none_without_warning() -> None:
    payload = _valid_payload()
    payload["personality"]["moon"] = None

    result = _parse(payload)

    assert result.raw_vision.personality.moon is None
    assert result.warnings == ()


def test_blank_activation_value_becomes_none_and_warns() -> None:
    payload = _valid_payload()
    payload["personality"]["moon"] = "   "

    result = _parse(payload)

    assert result.raw_vision.personality.moon is None
    warning = result.warnings[0]
    assert warning.code is ValidationCode.MISSING_ACTIVATION
    assert warning.source is ValidationSource.parser
    assert warning.severity is ValidationSeverity.ERROR
    assert warning.affects_validity is True


def test_omitted_non_sun_activation_becomes_none_and_warns_missing_activation() -> None:
    payload = _valid_payload()
    del payload["personality"]["moon"]

    result = _parse(payload)

    assert result.raw_vision.personality.moon is None
    assert _warning_codes(result.warnings) == (ValidationCode.MISSING_ACTIVATION,)


def test_omitted_personality_sun_uses_specific_warning_only() -> None:
    payload = _valid_payload()
    del payload["personality"]["sun"]

    result = _parse(payload)

    assert result.raw_vision.personality.sun is None
    assert _warning_codes(result.warnings) == (ValidationCode.MISSING_PERSONALITY_SUN,)
    assert ValidationCode.MISSING_ACTIVATION not in _warning_codes(result.warnings)


def test_omitted_design_sun_uses_specific_warning_only() -> None:
    payload = _valid_payload()
    del payload["design"]["sun"]

    result = _parse(payload)

    assert result.raw_vision.design.sun is None
    assert _warning_codes(result.warnings) == (ValidationCode.MISSING_DESIGN_SUN,)
    assert ValidationCode.MISSING_ACTIVATION not in _warning_codes(result.warnings)


@pytest.mark.parametrize("raw_value", ["unknown", "61.x", "abc", "61", "61.4.2"])
def test_malformed_activation_values_store_none_and_warn(raw_value: str) -> None:
    payload = _valid_payload()
    payload["personality"]["moon"] = raw_value

    result = _parse(payload)

    assert result.raw_vision.personality.moon is None
    warning = result.warnings[0]
    assert warning.code is ValidationCode.MALFORMED_ACTIVATION
    assert warning.source is ValidationSource.parser
    assert warning.severity is ValidationSeverity.ERROR
    assert warning.affects_validity is True


def test_out_of_range_gate_is_stored_and_warns_invalid_gate() -> None:
    payload = _valid_payload()
    payload["personality"]["moon"] = "99.1"

    result = _parse(payload)

    assert result.raw_vision.personality.moon == Activation(gate=99, line=1)
    assert _warning_codes(result.warnings) == (ValidationCode.INVALID_ACTIVATION_GATE,)


def test_out_of_range_line_is_stored_and_warns_invalid_line() -> None:
    payload = _valid_payload()
    payload["personality"]["moon"] = "60.9"

    result = _parse(payload)

    assert result.raw_vision.personality.moon == Activation(gate=60, line=9)
    assert _warning_codes(result.warnings) == (ValidationCode.INVALID_ACTIVATION_LINE,)


def test_channel_normalization_and_invalid_channel_warnings() -> None:
    payload = _valid_payload()
    payload["visible_colored_channels"] = [
        "10-34",
        "3 - 60",
        "57-10",
        "34-99",
        "10-99",
    ]

    result = _parse(payload)

    assert result.raw_vision.visible_colored_channels == (
        "10-34",
        "3-60",
        "10-57",
    )
    assert _warning_codes(result.warnings) == (
        ValidationCode.VISIBLE_CHANNEL_NORMALIZED,
        ValidationCode.INVALID_VISIBLE_CHANNEL,
        ValidationCode.INVALID_VISIBLE_CHANNEL,
    )
    normalized_warning = result.warnings[0]
    assert normalized_warning.severity is ValidationSeverity.INFO
    assert normalized_warning.affects_validity is False
    invalid_warnings = _warnings_by_code(
        result.warnings,
        ValidationCode.INVALID_VISIBLE_CHANNEL,
    )
    assert all(warning.severity is ValidationSeverity.WARNING for warning in invalid_warnings)
    assert all(warning.affects_validity is False for warning in invalid_warnings)


def test_visually_active_gates_normalize_to_integers_and_invalid_gates_warn() -> None:
    payload = _valid_payload()
    payload["visually_active_gates"] = [10, "20", 34, "99", "not-a-gate", True, 60.5]

    result = _parse(payload)

    assert result.raw_vision.visually_active_gates == (10, 20, 34)
    assert _warning_codes(result.warnings) == (
        ValidationCode.INVALID_VISUALLY_ACTIVE_GATE,
        ValidationCode.INVALID_VISUALLY_ACTIVE_GATE,
        ValidationCode.INVALID_VISUALLY_ACTIVE_GATE,
        ValidationCode.INVALID_VISUALLY_ACTIVE_GATE,
    )
    assert all(warning.source is ValidationSource.parser for warning in result.warnings)
    assert all(
        warning.severity is ValidationSeverity.WARNING
        and warning.affects_validity is False
        for warning in result.warnings
    )


def test_invalid_visual_gate_is_skipped_with_nonfatal_visual_warning() -> None:
    payload = _valid_payload()
    payload["visually_active_gates"] = [3, 60, 99]

    result = _parse(payload)

    assert result.raw_vision.visually_active_gates == (3, 60)
    assert _warning_codes(result.warnings) == (
        ValidationCode.INVALID_VISUALLY_ACTIVE_GATE,
    )
    warning = result.warnings[0]
    assert warning.severity is ValidationSeverity.WARNING
    assert warning.affects_validity is False
    assert warning.source is ValidationSource.parser


@pytest.mark.parametrize(
    "field_path",
    [
        "personality.sun",
        "design.mars",
        "visually_defined_centers[0]",
        "visually_active_gates[0]",
        "visible_colored_channels[0]",
    ],
)
def test_uncertain_items_accept_raw_vision_field_paths(field_path: str) -> None:
    payload = _valid_payload()
    payload["uncertain_items"] = [
        {
            "field_path": field_path,
            "observed_value": None,
            "reason": "Synthetic ambiguous raw observation.",
            "confidence": 0.5,
        }
    ]

    result = _parse(payload)

    assert result.raw_vision.uncertain_items[0].field_path == field_path


@pytest.mark.parametrize(
    "field_path",
    [
        "derived_chart_data.basic_info.type",
        "validation.is_valid",
        "active_channels",
        "type",
        "authority",
        "profile",
    ],
)
def test_uncertain_items_reject_derived_field_paths(field_path: str) -> None:
    payload = _valid_payload()
    payload["uncertain_items"] = [
        {
            "field_path": field_path,
            "observed_value": None,
            "reason": "Not a raw Vision field.",
            "confidence": 0.5,
        }
    ]

    with pytest.raises(BodyGraphParseError, match="uncertain_items"):
        _parse(payload)


def test_parser_warnings_merge_into_validation_result() -> None:
    payload = _valid_payload()
    payload["visible_colored_channels"] = ["57-10"]
    result = _parse(payload)

    validation = result.to_validation_result()

    assert validation.warnings == result.warnings
    assert validation.warnings[0].code is ValidationCode.VISIBLE_CHANNEL_NORMALIZED
    assert validation.warnings[0].source is ValidationSource.parser


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "type",
        "authority",
        "profile",
        "definition",
        "strategy",
        "not_self_theme",
        "signature",
        "active_gates",
        "active_channels",
        "defined_centers",
        "basic_info",
        "derived_chart_data",
    ],
)
def test_final_human_design_concepts_are_rejected_as_vision_facts(
    forbidden_field: str,
) -> None:
    payload = _valid_payload()
    payload[forbidden_field] = "Vision guessed value"

    with pytest.raises(BodyGraphParseError, match="final Human Design concepts"):
        _parse(payload)


def test_parser_does_not_derive_basic_info() -> None:
    result = _parse(_valid_payload())

    assert not hasattr(result.raw_vision, "basic_info")
    assert not hasattr(result.raw_vision, "derived_chart_data")
