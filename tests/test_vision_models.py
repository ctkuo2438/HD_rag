import math
from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any, cast, overload

import pytest

import human_design.vision as vision
from human_design.vision import (
    Activation,
    ActivationConfidenceColumn,
    BodyGraphExtractionResult,
    DerivedBasicInfo,
    DerivedChartData,
    DesignActivationColumn,
    ParseResult,
    PersonalityActivationColumn,
    RawVisionConfidence,
    RawVisionExtraction,
    UncertainItem,
    ValidationCode,
    ValidationResult,
    ValidationSeverity,
    ValidationSource,
    ValidationWarning,
)


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

PUBLIC_MODEL_NAMES = (
    "Activation",
    "PersonalityActivationColumn",
    "DesignActivationColumn",
    "ActivationConfidenceColumn",
    "RawVisionConfidence",
    "UncertainItem",
    "RawVisionExtraction",
    "DerivedBasicInfo",
    "DerivedChartData",
    "ValidationSeverity",
    "ValidationSource",
    "ValidationCode",
    "ValidationWarning",
    "ValidationResult",
    "ParseResult",
    "BodyGraphExtractionResult",
)

EXPECTED_VALIDATION_CODE_VALUES = (
    "VISIBLE_CHANNEL_NORMALIZED",
    "INVALID_VISIBLE_CHANNEL",
    "VISIBLE_CHANNEL_NOT_DERIVED",
    "DERIVED_CHANNEL_NOT_VISIBLE",
    "VISUALLY_ACTIVE_GATES_MISMATCH",
    "VISUALLY_DEFINED_CENTERS_MISMATCH",
    "UNSUPPORTED_AUTHORITY",
    "MISSING_PERSONALITY_SUN",
    "MISSING_DESIGN_SUN",
    "MISSING_ACTIVATION",
    "MALFORMED_ACTIVATION",
    "INVALID_ACTIVATION_GATE",
    "INVALID_ACTIVATION_LINE",
)


@overload
def _activation_column(
    model: type[PersonalityActivationColumn],
    *,
    missing_field: str | None = None,
) -> PersonalityActivationColumn: ...


@overload
def _activation_column(
    model: type[DesignActivationColumn],
    *,
    missing_field: str | None = None,
) -> DesignActivationColumn: ...


def _activation_column(
    model: type[PersonalityActivationColumn] | type[DesignActivationColumn],
    *,
    missing_field: str | None = None,
) -> PersonalityActivationColumn | DesignActivationColumn:
    values = {
        field_name: None
        if field_name == missing_field
        else Activation(gate=index + 1, line=(index % 6) + 1)
        for index, field_name in enumerate(PLANET_FIELDS)
    }
    constructor = cast(Any, model)
    return constructor(**values)


def _confidence_column(value: float = 0.8) -> ActivationConfidenceColumn:
    return ActivationConfidenceColumn(
        **{field_name: value for field_name in PLANET_FIELDS}
    )


def _raw_confidence(value: float = 0.8) -> RawVisionConfidence:
    return RawVisionConfidence(
        personality=_confidence_column(value),
        design=_confidence_column(value),
        visually_defined_centers=value,
        visually_undefined_centers=value,
        visually_active_gates=value,
        visible_colored_channels=value,
    )


def _raw_vision(
    *,
    personality: PersonalityActivationColumn | None = None,
    design: DesignActivationColumn | None = None,
) -> RawVisionExtraction:
    return RawVisionExtraction(
        personality=personality or _activation_column(PersonalityActivationColumn),
        design=design or _activation_column(DesignActivationColumn),
        visually_defined_centers=("Sacral", "Throat"),
        visually_undefined_centers=("Head", "Ajna"),
        visually_active_gates=(20, 34, 57),
        visible_colored_channels=("20-34",),
        confidence=_raw_confidence(),
        uncertain_items=(
            UncertainItem(
                field_path="personality.sun",
                observed_value="61.x",
                reason="Line digit is visually ambiguous.",
                confidence=0.35,
            ),
        ),
    )


def _derived_basic_info() -> DerivedBasicInfo:
    return DerivedBasicInfo(
        type="Manifesting Generator",
        authority="Sacral",
        profile="1/4",
        strategy="To Respond",
        definition="Single Definition",
        not_self_theme="Frustration",
        signature="Satisfaction",
    )


def _derived_chart_data() -> DerivedChartData:
    return DerivedChartData(
        basic_info=_derived_basic_info(),
        active_gates=(20, 34, 57),
        active_channels=("20-34",),
        defined_centers=("Sacral", "Throat"),
    )


def _warning(
    *,
    code: ValidationCode = ValidationCode.MISSING_ACTIVATION,
    severity: ValidationSeverity = ValidationSeverity.WARNING,
    affects_validity: bool = False,
    source: ValidationSource = ValidationSource.validation,
) -> ValidationWarning:
    return ValidationWarning(
        code=code,
        message=f"{code.value} test warning",
        severity=severity,
        affects_validity=affects_validity,
        source=source,
    )


@pytest.mark.parametrize(
    "model",
    [
        Activation,
        PersonalityActivationColumn,
        DesignActivationColumn,
        ActivationConfidenceColumn,
        RawVisionConfidence,
        UncertainItem,
        RawVisionExtraction,
        DerivedBasicInfo,
        DerivedChartData,
        ValidationWarning,
        ValidationResult,
        ParseResult,
        BodyGraphExtractionResult,
    ],
)
def test_models_are_frozen_dataclasses(model: type[object]) -> None:
    assert is_dataclass(model)
    params = cast(Any, model).__dataclass_params__
    assert params.frozen is True


def test_public_models_are_exported_from_vision_package() -> None:
    for model_name in PUBLIC_MODEL_NAMES:
        assert getattr(vision, model_name)


def test_complete_raw_vision_extraction_can_be_constructed() -> None:
    raw = _raw_vision()

    assert raw.personality.sun == Activation(gate=1, line=1)
    assert raw.design.pluto == Activation(gate=13, line=1)
    assert raw.visually_defined_centers == ("Sacral", "Throat")
    assert raw.visually_undefined_centers == ("Head", "Ajna")
    assert raw.visually_active_gates == (20, 34, 57)
    assert raw.visible_colored_channels == ("20-34",)
    assert raw.confidence.visually_active_gates == 0.8
    assert raw.uncertain_items[0].field_path == "personality.sun"


def test_raw_vision_collections_have_safe_empty_defaults() -> None:
    raw = RawVisionExtraction(
        personality=_activation_column(PersonalityActivationColumn),
        design=_activation_column(DesignActivationColumn),
        confidence=_raw_confidence(),
    )

    assert raw.visually_defined_centers == ()
    assert raw.visually_undefined_centers == ()
    assert raw.visually_active_gates == ()
    assert raw.visible_colored_channels == ()
    assert raw.uncertain_items == ()


def test_derived_basic_info_and_chart_data_can_be_constructed() -> None:
    basic_info = _derived_basic_info()
    derived = _derived_chart_data()

    assert basic_info.type == "Manifesting Generator"
    assert basic_info.authority == "Sacral"
    assert derived.basic_info == basic_info
    assert derived.active_gates == (20, 34, 57)
    assert derived.active_channels == ("20-34",)
    assert derived.defined_centers == ("Sacral", "Throat")


def test_personality_activation_column_has_all_canonical_fields() -> None:
    assert tuple(field.name for field in fields(PersonalityActivationColumn)) == (
        PLANET_FIELDS
    )


def test_design_activation_column_has_all_canonical_fields() -> None:
    assert tuple(field.name for field in fields(DesignActivationColumn)) == PLANET_FIELDS


def test_activation_values_and_none_are_representable() -> None:
    personality = _activation_column(PersonalityActivationColumn, missing_field="moon")
    design = _activation_column(DesignActivationColumn, missing_field="venus")

    assert personality.sun == Activation(gate=1, line=1)
    assert personality.moon is None
    assert design.sun == Activation(gate=1, line=1)
    assert design.venus is None


def test_out_of_range_integer_activation_values_are_representable() -> None:
    invalid_gate = Activation(gate=99, line=1)
    invalid_line = Activation(gate=60, line=9)

    assert invalid_gate.gate == 99
    assert invalid_gate.line == 1
    assert invalid_line.gate == 60
    assert invalid_line.line == 9


@pytest.mark.parametrize("bad_value", [True, False, "61", 61.0, None])
def test_activation_rejects_invalid_gate_types(bad_value: object) -> None:
    with pytest.raises(TypeError):
        Activation(gate=cast(int, bad_value), line=4)


@pytest.mark.parametrize("bad_value", [True, False, "4", 4.0, None])
def test_activation_rejects_invalid_line_types(bad_value: object) -> None:
    with pytest.raises(TypeError):
        Activation(gate=61, line=cast(int, bad_value))


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_valid_finite_confidence_values_are_accepted(value: float) -> None:
    confidence_column = _confidence_column(value)
    raw_confidence = _raw_confidence(value)
    uncertain_item = UncertainItem(
        field_path="design.mars",
        observed_value=None,
        reason="Activation is not readable.",
        confidence=value,
    )

    assert confidence_column.sun == value
    assert raw_confidence.visible_colored_channels == value
    assert uncertain_item.confidence == value


@pytest.mark.parametrize(
    "bad_value",
    [-0.01, 1.01, math.nan, math.inf, -math.inf, "high", "85%"],
)
def test_invalid_activation_confidence_values_are_rejected(
    bad_value: object,
) -> None:
    values: dict[str, float] = {field_name: 0.5 for field_name in PLANET_FIELDS}
    values["sun"] = cast(float, bad_value)

    with pytest.raises((TypeError, ValueError)):
        ActivationConfidenceColumn(**values)


@pytest.mark.parametrize(
    "bad_value",
    [-0.01, 1.01, math.nan, math.inf, -math.inf, "high", "85%"],
)
def test_invalid_raw_visual_confidence_values_are_rejected(
    bad_value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RawVisionConfidence(
            personality=_confidence_column(),
            design=_confidence_column(),
            visually_defined_centers=cast(float, bad_value),
            visually_undefined_centers=0.5,
            visually_active_gates=0.5,
            visible_colored_channels=0.5,
        )


@pytest.mark.parametrize(
    "bad_value",
    [-0.01, 1.01, math.nan, math.inf, -math.inf, "high", "85%"],
)
def test_invalid_uncertain_item_confidence_values_are_rejected(
    bad_value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        UncertainItem(
            field_path="personality.mercury",
            observed_value="unclear",
            reason="Activation is visually ambiguous.",
            confidence=cast(float, bad_value),
        )


def test_uncertain_item_preserves_ambiguous_raw_observations() -> None:
    item = UncertainItem(
        field_path="personality.sun",
        observed_value="61.x",
        reason="Line digit is visually ambiguous.",
        confidence=0.35,
    )

    assert item.field_path == "personality.sun"
    assert item.observed_value == "61.x"
    assert item.reason == "Line digit is visually ambiguous."
    assert item.confidence == 0.35


@pytest.mark.parametrize(
    "field_path",
    [
        *(f"personality.{field_name}" for field_name in PLANET_FIELDS),
        *(f"design.{field_name}" for field_name in PLANET_FIELDS),
        "visually_defined_centers",
        "visually_defined_centers[0]",
        "visually_undefined_centers",
        "visually_undefined_centers[0]",
        "visually_active_gates",
        "visually_active_gates[0]",
        "visible_colored_channels",
        "visible_colored_channels[0]",
    ],
)
def test_valid_raw_vision_field_paths_are_accepted(field_path: str) -> None:
    assert (
        UncertainItem(
            field_path=field_path,
            observed_value=None,
            reason="Raw field is unclear.",
            confidence=0.5,
        ).field_path
        == field_path
    )


@pytest.mark.parametrize(
    "field_path",
    [
        "derived_chart_data.basic_info.type",
        "validation.is_valid",
        "validation.warnings[0]",
        "personality.not_a_real_field",
        "design.unknown_planet",
        "visually_active_gates.not_a_field",
        "visible_colored_channels.invalid",
    ],
)
def test_derived_or_validation_field_paths_are_rejected(field_path: str) -> None:
    with pytest.raises(ValueError, match="raw Vision"):
        UncertainItem(
            field_path=field_path,
            observed_value=None,
            reason="Not a raw Vision field.",
            confidence=0.5,
        )


def test_validation_severity_supports_only_expected_values() -> None:
    assert tuple(severity.value for severity in ValidationSeverity) == (
        "INFO",
        "WARNING",
        "ERROR",
    )

    with pytest.raises(ValueError):
        ValidationSeverity("DEBUG")


def test_validation_source_supports_only_expected_values() -> None:
    assert tuple(source.value for source in ValidationSource) == (
        "parser",
        "interpreter",
        "validation",
    )

    with pytest.raises(ValueError):
        ValidationSource("client")


def test_validation_code_supports_complete_phase2_warning_contract() -> None:
    assert tuple(code.value for code in ValidationCode) == EXPECTED_VALIDATION_CODE_VALUES
    for code_value in EXPECTED_VALIDATION_CODE_VALUES:
        assert ValidationCode(code_value).value == code_value


def test_validation_warning_exposes_stable_assertion_fields() -> None:
    warning = ValidationWarning(
        code=ValidationCode.MISSING_ACTIVATION,
        message="Missing Personality Moon activation.",
        severity=ValidationSeverity.WARNING,
        affects_validity=True,
        source=ValidationSource.validation,
    )

    assert warning.code is ValidationCode.MISSING_ACTIVATION
    assert warning.message == "Missing Personality Moon activation."
    assert warning.severity is ValidationSeverity.WARNING
    assert warning.affects_validity is True
    assert warning.source is ValidationSource.validation


def test_validation_result_is_valid_when_no_warning_affects_validity() -> None:
    validation = ValidationResult(
        warnings=(
            _warning(severity=ValidationSeverity.INFO),
            _warning(severity=ValidationSeverity.WARNING),
        )
    )

    assert validation.is_valid is True


def test_validation_result_is_invalid_when_any_warning_affects_validity() -> None:
    validation = ValidationResult(
        warnings=(
            _warning(severity=ValidationSeverity.INFO),
            _warning(affects_validity=True),
        )
    )

    assert validation.is_valid is False


def test_validation_result_does_not_allow_manual_is_valid_override() -> None:
    with pytest.raises(TypeError):
        constructor = cast(Any, ValidationResult)
        constructor(warnings=(), is_valid=False)


def test_parse_result_preserves_parser_origin_warnings() -> None:
    parser_warning = _warning(
        code=ValidationCode.VISIBLE_CHANNEL_NORMALIZED,
        source=ValidationSource.parser,
    )

    parse_result = ParseResult(raw_vision=_raw_vision(), warnings=(parser_warning,))

    assert parse_result.raw_vision == _raw_vision()
    assert parse_result.warnings == (parser_warning,)


def test_parse_result_merges_parser_and_later_warnings_into_validation() -> None:
    parser_warning = _warning(
        code=ValidationCode.VISIBLE_CHANNEL_NORMALIZED,
        source=ValidationSource.parser,
    )
    later_warning = _warning(
        code=ValidationCode.MISSING_ACTIVATION,
        source=ValidationSource.validation,
        affects_validity=True,
    )
    parse_result = ParseResult(raw_vision=_raw_vision(), warnings=(parser_warning,))

    validation = parse_result.to_validation_result((later_warning,))

    assert validation.warnings == (parser_warning, later_warning)
    assert validation.is_valid is False


def test_missing_non_sun_activation_warning_data_is_representable() -> None:
    personality = _activation_column(PersonalityActivationColumn, missing_field="moon")
    raw = _raw_vision(personality=personality)
    warning = _warning(
        code=ValidationCode.MISSING_ACTIVATION,
        affects_validity=True,
    )
    validation = ValidationResult(warnings=(warning,))

    assert raw.personality.moon is None
    assert validation.warnings[0].code is ValidationCode.MISSING_ACTIVATION
    assert validation.is_valid is False


def test_missing_personality_sun_uses_specific_warning_only() -> None:
    personality = _activation_column(PersonalityActivationColumn, missing_field="sun")
    raw = _raw_vision(personality=personality)
    validation = ValidationResult(
        warnings=(
            _warning(
                code=ValidationCode.MISSING_PERSONALITY_SUN,
                affects_validity=True,
            ),
        )
    )

    codes = tuple(warning.code for warning in validation.warnings)
    assert raw.personality.sun is None
    assert codes == (ValidationCode.MISSING_PERSONALITY_SUN,)
    assert ValidationCode.MISSING_ACTIVATION not in codes
    assert validation.is_valid is False


def test_missing_design_sun_uses_specific_warning_only() -> None:
    design = _activation_column(DesignActivationColumn, missing_field="sun")
    raw = _raw_vision(design=design)
    validation = ValidationResult(
        warnings=(
            _warning(
                code=ValidationCode.MISSING_DESIGN_SUN,
                affects_validity=True,
            ),
        )
    )

    codes = tuple(warning.code for warning in validation.warnings)
    assert raw.design.sun is None
    assert codes == (ValidationCode.MISSING_DESIGN_SUN,)
    assert ValidationCode.MISSING_ACTIVATION not in codes
    assert validation.is_valid is False


def test_bodygraph_extraction_result_represents_full_structured_output() -> None:
    result = BodyGraphExtractionResult(
        raw_vision=_raw_vision(),
        derived_chart_data=_derived_chart_data(),
        validation=ValidationResult(warnings=()),
    )

    assert result.raw_vision.personality.sun == Activation(gate=1, line=1)
    assert result.derived_chart_data.basic_info.type == "Manifesting Generator"
    assert result.validation.is_valid is True


def test_models_are_immutable() -> None:
    activation = Activation(gate=61, line=4)

    with pytest.raises(FrozenInstanceError):
        setattr(activation, "line", 5)
