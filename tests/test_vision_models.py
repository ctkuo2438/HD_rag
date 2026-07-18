import math
from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any, cast, overload

import pytest

import human_design.vision as vision
from human_design.vision import (
    Activation,
    BodyGraphExtractionResult,
    DerivedBasicInfo,
    DerivedChartData,
    DesignActivationColumn,
    ParseResult,
    PersonalityActivationColumn,
    RawVisionExtraction,
    UncertainItem,
    ValidationCode,
    ValidationResult,
    ValidationSeverity,
    ValidationSource,
    ValidationWarning,
)
from human_design.vision.constants import PLANETARY_FIELDS as PLANET_FIELDS
from human_design.vision.models import warning_defaults


PUBLIC_MODEL_NAMES = (
    "Activation",
    "PersonalityActivationColumn",
    "DesignActivationColumn",
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

EXPECTED_VALIDATION_CODE_VALUES = {
    "VISIBLE_CHANNEL_NORMALIZED",
    "INVALID_VISIBLE_CHANNEL",
    "INVALID_VISUALLY_ACTIVE_GATE",
    "INVALID_VISUAL_CENTER",
    "INVALID_UNCERTAIN_ITEM",
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
    "INCONSISTENT_DERIVED_CHART",
}


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


def _raw_vision(
    *,
    personality: PersonalityActivationColumn | None = None,
    design: DesignActivationColumn | None = None,
) -> RawVisionExtraction:
    return RawVisionExtraction(
        personality=personality or _activation_column(PersonalityActivationColumn),
        design=design or _activation_column(DesignActivationColumn),
        visually_defined_centers=("Sacral", "Throat"),
        visually_active_gates=(20, 34, 57),
        visible_colored_channels=("20-34",),
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
    field_path: str = "test.field",
) -> ValidationWarning:
    return ValidationWarning(
        code=code,
        message=f"{code.value} test warning",
        severity=severity,
        affects_validity=affects_validity,
        source=source,
        field_path=field_path,
    )


@pytest.mark.parametrize(
    "model",
    [
        Activation,
        PersonalityActivationColumn,
        DesignActivationColumn,
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


def test_official_pipeline_is_exported_from_vision_package() -> None:
    assert callable(vision.extract_bodygraph)


def test_complete_raw_vision_extraction_can_be_constructed() -> None:
    raw = _raw_vision()

    assert raw.personality.sun == Activation(gate=1, line=1)
    assert raw.design.pluto == Activation(gate=13, line=1)
    assert raw.visually_defined_centers == ("Sacral", "Throat")
    assert raw.visually_active_gates == (20, 34, 57)
    assert raw.visible_colored_channels == ("20-34",)
    assert raw.uncertain_items[0].field_path == "personality.sun"


def test_raw_vision_collections_have_safe_empty_defaults() -> None:
    raw = RawVisionExtraction(
        personality=_activation_column(PersonalityActivationColumn),
        design=_activation_column(DesignActivationColumn),
    )

    assert raw.visually_defined_centers == ()
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


@pytest.mark.parametrize(
    "column_model",
    [PersonalityActivationColumn, DesignActivationColumn],
)
def test_activation_columns_have_all_canonical_fields(column_model: type[object]) -> None:
    assert {field.name for field in fields(column_model)} == set(PLANET_FIELDS)


def test_activation_values_and_none_are_representable() -> None:
    personality = _activation_column(PersonalityActivationColumn, missing_field="moon")
    design = _activation_column(DesignActivationColumn, missing_field="venus")

    assert personality.sun == Activation(gate=1, line=1)
    assert personality.moon is None
    assert design.sun == Activation(gate=1, line=1)
    assert design.venus is None


@pytest.mark.parametrize(
    ("gate", "line"),
    [(0, 1), (65, 1), (99, 1), (60, 0), (60, 7), (60, 9)],
)
def test_out_of_range_activation_values_are_rejected(gate: int, line: int) -> None:
    # Range validity is an Activation construction invariant: the parser
    # drops out-of-range raw values, so no in-range violation can exist.
    with pytest.raises(ValueError):
        Activation(gate=gate, line=line)


@pytest.mark.parametrize(("gate", "line"), [(1, 1), (64, 6), (34, 3)])
def test_in_range_activation_values_are_accepted(gate: int, line: int) -> None:
    activation = Activation(gate=gate, line=line)

    assert activation.gate == gate
    assert activation.line == line


@pytest.mark.parametrize("bad_value", [True, False, "61", 61.0, None])
def test_activation_rejects_invalid_gate_types(bad_value: object) -> None:
    with pytest.raises(TypeError):
        Activation(gate=cast(int, bad_value), line=4)


@pytest.mark.parametrize("bad_value", [True, False, "4", 4.0, None])
def test_activation_rejects_invalid_line_types(bad_value: object) -> None:
    with pytest.raises(TypeError):
        Activation(gate=61, line=cast(int, bad_value))


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_valid_finite_uncertain_item_confidence_values_are_accepted(value: float) -> None:
    uncertain_item = UncertainItem(
        field_path="design.mars",
        observed_value=None,
        reason="Activation is not readable.",
        confidence=value,
    )

    assert uncertain_item.confidence == value


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


@pytest.mark.parametrize("observed_value", [math.nan, math.inf, -math.inf])
def test_uncertain_item_rejects_nonfinite_float_observed_values(
    observed_value: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        UncertainItem(
            field_path="personality.sun",
            observed_value=observed_value,
            reason="Synthetic nonfinite observation.",
            confidence=0.5,
        )


@pytest.mark.parametrize("observed_value", ["61.4", 61, 61.4, None])
def test_uncertain_item_accepts_finite_scalar_observed_values(
    observed_value: str | int | float | None,
) -> None:
    item = UncertainItem(
        field_path="personality.sun",
        observed_value=observed_value,
        reason="Synthetic finite observation.",
        confidence=0.5,
    )

    assert item.observed_value == observed_value


@pytest.mark.parametrize(
    "field_path",
    [
        *(f"personality.{field_name}" for field_name in PLANET_FIELDS),
        *(f"design.{field_name}" for field_name in PLANET_FIELDS),
        "visually_defined_centers",
        "visually_defined_centers[0]",
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
    assert {code.value for code in ValidationCode} == EXPECTED_VALIDATION_CODE_VALUES
    for code_value in EXPECTED_VALIDATION_CODE_VALUES:
        assert ValidationCode(code_value).value == code_value


def test_every_validation_code_has_registered_warning_defaults() -> None:
    # An unregistered new ValidationCode must fail loudly (KeyError) instead
    # of silently falling back to a default severity.
    for code in ValidationCode:
        severity, affects_validity = warning_defaults(code)
        assert isinstance(severity, ValidationSeverity)
        assert isinstance(affects_validity, bool)


def test_validation_warning_exposes_stable_assertion_fields() -> None:
    warning = ValidationWarning(
        code=ValidationCode.MISSING_ACTIVATION,
        message="Missing Personality Moon activation.",
        severity=ValidationSeverity.WARNING,
        affects_validity=True,
        source=ValidationSource.validation,
        field_path="personality.moon",
    )

    assert warning.code is ValidationCode.MISSING_ACTIVATION
    assert warning.message == "Missing Personality Moon activation."
    assert warning.severity is ValidationSeverity.WARNING
    assert warning.affects_validity is True
    assert warning.source is ValidationSource.validation
    assert warning.field_path == "personality.moon"


@pytest.mark.parametrize("bad_field_path", ["", "   "])
def test_validation_warning_requires_nonempty_field_path(bad_field_path: str) -> None:
    with pytest.raises(ValueError, match="field_path"):
        ValidationWarning(
            code=ValidationCode.MISSING_ACTIVATION,
            message="Missing activation.",
            severity=ValidationSeverity.ERROR,
            affects_validity=True,
            source=ValidationSource.validation,
            field_path=bad_field_path,
        )


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


@pytest.mark.parametrize(
    ("column_attr", "missing_field", "code"),
    [
        ("personality", "moon", ValidationCode.MISSING_ACTIVATION),
        ("personality", "sun", ValidationCode.MISSING_PERSONALITY_SUN),
        ("design", "sun", ValidationCode.MISSING_DESIGN_SUN),
    ],
)
def test_missing_activation_scenarios_are_representable(
    column_attr: str,
    missing_field: str,
    code: ValidationCode,
) -> None:
    # Unavailable activations coexist with their warning code, Sun fields use
    # their specific codes without a duplicate generic MISSING_ACTIVATION,
    # and any of these codes invalidates the result. Behavior for producing
    # these warnings lives in the parser/validation tests; this pins that the
    # models can represent every scenario.
    if column_attr == "personality":
        raw = _raw_vision(
            personality=_activation_column(
                PersonalityActivationColumn, missing_field=missing_field
            )
        )
    else:
        raw = _raw_vision(
            design=_activation_column(
                DesignActivationColumn, missing_field=missing_field
            )
        )
    validation = ValidationResult(
        warnings=(_warning(code=code, affects_validity=True),)
    )

    codes = tuple(warning.code for warning in validation.warnings)
    assert getattr(getattr(raw, column_attr), missing_field) is None
    assert codes == (code,)
    if code is not ValidationCode.MISSING_ACTIVATION:
        assert ValidationCode.MISSING_ACTIVATION not in codes
    assert validation.is_valid is False


def test_bodygraph_extraction_result_represents_full_structured_output() -> None:
    result = BodyGraphExtractionResult(
        raw_vision=_raw_vision(),
        derived_chart_data=_derived_chart_data(),
        validation_result=ValidationResult(warnings=()),
    )

    assert result.raw_vision.personality.sun == Activation(gate=1, line=1)
    assert result.derived_chart_data.basic_info.type == "Manifesting Generator"
    assert result.validation_result.is_valid is True


def test_models_are_immutable() -> None:
    activation = Activation(gate=61, line=4)

    with pytest.raises(FrozenInstanceError):
        setattr(activation, "line", 5)