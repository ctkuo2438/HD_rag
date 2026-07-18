import json
from dataclasses import replace

import pytest

from human_design.vision.interpreter import (
    BodyGraphInterpretationResult,
    interpret_bodygraph,
)
from human_design.vision.models import (
    Activation,
    DerivedBasicInfo,
    DerivedChartData,
    DesignActivationColumn,
    ParseResult,
    PersonalityActivationColumn,
    RawVisionExtraction,
    ValidationCode,
    ValidationResult,
    ValidationSeverity,
    ValidationSource,
    ValidationWarning,
    warning_defaults,
)
from human_design.vision.parser import parse_bodygraph_raw_extraction_json
from human_design.vision.validation import (
    validate_bodygraph_components,
    validate_bodygraph_extraction,
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


def _activation(gate: int, line: int = 1) -> Activation:
    return Activation(gate=gate, line=line)


def _activation_values_for_gates(gates: tuple[int, ...]) -> dict[str, Activation]:
    return {
        field_name: _activation(gates[index % len(gates)], line=(index % 6) + 1)
        for index, field_name in enumerate(PLANET_FIELDS)
    }


def _raw_with_gates(
    gates: tuple[int, ...],
    *,
    personality_overrides: dict[str, Activation | None] | None = None,
    design_overrides: dict[str, Activation | None] | None = None,
    visually_defined_centers: tuple[str, ...] | None = None,
    visually_active_gates: tuple[int, ...] | None = None,
    visible_colored_channels: tuple[str, ...] | None = None,
) -> RawVisionExtraction:
    personality_values: dict[str, Activation | None] = _activation_values_for_gates(
        gates
    )
    design_values: dict[str, Activation | None] = _activation_values_for_gates(gates)
    personality_values.update(personality_overrides or {})
    design_values.update(design_overrides or {})

    raw_vision = RawVisionExtraction(
        personality=PersonalityActivationColumn(**personality_values),
        design=DesignActivationColumn(**design_values),
        visually_defined_centers=visually_defined_centers or (),
        visually_active_gates=visually_active_gates or (),
        visible_colored_channels=visible_colored_channels or (),
    )
    chart = interpret_bodygraph(raw_vision).derived_chart_data
    return replace(
        raw_vision,
        visually_defined_centers=(
            chart.defined_centers
            if visually_defined_centers is None
            else visually_defined_centers
        ),
        visually_active_gates=(
            chart.active_gates
            if visually_active_gates is None
            else visually_active_gates
        ),
        visible_colored_channels=(
            chart.active_channels
            if visible_colored_channels is None
            else visible_colored_channels
        ),
    )


def _valid_generator_raw(**overrides: object) -> RawVisionExtraction:
    defaults = {
        "visually_defined_centers": ("Sacral", "Root"),
        "visually_active_gates": (3, 60),
        "visible_colored_channels": ("3-60",),
    }
    defaults.update(overrides)
    return _raw_with_gates((3, 60), **defaults)


def _valid_reflector_raw(**overrides: object) -> RawVisionExtraction:
    return _raw_with_gates((1,), **overrides)


def _validation_for_raw(
    raw_vision: RawVisionExtraction,
    *,
    parser_warnings: tuple[ValidationWarning, ...] = (),
    interpreter_warnings_override: tuple[ValidationWarning, ...] | None = None,
) -> ValidationResult:
    interpretation = interpret_bodygraph(raw_vision)
    interpreter_warnings = (
        interpretation.warnings
        if interpreter_warnings_override is None
        else interpreter_warnings_override
    )
    return validate_bodygraph_components(
        raw_vision=raw_vision,
        derived_chart_data=interpretation.derived_chart_data,
        parser_warnings=parser_warnings,
        interpreter_warnings=interpreter_warnings,
    )


def _warning(
    code: ValidationCode,
    *,
    source: ValidationSource = ValidationSource.parser,
    severity: ValidationSeverity | None = None,
    affects_validity: bool | None = None,
    field_path: str = "test.field",
) -> ValidationWarning:
    default_severity, default_affects_validity = warning_defaults(code)
    return ValidationWarning(
        code=code,
        message=f"{code.value} at {field_path}",
        severity=severity or default_severity,
        affects_validity=(
            default_affects_validity
            if affects_validity is None
            else affects_validity
        ),
        source=source,
        field_path=field_path,
    )


def _warnings_by_code(
    result: ValidationResult,
    code: ValidationCode,
) -> tuple[ValidationWarning, ...]:
    return tuple(warning for warning in result.warnings if warning.code is code)


def _warning_codes(result: ValidationResult) -> tuple[ValidationCode, ...]:
    return tuple(warning.code for warning in result.warnings)


def _assert_warning_metadata(
    warning: ValidationWarning,
    *,
    code: ValidationCode,
    severity: ValidationSeverity,
    affects_validity: bool,
    source: ValidationSource,
) -> None:
    assert warning.code is code
    assert warning.severity is severity
    assert warning.affects_validity is affects_validity
    assert warning.source is source


def _parser_payload(
    *,
    visible_colored_channels: list[str],
    gates: tuple[int, ...] = (10, 57),
) -> dict[str, object]:
    activations = {
        field_name: f"{gates[index % len(gates)]}.{(index % 6) + 1}"
        for index, field_name in enumerate(PLANET_FIELDS)
    }
    return {
        "personality": activations,
        "design": activations,
        "visually_defined_centers": ["G", "Spleen"],
        "visually_active_gates": list(gates),
        "visible_colored_channels": visible_colored_channels,
        "uncertain_items": [],
    }


def _mock_raw_json_payload(
    *,
    personality: dict[str, str | None],
    design: dict[str, str | None],
    visually_defined_centers: list[str] | None = None,
    visually_active_gates: list[int | str] | None = None,
    visible_colored_channels: list[str] | None = None,
) -> dict[str, object]:
    return {
        "personality": personality,
        "design": design,
        "visually_defined_centers": visually_defined_centers or [],
        "visually_active_gates": visually_active_gates or [],
        "visible_colored_channels": visible_colored_channels or [],
        "uncertain_items": [],
    }


def _activation_payload_from_gates(gates: tuple[int, ...]) -> dict[str, str]:
    return {
        field_name: f"{gates[index % len(gates)]}.{(index % 6) + 1}"
        for index, field_name in enumerate(PLANET_FIELDS)
    }


def _generator_pipeline_personality() -> dict[str, str]:
    return {
        **_activation_payload_from_gates((61, 3, 10, 34, 60, 32)),
        "sun": "61.4",
    }


def _generator_pipeline_design() -> dict[str, str]:
    return {
        **_activation_payload_from_gates((32, 60, 3, 34, 10, 61)),
        "sun": "32.6",
    }


def _run_pipeline(
    payload: dict[str, object],
) -> tuple[ParseResult, BodyGraphInterpretationResult, ValidationResult]:
    parse_result = parse_bodygraph_raw_extraction_json(json.dumps(payload))
    interpretation_result = interpret_bodygraph(parse_result.raw_vision)
    validation_result = validate_bodygraph_extraction(
        parse_result=parse_result,
        interpretation_result=interpretation_result,
    )
    return parse_result, interpretation_result, validation_result


def test_public_api_returns_validation_result() -> None:

    raw = _valid_generator_raw()
    interpretation = interpret_bodygraph(raw)
    result = validate_bodygraph_extraction(
        parse_result=ParseResult(raw_vision=raw),
        interpretation_result=interpretation,
    )

    assert isinstance(result, ValidationResult)


def test_valid_generator_has_no_validation_warnings() -> None:
    result = _validation_for_raw(_valid_generator_raw())

    assert result.warnings == ()
    assert result.is_valid is True


def test_valid_reflector_like_chart_has_no_validation_warnings() -> None:
    raw = _valid_reflector_raw()
    interpretation = interpret_bodygraph(raw)

    result = validate_bodygraph_components(
        raw_vision=raw,
        derived_chart_data=interpretation.derived_chart_data,
        interpreter_warnings=interpretation.warnings,
    )

    assert interpretation.derived_chart_data.active_channels == ()
    assert interpretation.derived_chart_data.defined_centers == ()
    assert interpretation.derived_chart_data.basic_info.type == "Reflector"
    assert interpretation.derived_chart_data.basic_info.authority == "Lunar"
    assert result.warnings == ()
    assert result.is_valid is True


def test_parser_warnings_are_preserved_in_order_with_metadata() -> None:
    raw = _valid_generator_raw()
    parser_warnings = (
        _warning(
            ValidationCode.VISIBLE_CHANNEL_NORMALIZED,
            field_path="visible_colored_channels[0]",
        ),
        _warning(
            ValidationCode.INVALID_VISIBLE_CHANNEL,
            field_path="visible_colored_channels[1]",
        ),
    )

    result = _validation_for_raw(raw, parser_warnings=parser_warnings)

    assert result.warnings[:2] == parser_warnings
    assert result.is_valid is True
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.VISIBLE_CHANNEL_NORMALIZED,
        severity=ValidationSeverity.INFO,
        affects_validity=False,
        source=ValidationSource.parser,
    )
    _assert_warning_metadata(
        result.warnings[1],
        code=ValidationCode.INVALID_VISIBLE_CHANNEL,
        severity=ValidationSeverity.WARNING,
        affects_validity=False,
        source=ValidationSource.parser,
    )


def test_malformed_activation_parser_warning_is_preserved_and_invalidates() -> None:
    parser_warning = _warning(
        ValidationCode.MALFORMED_ACTIVATION,
        field_path="personality.moon",
    )

    result = _validation_for_raw(
        _valid_generator_raw(),
        parser_warnings=(parser_warning,),
    )

    assert result.warnings[0] == parser_warning
    assert result.is_valid is False


def test_interpreter_warning_is_preserved_and_does_not_invalidate() -> None:
    raw = _raw_with_gates((24, 61))
    interpretation = interpret_bodygraph(raw)

    result = validate_bodygraph_extraction(
        parse_result=ParseResult(raw_vision=raw),
        interpretation_result=interpretation,
    )

    assert interpretation.warnings[0].code is ValidationCode.UNSUPPORTED_AUTHORITY
    assert result.warnings == interpretation.warnings
    assert result.is_valid is True


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        (
            {"personality_overrides": {"sun": None}},
            ValidationCode.MISSING_PERSONALITY_SUN,
        ),
        (
            {"design_overrides": {"sun": None}},
            ValidationCode.MISSING_DESIGN_SUN,
        ),
        (
            {"personality_overrides": {"moon": None}},
            ValidationCode.MISSING_ACTIVATION,
        ),
    ],
)
def test_missing_activation_validation_warnings_invalidate(
    overrides: dict[str, dict[str, Activation | None]],
    expected_code: ValidationCode,
) -> None:
    raw = _raw_with_gates((3, 60), **overrides)

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (expected_code,)
    warning = result.warnings[0]
    _assert_warning_metadata(
        warning,
        code=expected_code,
        severity=ValidationSeverity.ERROR,
        affects_validity=True,
        source=ValidationSource.validation,
    )
    assert result.is_valid is False
    if expected_code is ValidationCode.MISSING_PERSONALITY_SUN:
        assert ValidationCode.MISSING_ACTIVATION not in _warning_codes(result)
    if expected_code is ValidationCode.MISSING_DESIGN_SUN:
        assert ValidationCode.MISSING_ACTIVATION not in _warning_codes(result)


def test_parser_activation_warnings_are_not_duplicated_by_validation() -> None:
    raw = _raw_with_gates((3, 60), personality_overrides={"moon": None})
    parser_warning = _warning(
        ValidationCode.MALFORMED_ACTIVATION,
        field_path="personality.moon",
    )

    result = _validation_for_raw(raw, parser_warnings=(parser_warning,))

    assert _warning_codes(result) == (ValidationCode.MALFORMED_ACTIVATION,)
    assert result.is_valid is False


def test_invalid_visible_channel_warns_without_invalidating_or_deriving_channel() -> None:
    raw = _valid_reflector_raw(visible_colored_channels=("34-99",))

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (ValidationCode.INVALID_VISIBLE_CHANNEL,)
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.INVALID_VISIBLE_CHANNEL,
        severity=ValidationSeverity.WARNING,
        affects_validity=False,
        source=ValidationSource.validation,
    )
    assert interpret_bodygraph(raw).derived_chart_data.active_channels == ()
    assert result.is_valid is True


def test_reversed_valid_visible_channel_parser_warning_is_preserved() -> None:
    parse_result = parse_bodygraph_raw_extraction_json(
        json.dumps(_parser_payload(visible_colored_channels=["57-10"]))
    )
    interpretation = interpret_bodygraph(parse_result.raw_vision)

    result = validate_bodygraph_extraction(
        parse_result=parse_result,
        interpretation_result=interpretation,
    )

    assert parse_result.raw_vision.visible_colored_channels == ("10-57",)
    assert _warning_codes(result) == (ValidationCode.VISIBLE_CHANNEL_NORMALIZED,)
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.VISIBLE_CHANNEL_NORMALIZED,
        severity=ValidationSeverity.INFO,
        affects_validity=False,
        source=ValidationSource.parser,
    )
    assert result.is_valid is True


def test_visible_channel_not_derived_warns_without_invalidating() -> None:
    raw = _valid_reflector_raw(visible_colored_channels=("3-60",))

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (ValidationCode.VISIBLE_CHANNEL_NOT_DERIVED,)
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.VISIBLE_CHANNEL_NOT_DERIVED,
        severity=ValidationSeverity.WARNING,
        affects_validity=False,
        source=ValidationSource.validation,
    )
    assert result.is_valid is True


def test_derived_channel_not_visible_warns_without_invalidating() -> None:
    raw = _raw_with_gates((3, 60, 10, 57), visible_colored_channels=("3-60",))

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (ValidationCode.DERIVED_CHANNEL_NOT_VISIBLE,)
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.DERIVED_CHANNEL_NOT_VISIBLE,
        severity=ValidationSeverity.WARNING,
        affects_validity=False,
        source=ValidationSource.validation,
    )
    assert result.is_valid is True


def test_colored_channel_mismatch_uses_visual_and_derived_warning_codes() -> None:
    raw = _valid_generator_raw(visible_colored_channels=("10-57",))

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (
        ValidationCode.VISIBLE_CHANNEL_NOT_DERIVED,
        ValidationCode.DERIVED_CHANNEL_NOT_VISIBLE,
    )
    assert all(warning.affects_validity is False for warning in result.warnings)
    assert result.is_valid is True


def test_empty_visible_channels_warn_for_missing_derived_channels() -> None:
    raw = _valid_generator_raw(visible_colored_channels=())

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (ValidationCode.DERIVED_CHANNEL_NOT_VISIBLE,)
    assert result.warnings[0].affects_validity is False
    assert result.is_valid is True


def test_visually_active_gates_mismatch_warns_without_invalidating() -> None:
    raw = _valid_generator_raw(visually_active_gates=(3,))

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (ValidationCode.VISUALLY_ACTIVE_GATES_MISMATCH,)
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.VISUALLY_ACTIVE_GATES_MISMATCH,
        severity=ValidationSeverity.WARNING,
        affects_validity=False,
        source=ValidationSource.validation,
    )
    assert result.is_valid is True


def test_empty_visually_active_gates_warn_when_derived_gates_are_nonempty() -> None:
    raw = _valid_generator_raw(visually_active_gates=())

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (ValidationCode.VISUALLY_ACTIVE_GATES_MISMATCH,)
    assert result.warnings[0].affects_validity is False
    assert result.is_valid is True


def test_visually_defined_centers_mismatch_warns_without_invalidating() -> None:
    raw = _valid_generator_raw(visually_defined_centers=("Throat",))

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (
        ValidationCode.VISUALLY_DEFINED_CENTERS_MISMATCH,
    )
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.VISUALLY_DEFINED_CENTERS_MISMATCH,
        severity=ValidationSeverity.WARNING,
        affects_validity=False,
        source=ValidationSource.validation,
    )
    assert result.is_valid is True


def test_empty_visually_defined_centers_warn_when_derived_centers_are_nonempty() -> None:
    raw = _valid_generator_raw(visually_defined_centers=())

    result = _validation_for_raw(raw)

    assert _warning_codes(result) == (
        ValidationCode.VISUALLY_DEFINED_CENTERS_MISMATCH,
    )
    assert result.warnings[0].affects_validity is False
    assert result.is_valid is True


def test_empty_visual_and_derived_sets_match_without_warnings() -> None:
    raw = _raw_with_gates(
        (1,),
        visually_defined_centers=(),
        visually_active_gates=(),
        visible_colored_channels=(),
    )
    interpreted = interpret_bodygraph(raw).derived_chart_data
    empty_derived_chart = DerivedChartData(
        basic_info=interpreted.basic_info,
        active_gates=(),
        active_channels=(),
        defined_centers=(),
    )

    result = validate_bodygraph_components(
        raw_vision=raw,
        derived_chart_data=empty_derived_chart,
    )

    assert result.warnings == ()
    assert result.is_valid is True


def test_invalid_visual_gate_does_not_invalidate_complete_pipeline() -> None:
    payload = _mock_raw_json_payload(
        personality=_activation_payload_from_gates((3, 60)),
        design=_activation_payload_from_gates((3, 60)),
        visually_defined_centers=["Sacral", "Root"],
        visually_active_gates=[3, 60, 99],
        visible_colored_channels=["3-60"],
    )

    parse_result, _, validation_result = _run_pipeline(payload)

    assert parse_result.raw_vision.visually_active_gates == (3, 60)
    assert _warning_codes(validation_result) == (
        ValidationCode.INVALID_VISUALLY_ACTIVE_GATE,
    )
    warning = validation_result.warnings[0]
    assert warning.source is ValidationSource.parser
    assert warning.severity is ValidationSeverity.WARNING
    assert warning.affects_validity is False
    assert validation_result.is_valid is True


def test_reflector_with_inconsistent_derived_data_is_fatal() -> None:
    raw = _valid_reflector_raw(
        visually_defined_centers=("Sacral", "Root"),
        visible_colored_channels=("3-60",),
    )
    interpretation = interpret_bodygraph(raw)
    inconsistent_basic_info = DerivedBasicInfo(
        type="Reflector",
        authority="Lunar",
        profile=interpretation.derived_chart_data.basic_info.profile,
        strategy="Wait a Lunar Cycle",
        definition="No Definition",
        not_self_theme="Disappointment",
        signature="Surprise",
    )
    inconsistent_chart = DerivedChartData(
        basic_info=inconsistent_basic_info,
        active_gates=interpretation.derived_chart_data.active_gates,
        active_channels=("3-60",),
        defined_centers=("Sacral", "Root"),
    )

    result = validate_bodygraph_components(
        raw_vision=raw,
        derived_chart_data=inconsistent_chart,
    )

    assert _warning_codes(result) == (
        ValidationCode.INCONSISTENT_DERIVED_CHART,
    )
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.INCONSISTENT_DERIVED_CHART,
        severity=ValidationSeverity.ERROR,
        affects_validity=True,
        source=ValidationSource.validation,
    )
    assert result.is_valid is False


@pytest.mark.parametrize(
    ("active_channels", "defined_centers", "definition", "authority"),
    [
        (("3-60",), (), "No Definition", "Lunar"),
        ((), ("Root",), "No Definition", "Lunar"),
        ((), (), "Single Definition", "Lunar"),
        ((), (), "No Definition", "Emotional"),
    ],
)
def test_each_reflector_invariant_violation_is_one_fatal_warning(
    active_channels: tuple[str, ...],
    defined_centers: tuple[str, ...],
    definition: str,
    authority: str,
) -> None:
    raw = _valid_reflector_raw(
        visually_defined_centers=defined_centers,
        visible_colored_channels=active_channels,
    )
    interpreted = interpret_bodygraph(raw).derived_chart_data
    chart = DerivedChartData(
        basic_info=replace(
            interpreted.basic_info,
            type="Reflector",
            definition=definition,
            authority=authority,
        ),
        active_gates=interpreted.active_gates,
        active_channels=active_channels,
        defined_centers=defined_centers,
    )

    result = validate_bodygraph_components(
        raw_vision=raw,
        derived_chart_data=chart,
    )

    assert _warning_codes(result) == (ValidationCode.INCONSISTENT_DERIVED_CHART,)
    assert result.warnings[0].severity is ValidationSeverity.ERROR
    assert result.warnings[0].affects_validity is True
    assert result.warnings[0].source is ValidationSource.validation
    assert result.is_valid is False


def test_invalid_visual_center_warning_does_not_invalidate_complete_pipeline() -> None:
    payload = _mock_raw_json_payload(
        personality=_activation_payload_from_gates((3, 60)),
        design=_activation_payload_from_gates((3, 60)),
        visually_defined_centers=["Sacral", "mystery_center", "Root"],
        visually_active_gates=[3, 60],
        visible_colored_channels=["3-60"],
    )

    parse_result, _, validation_result = _run_pipeline(payload)

    assert parse_result.raw_vision.visually_defined_centers == ("Sacral", "Root")
    assert _warning_codes(validation_result) == (
        ValidationCode.INVALID_VISUAL_CENTER,
    )
    _assert_warning_metadata(
        validation_result.warnings[0],
        code=ValidationCode.INVALID_VISUAL_CENTER,
        severity=ValidationSeverity.WARNING,
        affects_validity=False,
        source=ValidationSource.parser,
    )
    assert validation_result.is_valid is True


def test_needs_review_authority_without_interpreter_warning_gets_validation_warning() -> None:
    raw = _raw_with_gates((24, 61))
    interpretation = interpret_bodygraph(raw)

    result = _validation_for_raw(
        raw,
        interpreter_warnings_override=(),
    )

    assert interpretation.derived_chart_data.basic_info.authority == "Needs Review"
    assert _warning_codes(result) == (ValidationCode.UNSUPPORTED_AUTHORITY,)
    _assert_warning_metadata(
        result.warnings[0],
        code=ValidationCode.UNSUPPORTED_AUTHORITY,
        severity=ValidationSeverity.WARNING,
        affects_validity=False,
        source=ValidationSource.validation,
    )
    assert result.is_valid is True


def test_existing_unsupported_authority_warning_is_not_duplicated() -> None:
    raw = _raw_with_gates((24, 61))
    interpretation = interpret_bodygraph(raw)

    result = validate_bodygraph_components(
        raw_vision=raw,
        derived_chart_data=interpretation.derived_chart_data,
        interpreter_warnings=interpretation.warnings,
    )

    assert len(_warnings_by_code(result, ValidationCode.UNSUPPORTED_AUTHORITY)) == 1
    assert result.warnings == interpretation.warnings
    assert result.warnings[0].source is ValidationSource.interpreter
    assert result.is_valid is True


def test_mocked_generator_pipeline_derives_expected_chart_data() -> None:
    payload = _mock_raw_json_payload(
        personality=_generator_pipeline_personality(),
        design=_generator_pipeline_design(),
        visually_defined_centers=["G", "Sacral", "Root"],
        visually_active_gates=[3, 10, 32, 34, 60, 61],
        visible_colored_channels=["3-60", "10-34"],
    )

    parse_result, interpretation_result, validation_result = _run_pipeline(payload)
    chart = interpretation_result.derived_chart_data
    basic_info = chart.basic_info

    assert isinstance(parse_result.raw_vision, RawVisionExtraction)
    assert isinstance(chart, DerivedChartData)
    assert basic_info.profile == "4/6"
    assert {3, 10, 32, 34, 60, 61}.issubset(chart.active_gates)
    assert "3-60" in chart.active_channels
    assert "10-34" in chart.active_channels
    assert "20-34" not in chart.active_channels
    assert {"G", "Sacral", "Root"}.issubset(chart.defined_centers)
    assert basic_info.type == "Generator"
    assert basic_info.authority == "Sacral"
    assert basic_info.strategy == "To Respond"
    assert basic_info.not_self_theme == "Frustration"
    assert basic_info.signature == "Satisfaction"
    assert validation_result.warnings == ()
    assert validation_result.is_valid is True


def test_mocked_reflector_pipeline_derives_reflector_and_lunar_authority() -> None:
    reflector_personality = _activation_payload_from_gates((1,))
    reflector_design = _activation_payload_from_gates((1,))
    payload = _mock_raw_json_payload(
        personality=reflector_personality,
        design=reflector_design,
        visually_active_gates=[1],
    )

    parse_result, interpretation_result, validation_result = _run_pipeline(payload)
    chart = interpretation_result.derived_chart_data
    basic_info = chart.basic_info

    assert isinstance(parse_result.raw_vision, RawVisionExtraction)
    assert chart.active_channels == ()
    assert chart.defined_centers == ()
    assert basic_info.type == "Reflector"
    assert basic_info.authority == "Lunar"
    assert basic_info.definition == "No Definition"
    assert basic_info.strategy == "Wait a Lunar Cycle"
    assert basic_info.not_self_theme == "Disappointment"
    assert basic_info.signature == "Surprise"
    assert validation_result.warnings == ()
    assert validation_result.is_valid is True


def test_mocked_pipeline_visual_disagreements_warn_without_invalidating() -> None:
    payload = _mock_raw_json_payload(
        personality=_activation_payload_from_gates((3, 60)),
        design=_activation_payload_from_gates((3, 60)),
        visually_defined_centers=["Throat"],
        visually_active_gates=[3],
        visible_colored_channels=["10-57"],
    )

    _, interpretation_result, validation_result = _run_pipeline(payload)

    assert interpretation_result.derived_chart_data.active_channels == ("3-60",)
    assert validation_result.is_valid is True
    assert set(_warning_codes(validation_result)) == {
        ValidationCode.VISIBLE_CHANNEL_NOT_DERIVED,
        ValidationCode.DERIVED_CHANNEL_NOT_VISIBLE,
        ValidationCode.VISUALLY_ACTIVE_GATES_MISMATCH,
        ValidationCode.VISUALLY_DEFINED_CENTERS_MISMATCH,
    }
    for warning in validation_result.warnings:
        _assert_warning_metadata(
            warning,
            code=warning.code,
            severity=ValidationSeverity.WARNING,
            affects_validity=False,
            source=ValidationSource.validation,
        )


def test_mocked_pipeline_preserves_normalized_visible_channel_parser_warning() -> None:
    payload = _mock_raw_json_payload(
        personality=_activation_payload_from_gates((10, 57)),
        design=_activation_payload_from_gates((10, 57)),
        visually_defined_centers=["G", "Spleen"],
        visually_active_gates=[10, 57],
        visible_colored_channels=["57-10"],
    )

    parse_result, interpretation_result, validation_result = _run_pipeline(payload)

    assert parse_result.raw_vision.visible_colored_channels == ("10-57",)
    assert interpretation_result.derived_chart_data.active_channels == ("10-57",)
    assert _warning_codes(validation_result) == (
        ValidationCode.VISIBLE_CHANNEL_NORMALIZED,
    )
    _assert_warning_metadata(
        validation_result.warnings[0],
        code=ValidationCode.VISIBLE_CHANNEL_NORMALIZED,
        severity=ValidationSeverity.INFO,
        affects_validity=False,
        source=ValidationSource.parser,
    )
    assert ValidationCode.VISIBLE_CHANNEL_NOT_DERIVED not in _warning_codes(
        validation_result
    )
    assert validation_result.is_valid is True


def test_mocked_pipeline_malformed_activation_parser_warning_invalidates() -> None:
    personality = {
        **_generator_pipeline_personality(),
        "moon": "61.x",
    }
    payload = _mock_raw_json_payload(
        personality=personality,
        design=_generator_pipeline_design(),
        visually_defined_centers=["G", "Sacral", "Root"],
        visually_active_gates=[3, 10, 32, 34, 60, 61],
        visible_colored_channels=["3-60", "10-34"],
    )

    parse_result, interpretation_result, validation_result = _run_pipeline(payload)

    assert isinstance(parse_result.raw_vision, RawVisionExtraction)
    assert isinstance(interpretation_result.derived_chart_data, DerivedChartData)
    assert validation_result.is_valid is False
    assert _warning_codes(validation_result) == (ValidationCode.MALFORMED_ACTIVATION,)
    _assert_warning_metadata(
        validation_result.warnings[0],
        code=ValidationCode.MALFORMED_ACTIVATION,
        severity=ValidationSeverity.ERROR,
        affects_validity=True,
        source=ValidationSource.parser,
    )