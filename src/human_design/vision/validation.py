"""Validation checks for parsed and interpreted BodyGraph extraction results."""

from __future__ import annotations

from collections.abc import Iterable

from human_design.vision.constants import ALL_CHANNELS, PLANETARY_FIELDS
from human_design.vision.interpreter import BodyGraphInterpretationResult
from human_design.vision.models import (
    DerivedChartData,
    ParseResult,
    RawVisionExtraction,
    ValidationCode,
    ValidationResult,
    ValidationSource,
    ValidationWarning,
    warning_defaults,
)


_ALL_CHANNELS_SET = frozenset(ALL_CHANNELS)

# Parser codes that already account for an activation field being unusable.
# When the parser warned for a field, validation must not double-report it.
_ACTIVATION_WARNING_CODES = frozenset(
    {
        ValidationCode.MISSING_PERSONALITY_SUN,
        ValidationCode.MISSING_DESIGN_SUN,
        ValidationCode.MISSING_ACTIVATION,
        ValidationCode.MALFORMED_ACTIVATION,
        ValidationCode.INVALID_ACTIVATION_GATE,
        ValidationCode.INVALID_ACTIVATION_LINE,
    }
)


def validate_bodygraph_extraction(
    *,
    parse_result: ParseResult,
    interpretation_result: BodyGraphInterpretationResult,
) -> ValidationResult:
    """Validate parser and interpreter outputs as one pipeline step."""
    return validate_bodygraph_components(
        raw_vision=parse_result.raw_vision,
        derived_chart_data=interpretation_result.derived_chart_data,
        parser_warnings=parse_result.warnings,
        interpreter_warnings=interpretation_result.warnings,
    )


def validate_bodygraph_components(
    *,
    raw_vision: RawVisionExtraction,
    derived_chart_data: DerivedChartData,
    parser_warnings: tuple[ValidationWarning, ...] = (),
    interpreter_warnings: tuple[ValidationWarning, ...] = (),
) -> ValidationResult:
    """Validate raw facts against deterministic derived chart data.

    Completeness policy: every activation column field must hold a value.
    A single missing planet (JSON null) makes the whole result invalid,
    because active_gates derive from all 26 activations; one missing planet
    can silently drop a gate, a channel, and flip type/authority. The parser
    treats null as an honest empty value; completeness is judged here.
    """
    merged_warnings: list[ValidationWarning] = [
        *parser_warnings,
        *interpreter_warnings,
    ]

    validation_warnings: list[ValidationWarning] = [
        *_activation_warnings(raw_vision, parser_warnings),
        *_visible_channel_warnings(raw_vision, derived_chart_data),
        *_visual_gate_warnings(raw_vision, derived_chart_data),
        *_visual_center_warnings(raw_vision, derived_chart_data),
        *_reflector_consistency_warnings(derived_chart_data),
        *_unsupported_authority_warnings(derived_chart_data, merged_warnings),
    ]

    seen = {_warning_key(warning) for warning in merged_warnings}
    for warning in validation_warnings:
        key = _warning_key(warning)
        if key not in seen:
            seen.add(key)
            merged_warnings.append(warning)

    return ValidationResult(warnings=tuple(merged_warnings))


def _activation_warnings(
    raw_vision: RawVisionExtraction,
    parser_warnings: tuple[ValidationWarning, ...],
) -> tuple[ValidationWarning, ...]:
    # An existing Activation is in-range by construction (models enforces it),
    # so the only thing left to check here is completeness.
    warnings: list[ValidationWarning] = []
    for column_name, column in (
        ("personality", raw_vision.personality),
        ("design", raw_vision.design),
    ):
        for field_name in PLANETARY_FIELDS:
            if getattr(column, field_name) is not None:
                continue
            field_path = f"{column_name}.{field_name}"
            if _parser_warned_for_activation_field(parser_warnings, field_path):
                continue
            warnings.append(
                _validation_warning(
                    _missing_activation_code(column_name, field_name),
                    field_path,
                )
            )
    return tuple(warnings)


def _missing_activation_code(column_name: str, field_name: str) -> ValidationCode:
    if column_name == "personality" and field_name == "sun":
        return ValidationCode.MISSING_PERSONALITY_SUN
    if column_name == "design" and field_name == "sun":
        return ValidationCode.MISSING_DESIGN_SUN
    return ValidationCode.MISSING_ACTIVATION


def _parser_warned_for_activation_field(
    parser_warnings: tuple[ValidationWarning, ...],
    field_path: str,
) -> bool:
    return any(
        warning.source is ValidationSource.parser
        and warning.code in _ACTIVATION_WARNING_CODES
        and warning.field_path == field_path
        for warning in parser_warnings
    )


def _visible_channel_warnings(
    raw_vision: RawVisionExtraction,
    derived_chart_data: DerivedChartData,
) -> tuple[ValidationWarning, ...]:
    warnings: list[ValidationWarning] = []
    visible_channels = raw_vision.visible_colored_channels
    derived_channels = set(derived_chart_data.active_channels)

    for index, channel in enumerate(visible_channels):
        field_path = f"visible_colored_channels[{index}]"
        # Reachable via the public components API: RawVisionExtraction does
        # not constrain channel strings, only the parser does.
        if channel not in _ALL_CHANNELS_SET:
            warnings.append(
                _validation_warning(ValidationCode.INVALID_VISIBLE_CHANNEL, field_path)
            )
            continue
        if channel not in derived_channels:
            warnings.append(
                _validation_warning(
                    ValidationCode.VISIBLE_CHANNEL_NOT_DERIVED,
                    field_path,
                )
            )

    visible_channel_set = set(visible_channels)
    for index, channel in enumerate(derived_chart_data.active_channels):
        if channel not in visible_channel_set:
            warnings.append(
                _validation_warning(
                    ValidationCode.DERIVED_CHANNEL_NOT_VISIBLE,
                    f"active_channels[{index}]",
                )
            )

    return tuple(warnings)


def _visual_gate_warnings(
    raw_vision: RawVisionExtraction,
    derived_chart_data: DerivedChartData,
) -> tuple[ValidationWarning, ...]:
    if set(raw_vision.visually_active_gates) == set(derived_chart_data.active_gates):
        return ()
    return (
        _validation_warning(
            ValidationCode.VISUALLY_ACTIVE_GATES_MISMATCH,
            "visually_active_gates",
        ),
    )


def _visual_center_warnings(
    raw_vision: RawVisionExtraction,
    derived_chart_data: DerivedChartData,
) -> tuple[ValidationWarning, ...]:
    if set(raw_vision.visually_defined_centers) == set(derived_chart_data.defined_centers):
        return ()
    return (
        _validation_warning(
            ValidationCode.VISUALLY_DEFINED_CENTERS_MISMATCH,
            "visually_defined_centers",
        ),
    )


def _reflector_consistency_warnings(
    derived_chart_data: DerivedChartData,
) -> tuple[ValidationWarning, ...]:
    if derived_chart_data.basic_info.type != "Reflector":
        return ()
    if (
        not derived_chart_data.active_channels
        and not derived_chart_data.defined_centers
        and derived_chart_data.basic_info.definition == "No Definition"
        and derived_chart_data.basic_info.authority == "Lunar"
    ):
        return ()
    return (
        _validation_warning(
            ValidationCode.INCONSISTENT_DERIVED_CHART,
            "derived_chart_data",
        ),
    )


def _unsupported_authority_warnings(
    derived_chart_data: DerivedChartData,
    existing_warnings: Iterable[ValidationWarning],
) -> tuple[ValidationWarning, ...]:
    if derived_chart_data.basic_info.authority not in {"Needs Review", "Unknown"}:
        return ()
    if any(
        warning.code is ValidationCode.UNSUPPORTED_AUTHORITY
        for warning in existing_warnings
    ):
        return ()
    return (
        _validation_warning(
            ValidationCode.UNSUPPORTED_AUTHORITY,
            "derived_chart_data.basic_info.authority",
        ),
    )


def _validation_warning(
    code: ValidationCode,
    field_path: str,
) -> ValidationWarning:
    severity, affects_validity = warning_defaults(code)
    return ValidationWarning(
        code=code,
        message=f"{code.value} at {field_path}",
        severity=severity,
        affects_validity=affects_validity,
        source=ValidationSource.validation,
        field_path=field_path,
    )


def _warning_key(warning: ValidationWarning) -> tuple[ValidationCode, str]:
    return warning.code, warning.field_path


__all__ = [
    "validate_bodygraph_extraction",
    "validate_bodygraph_components",
]