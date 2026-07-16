"""Validation checks for parsed and interpreted BodyGraph extraction results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields

from human_design.vision.constants import ALL_CHANNELS, CANONICAL_CENTERS
from human_design.vision.interpreter import BodyGraphInterpretationResult
from human_design.vision.models import (
    Activation,
    DerivedChartData,
    ParseResult,
    RawVisionExtraction,
    ValidationCode,
    ValidationResult,
    ValidationSeverity,
    ValidationSource,
    ValidationWarning,
)


_ALL_CHANNELS_SET = frozenset(ALL_CHANNELS)
_CANONICAL_CENTERS_SET = frozenset(CANONICAL_CENTERS)
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
    """Validate raw facts against deterministic derived chart data."""
    merged_warnings: list[ValidationWarning] = [
        *parser_warnings,
        *interpreter_warnings,
    ]
    validation_warnings: list[ValidationWarning] = []

    validation_warnings.extend(_activation_warnings(raw_vision, parser_warnings))
    validation_warnings.extend(
        _visible_channel_warnings(raw_vision, derived_chart_data)
    )
    validation_warnings.extend(_visual_gate_warnings(raw_vision, derived_chart_data))
    validation_warnings.extend(_visual_center_warnings(raw_vision, derived_chart_data))
    validation_warnings.extend(_reflector_consistency_warnings(derived_chart_data))
    validation_warnings.extend(
        _unsupported_authority_warnings(derived_chart_data, merged_warnings)
    )

    for warning in validation_warnings:
        _append_warning_if_new(merged_warnings, warning)

    return ValidationResult(warnings=tuple(merged_warnings))


def _activation_warnings(
    raw_vision: RawVisionExtraction,
    parser_warnings: tuple[ValidationWarning, ...],
) -> tuple[ValidationWarning, ...]:
    warnings: list[ValidationWarning] = []
    for column_name, column in (
        ("personality", raw_vision.personality),
        ("design", raw_vision.design),
    ):
        for field in fields(column):
            field_name = field.name
            field_path = f"{column_name}.{field_name}"
            activation = getattr(column, field_name)

            if activation is None:
                if _parser_warned_for_activation_field(parser_warnings, field_path):
                    continue
                warnings.append(
                    _validation_warning(
                        _missing_activation_code(column_name, field_name),
                        field_path,
                    )
                )
                continue

            warnings.extend(
                _invalid_activation_warnings(
                    activation,
                    field_path,
                    parser_warnings,
                )
            )

    return tuple(warnings)


def _missing_activation_code(column_name: str, field_name: str) -> ValidationCode:
    if column_name == "personality" and field_name == "sun":
        return ValidationCode.MISSING_PERSONALITY_SUN
    if column_name == "design" and field_name == "sun":
        return ValidationCode.MISSING_DESIGN_SUN
    return ValidationCode.MISSING_ACTIVATION


def _invalid_activation_warnings(
    activation: Activation,
    field_path: str,
    parser_warnings: tuple[ValidationWarning, ...],
) -> tuple[ValidationWarning, ...]:
    warnings: list[ValidationWarning] = []
    if not 1 <= activation.gate <= 64 and not _parser_warned_for_activation_code(
        parser_warnings,
        field_path,
        ValidationCode.INVALID_ACTIVATION_GATE,
    ):
        warnings.append(
            _validation_warning(ValidationCode.INVALID_ACTIVATION_GATE, field_path)
        )
    if not 1 <= activation.line <= 6 and not _parser_warned_for_activation_code(
        parser_warnings,
        field_path,
        ValidationCode.INVALID_ACTIVATION_LINE,
    ):
        warnings.append(
            _validation_warning(ValidationCode.INVALID_ACTIVATION_LINE, field_path)
        )
    return tuple(warnings)


def _parser_warned_for_activation_field(
    parser_warnings: tuple[ValidationWarning, ...],
    field_path: str,
) -> bool:
    return any(
        warning.source is ValidationSource.parser
        and warning.code in _ACTIVATION_WARNING_CODES
        and _warning_mentions_field_path(warning, field_path)
        for warning in parser_warnings
    )


def _parser_warned_for_activation_code(
    parser_warnings: tuple[ValidationWarning, ...],
    field_path: str,
    code: ValidationCode,
) -> bool:
    return any(
        warning.source is ValidationSource.parser
        and warning.code is code
        and _warning_mentions_field_path(warning, field_path)
        for warning in parser_warnings
    )


def _warning_mentions_field_path(
    warning: ValidationWarning,
    field_path: str,
) -> bool:
    return f" at {field_path}" in warning.message


def _visible_channel_warnings(
    raw_vision: RawVisionExtraction,
    derived_chart_data: DerivedChartData,
) -> tuple[ValidationWarning, ...]:
    warnings: list[ValidationWarning] = []
    visible_channels = raw_vision.visible_colored_channels
    derived_channels = set(derived_chart_data.active_channels)

    for index, channel in enumerate(visible_channels):
        field_path = f"visible_colored_channels[{index}]"
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
    visual_centers = _center_set(raw_vision.visually_defined_centers)
    derived_centers = _center_set(derived_chart_data.defined_centers)
    if visual_centers == derived_centers:
        return ()
    return (
        _validation_warning(
            ValidationCode.VISUALLY_DEFINED_CENTERS_MISMATCH,
            "visually_defined_centers",
        ),
    )


def _center_set(centers: Iterable[str]) -> set[str]:
    center_tuple = tuple(centers)
    return {
        *(center for center in CANONICAL_CENTERS if center in center_tuple),
        *(center for center in center_tuple if center not in _CANONICAL_CENTERS_SET),
    }


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
    severity, affects_validity = _warning_defaults(code)
    return ValidationWarning(
        code=code,
        message=f"{code.value} at {field_path}",
        severity=severity,
        affects_validity=affects_validity,
        source=ValidationSource.validation,
    )


def _warning_defaults(code: ValidationCode) -> tuple[ValidationSeverity, bool]:
    if code is ValidationCode.VISIBLE_CHANNEL_NORMALIZED:
        return ValidationSeverity.INFO, False
    if code in {
        ValidationCode.INVALID_VISIBLE_CHANNEL,
        ValidationCode.INVALID_VISUALLY_ACTIVE_GATE,
        ValidationCode.INVALID_VISUAL_CENTER,
        ValidationCode.VISIBLE_CHANNEL_NOT_DERIVED,
        ValidationCode.DERIVED_CHANNEL_NOT_VISIBLE,
        ValidationCode.VISUALLY_ACTIVE_GATES_MISMATCH,
        ValidationCode.VISUALLY_DEFINED_CENTERS_MISMATCH,
        ValidationCode.UNSUPPORTED_AUTHORITY,
    }:
        return ValidationSeverity.WARNING, False
    return ValidationSeverity.ERROR, True


def _append_warning_if_new(
    warnings: list[ValidationWarning],
    warning: ValidationWarning,
) -> None:
    warning_key = _warning_key(warning)
    if warning_key not in {_warning_key(existing) for existing in warnings}:
        warnings.append(warning)


def _warning_key(
    warning: ValidationWarning,
) -> tuple[ValidationCode, str]:
    return warning.code, warning.message


__all__ = [
    "validate_bodygraph_extraction",
    "validate_bodygraph_components",
]
