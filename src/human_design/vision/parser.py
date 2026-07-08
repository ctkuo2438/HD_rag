"""Strict offline parser for mocked BodyGraph raw Vision JSON."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from numbers import Real
from typing import Any

from human_design.vision.constants import (
    ALL_CHANNELS,
    CANONICAL_CENTERS,
    CENTER_ALIASES,
)
from human_design.vision.models import (
    Activation,
    ActivationConfidenceColumn,
    ParseResult,
    PersonalityActivationColumn,
    DesignActivationColumn,
    RawVisionConfidence,
    RawVisionExtraction,
    UncertainItem,
    ValidationCode,
    ValidationSeverity,
    ValidationSource,
    ValidationWarning,
)


class BodyGraphParseError(ValueError):
    """Raised when raw Vision JSON cannot be parsed into the raw schema."""


_PLANETARY_FIELDS = (
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

_REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "personality",
        "design",
        "visually_defined_centers",
        "visually_undefined_centers",
        "visually_active_gates",
        "visible_colored_channels",
        "confidence",
        "uncertain_items",
    }
)

_FORBIDDEN_FINAL_CONCEPT_FIELDS = frozenset(
    {
        "basic_info",
        "derived_chart_data",
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
    }
)

_CHANNEL_PATTERN = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_ACTIVATION_PATTERN = re.compile(r"^(\d+)\.(\d+)$")
_ALL_CHANNELS_SET = frozenset(ALL_CHANNELS)
_REVERSED_CHANNELS = {
    "-".join(reversed(channel.split("-"))): channel for channel in ALL_CHANNELS
}
_CANONICAL_CENTERS_SET = frozenset(CANONICAL_CENTERS)


def parse_bodygraph_raw_extraction_json(raw_json: str) -> ParseResult:
    """Parse mocked raw Vision JSON into typed raw extraction models."""
    payload = _parse_json_object(raw_json)
    _validate_top_level_schema(payload)

    warnings: list[ValidationWarning] = []
    personality = _parse_activation_column(
        payload["personality"],
        column_name="personality",
        warnings=warnings,
    )
    design = _parse_activation_column(
        payload["design"],
        column_name="design",
        warnings=warnings,
    )
    confidence = _parse_confidence(payload["confidence"])
    raw_vision = RawVisionExtraction(
        personality=personality,
        design=design,
        visually_defined_centers=_parse_centers(
            payload["visually_defined_centers"],
            "visually_defined_centers",
        ),
        visually_undefined_centers=_parse_centers(
            payload["visually_undefined_centers"],
            "visually_undefined_centers",
        ),
        visually_active_gates=_parse_visible_gates(
            payload["visually_active_gates"],
            warnings,
        ),
        visible_colored_channels=_parse_visible_channels(
            payload["visible_colored_channels"],
            warnings,
        ),
        confidence=confidence,
        uncertain_items=_parse_uncertain_items(payload["uncertain_items"]),
    )
    return ParseResult(raw_vision=raw_vision, warnings=tuple(warnings))


def _parse_json_object(raw_json: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise BodyGraphParseError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise BodyGraphParseError("Raw Vision JSON must be a top-level object")
    return payload


def _validate_top_level_schema(payload: Mapping[str, Any]) -> None:
    for field_name in sorted(_FORBIDDEN_FINAL_CONCEPT_FIELDS & payload.keys()):
        raise BodyGraphParseError(
            f"final Human Design concepts are not accepted from Vision output: "
            f"{field_name}"
        )

    missing = _REQUIRED_TOP_LEVEL_FIELDS - payload.keys()
    if missing:
        missing_field = sorted(missing)[0]
        raise BodyGraphParseError(f"Missing required top-level field: {missing_field}")

    extras = set(payload) - _REQUIRED_TOP_LEVEL_FIELDS
    if extras:
        extra_field = sorted(extras)[0]
        raise BodyGraphParseError(f"Unexpected top-level raw Vision field: {extra_field}")


def _parse_activation_column(
    raw_column: Any,
    *,
    column_name: str,
    warnings: list[ValidationWarning],
) -> PersonalityActivationColumn | DesignActivationColumn:
    if not isinstance(raw_column, dict):
        raise BodyGraphParseError(f"{column_name} must be a JSON object")

    parsed: dict[str, Activation | None] = {}
    for field_name in _PLANETARY_FIELDS:
        field_path = f"{column_name}.{field_name}"
        if field_name not in raw_column:
            parsed[field_name] = None
            warnings.append(_missing_activation_warning(column_name, field_name))
            continue
        parsed[field_name] = _parse_activation_value(
            raw_column[field_name],
            field_path,
            column_name=column_name,
            field_name=field_name,
            warnings=warnings,
        )

    if column_name == "personality":
        return PersonalityActivationColumn(**parsed)
    return DesignActivationColumn(**parsed)


def _parse_activation_value(
    value: Any,
    field_path: str,
    *,
    column_name: str,
    field_name: str,
    warnings: list[ValidationWarning],
) -> Activation | None:
    if value is None:
        return None

    if not isinstance(value, str):
        warnings.append(_parser_warning(ValidationCode.MALFORMED_ACTIVATION, field_path))
        return None

    stripped = value.strip()
    if not stripped:
        warnings.append(_missing_activation_warning(column_name, field_name))
        return None

    match = _ACTIVATION_PATTERN.fullmatch(stripped)
    if match is None:
        warnings.append(_parser_warning(ValidationCode.MALFORMED_ACTIVATION, field_path))
        return None

    gate = int(match.group(1))
    line = int(match.group(2))
    activation = Activation(gate=gate, line=line)
    if not 1 <= gate <= 64:
        warnings.append(_parser_warning(ValidationCode.INVALID_ACTIVATION_GATE, field_path))
    if not 1 <= line <= 6:
        warnings.append(_parser_warning(ValidationCode.INVALID_ACTIVATION_LINE, field_path))
    return activation


def _missing_activation_warning(
    column_name: str,
    field_name: str,
) -> ValidationWarning:
    if column_name == "personality" and field_name == "sun":
        code = ValidationCode.MISSING_PERSONALITY_SUN
    elif column_name == "design" and field_name == "sun":
        code = ValidationCode.MISSING_DESIGN_SUN
    else:
        code = ValidationCode.MISSING_ACTIVATION
    return _parser_warning(code, f"{column_name}.{field_name}")


def _parse_visible_channels(
    raw_channels: Any,
    warnings: list[ValidationWarning],
) -> tuple[str, ...]:
    if not isinstance(raw_channels, list):
        raise BodyGraphParseError("visible_colored_channels must be a list")

    parsed: list[str] = []
    for index, raw_channel in enumerate(raw_channels):
        field_path = f"visible_colored_channels[{index}]"
        if not isinstance(raw_channel, str):
            warnings.append(_parser_warning(ValidationCode.INVALID_VISIBLE_CHANNEL, field_path))
            continue

        match = _CHANNEL_PATTERN.fullmatch(raw_channel.strip())
        if match is None:
            warnings.append(_parser_warning(ValidationCode.INVALID_VISIBLE_CHANNEL, field_path))
            continue

        channel = f"{int(match.group(1))}-{int(match.group(2))}"
        if channel in _ALL_CHANNELS_SET:
            parsed.append(channel)
            continue

        canonical = _REVERSED_CHANNELS.get(channel)
        if canonical is not None:
            parsed.append(canonical)
            warnings.append(
                _parser_warning(ValidationCode.VISIBLE_CHANNEL_NORMALIZED, field_path)
            )
            continue

        warnings.append(_parser_warning(ValidationCode.INVALID_VISIBLE_CHANNEL, field_path))

    return tuple(parsed)


def _parse_visible_gates(
    raw_gates: Any,
    warnings: list[ValidationWarning],
) -> tuple[int, ...]:
    if not isinstance(raw_gates, list):
        raise BodyGraphParseError("visually_active_gates must be a list")

    parsed: list[int] = []
    for index, raw_gate in enumerate(raw_gates):
        gate = _coerce_visible_gate(raw_gate)
        if gate is None or not 1 <= gate <= 64:
            warnings.append(
                _parser_warning(
                    ValidationCode.INVALID_ACTIVATION_GATE,
                    f"visually_active_gates[{index}]",
                )
            )
            continue
        parsed.append(gate)
    return tuple(parsed)


def _coerce_visible_gate(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _parse_centers(raw_centers: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(raw_centers, list):
        raise BodyGraphParseError(f"{field_name} must be a list")

    parsed: list[str] = []
    for index, raw_center in enumerate(raw_centers):
        if not isinstance(raw_center, str):
            raise BodyGraphParseError(f"{field_name}[{index}] must be a center string")
        center = raw_center.strip()
        canonical = CENTER_ALIASES.get(center, center)
        if canonical not in _CANONICAL_CENTERS_SET:
            raise BodyGraphParseError(f"Unknown center in {field_name}: {raw_center!r}")
        parsed.append(canonical)
    return tuple(parsed)


def _parse_confidence(raw_confidence: Any) -> RawVisionConfidence:
    if not isinstance(raw_confidence, dict):
        raise BodyGraphParseError("confidence must be a JSON object")

    personality = _parse_confidence_column(
        raw_confidence.get("personality"),
        "confidence.personality",
    )
    design = _parse_confidence_column(
        raw_confidence.get("design"),
        "confidence.design",
    )
    return RawVisionConfidence(
        personality=personality,
        design=design,
        visually_defined_centers=_parse_confidence_value(
            raw_confidence.get("visually_defined_centers"),
            "confidence.visually_defined_centers",
        ),
        visually_undefined_centers=_parse_confidence_value(
            raw_confidence.get("visually_undefined_centers"),
            "confidence.visually_undefined_centers",
        ),
        visually_active_gates=_parse_confidence_value(
            raw_confidence.get("visually_active_gates"),
            "confidence.visually_active_gates",
        ),
        visible_colored_channels=_parse_confidence_value(
            raw_confidence.get("visible_colored_channels"),
            "confidence.visible_colored_channels",
        ),
    )


def _parse_confidence_column(raw_column: Any, field_path: str) -> ActivationConfidenceColumn:
    if not isinstance(raw_column, dict):
        raise BodyGraphParseError(f"{field_path} must be a JSON object")

    values = {
        field_name: _parse_confidence_value(
            raw_column.get(field_name),
            f"{field_path}.{field_name}",
        )
        for field_name in _PLANETARY_FIELDS
    }
    return ActivationConfidenceColumn(**values)


def _parse_confidence_value(value: Any, field_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BodyGraphParseError(f"{field_path} confidence must be a JSON number")

    confidence = float(value)
    if not math.isfinite(confidence):
        raise BodyGraphParseError(f"{field_path} confidence must be finite")
    if not 0.0 <= confidence <= 1.0:
        raise BodyGraphParseError(
            f"{field_path} confidence must be between 0.0 and 1.0"
        )
    return confidence


def _parse_uncertain_items(raw_items: Any) -> tuple[UncertainItem, ...]:
    if not isinstance(raw_items, list):
        raise BodyGraphParseError("uncertain_items must be a list")

    parsed: list[UncertainItem] = []
    required_fields = {"field_path", "observed_value", "reason", "confidence"}
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise BodyGraphParseError(f"uncertain_items[{index}] must be an object")
        missing = required_fields - raw_item.keys()
        if missing:
            raise BodyGraphParseError(
                f"uncertain_items[{index}] missing required field: "
                f"{sorted(missing)[0]}"
            )
        try:
            parsed.append(
                UncertainItem(
                    field_path=raw_item["field_path"],
                    observed_value=raw_item["observed_value"],
                    reason=raw_item["reason"],
                    confidence=raw_item["confidence"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise BodyGraphParseError(f"Invalid uncertain_items[{index}]: {exc}") from exc
    return tuple(parsed)


def _parser_warning(code: ValidationCode, field_path: str) -> ValidationWarning:
    severity, affects_validity = _warning_defaults(code)
    return ValidationWarning(
        code=code,
        message=f"{code.value} at {field_path}",
        severity=severity,
        affects_validity=affects_validity,
        source=ValidationSource.parser,
    )


def _warning_defaults(code: ValidationCode) -> tuple[ValidationSeverity, bool]:
    if code is ValidationCode.VISIBLE_CHANNEL_NORMALIZED:
        return ValidationSeverity.INFO, False
    if code is ValidationCode.INVALID_VISIBLE_CHANNEL:
        return ValidationSeverity.WARNING, False
    return ValidationSeverity.ERROR, True


__all__ = [
    "BodyGraphParseError",
    "parse_bodygraph_raw_extraction_json",
]
