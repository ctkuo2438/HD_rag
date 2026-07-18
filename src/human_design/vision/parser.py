from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from typing import Any, Literal, NoReturn, overload

from human_design.vision.constants import (
    ALL_CHANNELS,
    CANONICAL_CENTERS,
    CENTER_ALIASES,
    PLANETARY_FIELDS,
)
from human_design.vision.models import (
    Activation,
    ParseResult,
    PersonalityActivationColumn,
    DesignActivationColumn,
    RawVisionExtraction,
    UncertainItem,
    ValidationCode,
    ValidationSource,
    ValidationWarning,
    warning_defaults,
)


class BodyGraphParseError(ValueError):
    """Raised when raw Vision JSON cannot be parsed into the raw schema."""


_REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "personality",
        "design",
        "visually_defined_centers",
        "visually_active_gates",
        "visible_colored_channels",
        "uncertain_items",
    }
)

_UNCERTAIN_ITEM_FIELDS = frozenset(
    {"field_path", "observed_value", "reason", "confidence"}
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


def _center_key(value: str) -> str:
    normalized = re.sub(r"[_-]+", " ", value.strip().casefold())
    words = normalized.split()
    if words and words[-1] in {"center", "centre"}:
        words.pop()
    return " ".join(words)


_ALL_CHANNELS_SET = frozenset(ALL_CHANNELS)
_REVERSED_CHANNELS = {
    "-".join(reversed(channel.split("-"))): channel for channel in ALL_CHANNELS
}
_CANONICAL_CENTERS_SET = frozenset(CANONICAL_CENTERS)
_CENTER_NAME_LOOKUP = {
    **{_center_key(center): center for center in CANONICAL_CENTERS},
    **{_center_key(alias): canonical for alias, canonical in CENTER_ALIASES.items()},
}


def parse_bodygraph_raw_extraction_json(raw_json: str) -> ParseResult:
    """Parse raw Vision JSON into typed raw extraction models."""
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
    raw_vision = RawVisionExtraction(
        personality=personality,
        design=design,
        visually_defined_centers=_parse_centers(
            payload["visually_defined_centers"],
            "visually_defined_centers",
            warnings,
        ),
        visually_active_gates=_parse_visible_gates(
            payload["visually_active_gates"],
            warnings,
        ),
        visible_colored_channels=_parse_visible_channels(
            payload["visible_colored_channels"],
            warnings,
        ),
        uncertain_items=_parse_uncertain_items(payload["uncertain_items"], warnings),
    )
    return ParseResult(raw_vision=raw_vision, warnings=tuple(warnings))


def _parse_json_object(raw_json: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            raw_json,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise BodyGraphParseError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise BodyGraphParseError("Raw Vision JSON must be a top-level object")
    return payload


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise BodyGraphParseError(
        f"Invalid JSON: non-standard numeric constant {value}"
    )


def _validate_top_level_schema(payload: Mapping[str, Any]) -> None:
    forbidden = _FORBIDDEN_FINAL_CONCEPT_FIELDS & payload.keys()
    if forbidden:
        raise BodyGraphParseError(
            "final Human Design concepts are not accepted from Vision output: "
            + ", ".join(sorted(forbidden))
        )

    missing = _REQUIRED_TOP_LEVEL_FIELDS - payload.keys()
    if missing:
        raise BodyGraphParseError(
            f"Missing required top-level fields: {', '.join(sorted(missing))}"
        )

    extras = payload.keys() - _REQUIRED_TOP_LEVEL_FIELDS
    if extras:
        raise BodyGraphParseError(
            f"Unexpected top-level raw Vision fields: {', '.join(sorted(extras))}"
        )


@overload
def _parse_activation_column(
    raw_column: Any,
    *,
    column_name: Literal["personality"],
    warnings: list[ValidationWarning],
) -> PersonalityActivationColumn: ...


@overload
def _parse_activation_column(
    raw_column: Any,
    *,
    column_name: Literal["design"],
    warnings: list[ValidationWarning],
) -> DesignActivationColumn: ...


def _parse_activation_column(
    raw_column: Any,
    *,
    column_name: str,
    warnings: list[ValidationWarning],
) -> PersonalityActivationColumn | DesignActivationColumn:
    if not isinstance(raw_column, dict):
        raise BodyGraphParseError(f"{column_name} must be a JSON object")
    _reject_extra_keys(
        raw_column,
        allowed_keys=PLANETARY_FIELDS,
        field_path=column_name,
    )

    parsed: dict[str, Activation | None] = {}
    for field_name in PLANETARY_FIELDS:
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

    if not 1 <= gate <= 64:
        warnings.append(_parser_warning(ValidationCode.INVALID_ACTIVATION_GATE, field_path))
        return None
    if not 1 <= line <= 6:
        warnings.append(_parser_warning(ValidationCode.INVALID_ACTIVATION_LINE, field_path))
        return None

    return Activation(gate=gate, line=line)


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
                    ValidationCode.INVALID_VISUALLY_ACTIVE_GATE,
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


def _parse_centers(
    raw_centers: Any,
    field_name: str,
    warnings: list[ValidationWarning],
) -> tuple[str, ...]:
    if not isinstance(raw_centers, list):
        raise BodyGraphParseError(f"{field_name} must be a list")

    parsed: list[str] = []
    for index, raw_center in enumerate(raw_centers):
        field_path = f"{field_name}[{index}]"
        if not isinstance(raw_center, str):
            warnings.append(
                _parser_warning(ValidationCode.INVALID_VISUAL_CENTER, field_path)
            )
            continue
        canonical = _CENTER_NAME_LOOKUP.get(_center_key(raw_center))
        if canonical is None or canonical not in _CANONICAL_CENTERS_SET:
            warnings.append(
                _parser_warning(ValidationCode.INVALID_VISUAL_CENTER, field_path)
            )
            continue
        parsed.append(canonical)
    return tuple(parsed)


def _parse_uncertain_items(
    raw_items: Any,
    warnings: list[ValidationWarning],
) -> tuple[UncertainItem, ...]:
    if not isinstance(raw_items, list):
        raise BodyGraphParseError("uncertain_items must be a list")

    # Element-level failures downgrade to warning + skip, matching the other
    # list parsers: uncertain_items is self-reported metadata, so a bad item
    # should never fail the whole parse.
    parsed: list[UncertainItem] = []
    for index, raw_item in enumerate(raw_items):
        field_path = f"uncertain_items[{index}]"
        if not isinstance(raw_item, dict) or raw_item.keys() != _UNCERTAIN_ITEM_FIELDS:
            warnings.append(
                _parser_warning(ValidationCode.INVALID_UNCERTAIN_ITEM, field_path)
            )
            continue
        try:
            parsed.append(UncertainItem(**raw_item))
        except (TypeError, ValueError):
            warnings.append(
                _parser_warning(ValidationCode.INVALID_UNCERTAIN_ITEM, field_path)
            )
    return tuple(parsed)


def _parser_warning(code: ValidationCode, field_path: str) -> ValidationWarning:
    severity, affects_validity = warning_defaults(code)
    return ValidationWarning(
        code=code,
        message=f"{code.value} at {field_path}",
        severity=severity,
        affects_validity=affects_validity,
        source=ValidationSource.parser,
        field_path=field_path,
    )


def _reject_extra_keys(
    value: Mapping[str, Any],
    *,
    allowed_keys: Collection[str],
    field_path: str,
) -> None:
    extra_keys = set(value) - set(allowed_keys)
    if extra_keys:
        raise BodyGraphParseError(
            f"Unexpected fields at {field_path}: {', '.join(sorted(extra_keys))}"
        )


__all__ = [
    "BodyGraphParseError",
    "parse_bodygraph_raw_extraction_json",
]