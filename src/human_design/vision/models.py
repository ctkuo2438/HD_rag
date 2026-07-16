"""
Typed models for Phase 2 BodyGraph Vision extraction.

Vision API original observe
    ↓
RawVisionExtraction
    ↓
Parser preserve normalization warnings
    ↓
ParseResult
    ↓
Interpreter generate deterministic chart facts
    ↓
DerivedChartData
    ↓
ValidationResult
    ↓
BodyGraphExtractionResult

"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from typing import TypeVar


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

_RAW_VISION_COLLECTION_FIELDS = frozenset(
    (
        "visually_defined_centers",
        "visually_active_gates",
        "visible_colored_channels",
    )
)

_T = TypeVar("_T")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


@dataclass(frozen=True)
class Activation:
    """A typed Human Design Gate.Line activation value.

    Task 16 guarantees integer Gate.Line storage only. Task 18 parser behavior
    records malformed or out-of-range activation issues, and Task 20 validation
    determines final validity using structured warnings.
    """

    gate: int
    line: int

    def __post_init__(self) -> None:
        _validate_integer(self.gate, "gate")
        _validate_integer(self.line, "line")


@dataclass(frozen=True)
class PersonalityActivationColumn:
    """Typed Personality activation column with canonical planetary fields."""

    sun: Activation | None = None
    earth: Activation | None = None
    north_node: Activation | None = None
    south_node: Activation | None = None
    moon: Activation | None = None
    mercury: Activation | None = None
    venus: Activation | None = None
    mars: Activation | None = None
    jupiter: Activation | None = None
    saturn: Activation | None = None
    uranus: Activation | None = None
    neptune: Activation | None = None
    pluto: Activation | None = None

    def __post_init__(self) -> None:
        _validate_activation_column(self)


@dataclass(frozen=True)
class DesignActivationColumn:
    """Typed Design activation column with canonical planetary fields."""

    sun: Activation | None = None
    earth: Activation | None = None
    north_node: Activation | None = None
    south_node: Activation | None = None
    moon: Activation | None = None
    mercury: Activation | None = None
    venus: Activation | None = None
    mars: Activation | None = None
    jupiter: Activation | None = None
    saturn: Activation | None = None
    uranus: Activation | None = None
    neptune: Activation | None = None
    pluto: Activation | None = None

    def __post_init__(self) -> None:
        _validate_activation_column(self)


@dataclass(frozen=True)
class UncertainItem:
    """A raw Vision observation the model could not read confidently."""

    field_path: str
    observed_value: str | int | float | None
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        _validate_uncertain_field_path(self.field_path)
        _validate_observed_value(self.observed_value)
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        object.__setattr__(
            self,
            "confidence",
            _validate_confidence_value(self.confidence, "confidence"),
        )


@dataclass(frozen=True)
class RawVisionExtraction:
    """Raw visible facts extracted from a BodyGraph image."""

    personality: PersonalityActivationColumn
    design: DesignActivationColumn
    visually_defined_centers: tuple[str, ...] = field(default_factory=tuple)
    visually_active_gates: tuple[int, ...] = field(default_factory=tuple)
    visible_colored_channels: tuple[str, ...] = field(default_factory=tuple)
    uncertain_items: tuple[UncertainItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.personality, PersonalityActivationColumn):
            raise TypeError("personality must be PersonalityActivationColumn")
        if not isinstance(self.design, DesignActivationColumn):
            raise TypeError("design must be DesignActivationColumn")
        object.__setattr__(
            self,
            "visually_defined_centers",
            _typed_tuple(self.visually_defined_centers, str, "visually_defined_centers"),
        )
        object.__setattr__(
            self,
            "visually_active_gates",
            _typed_tuple(self.visually_active_gates, int, "visually_active_gates"),
        )
        object.__setattr__(
            self,
            "visible_colored_channels",
            _typed_tuple(
                self.visible_colored_channels,
                str,
                "visible_colored_channels",
            ),
        )
        object.__setattr__(
            self,
            "uncertain_items",
            _typed_tuple(self.uncertain_items, UncertainItem, "uncertain_items"),
        )


@dataclass(frozen=True)
class DerivedBasicInfo:
    """Deterministically derived basic chart information."""

    type: str
    authority: str
    profile: str
    strategy: str
    definition: str
    not_self_theme: str
    signature: str


@dataclass(frozen=True)
class DerivedChartData:
    """Deterministically derived chart facts."""

    basic_info: DerivedBasicInfo
    active_gates: tuple[int, ...]
    active_channels: tuple[str, ...]
    defined_centers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.basic_info, DerivedBasicInfo):
            raise TypeError("basic_info must be DerivedBasicInfo")

        object.__setattr__(
            self,
            "active_gates",
            _typed_tuple(self.active_gates, int, "active_gates"),
        )
        object.__setattr__(
            self,
            "active_channels",
            _typed_tuple(self.active_channels, str, "active_channels"),
        )
        object.__setattr__(
            self,
            "defined_centers",
            _typed_tuple(self.defined_centers, str, "defined_centers"),
        )


class ValidationSeverity(StrEnum):
    """Severity levels for structured validation warnings."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationSource(StrEnum):
    """Pipeline source that produced a structured warning."""

    parser = "parser"
    interpreter = "interpreter"
    validation = "validation"


class ValidationCode(StrEnum):
    """Machine-readable validation and parser warning codes."""

    VISIBLE_CHANNEL_NORMALIZED = "VISIBLE_CHANNEL_NORMALIZED"
    INVALID_VISIBLE_CHANNEL = "INVALID_VISIBLE_CHANNEL"
    INVALID_VISUALLY_ACTIVE_GATE = "INVALID_VISUALLY_ACTIVE_GATE"
    INVALID_VISUAL_CENTER = "INVALID_VISUAL_CENTER"
    VISIBLE_CHANNEL_NOT_DERIVED = "VISIBLE_CHANNEL_NOT_DERIVED"
    DERIVED_CHANNEL_NOT_VISIBLE = "DERIVED_CHANNEL_NOT_VISIBLE"
    VISUALLY_ACTIVE_GATES_MISMATCH = "VISUALLY_ACTIVE_GATES_MISMATCH"
    VISUALLY_DEFINED_CENTERS_MISMATCH = "VISUALLY_DEFINED_CENTERS_MISMATCH"
    UNSUPPORTED_AUTHORITY = "UNSUPPORTED_AUTHORITY"
    MISSING_PERSONALITY_SUN = "MISSING_PERSONALITY_SUN"
    MISSING_DESIGN_SUN = "MISSING_DESIGN_SUN"
    MISSING_ACTIVATION = "MISSING_ACTIVATION"
    MALFORMED_ACTIVATION = "MALFORMED_ACTIVATION"
    INVALID_ACTIVATION_GATE = "INVALID_ACTIVATION_GATE"
    INVALID_ACTIVATION_LINE = "INVALID_ACTIVATION_LINE"
    INCONSISTENT_DERIVED_CHART = "INCONSISTENT_DERIVED_CHART"


@dataclass(frozen=True)
class ValidationWarning:
    """A machine-readable structured warning from the extraction pipeline."""

    code: ValidationCode
    message: str
    severity: ValidationSeverity
    affects_validity: bool
    source: ValidationSource

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code",
            _coerce_enum(self.code, ValidationCode, "code"),
        )
        object.__setattr__(
            self,
            "severity",
            _coerce_enum(self.severity, ValidationSeverity, "severity"),
        )
        object.__setattr__(
            self,
            "source",
            _coerce_enum(self.source, ValidationSource, "source"),
        )

        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")
        if not isinstance(self.affects_validity, bool):
            raise TypeError("affects_validity must be a bool")


@dataclass(frozen=True)
class ValidationResult:
    """Validation output with validity derived from warning effects."""

    warnings: tuple[ValidationWarning, ...] = field(default_factory=tuple)
    is_valid: bool = field(init=False)

    def __post_init__(self) -> None:
        warnings = _typed_tuple(self.warnings, ValidationWarning, "warnings")
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "is_valid",
            not any(warning.affects_validity for warning in warnings),
        )


@dataclass(frozen=True)
class ParseResult:
    """Parser output that preserves parser-origin warnings."""

    raw_vision: RawVisionExtraction
    warnings: tuple[ValidationWarning, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.raw_vision, RawVisionExtraction):
            raise TypeError("raw_vision must be RawVisionExtraction")
        object.__setattr__(
            self,
            "warnings",
            _typed_tuple(self.warnings, ValidationWarning, "warnings"),
        )

    def to_validation_result(
        self,
        later_warnings: Iterable[ValidationWarning] = (),
    ) -> ValidationResult:
        """Merge parser warnings with later warnings into validation output."""
        later_warning_tuple = _typed_tuple(
            later_warnings,
            ValidationWarning,
            "later_warnings",
        )
        return ValidationResult(warnings=(*self.warnings, *later_warning_tuple))


@dataclass(frozen=True)
class BodyGraphExtractionResult:
    """Full structured BodyGraph extraction output."""

    raw_vision: RawVisionExtraction
    derived_chart_data: DerivedChartData
    validation: ValidationResult

    def __post_init__(self) -> None:
        if not isinstance(self.raw_vision, RawVisionExtraction):
            raise TypeError("raw_vision must be RawVisionExtraction")
        if not isinstance(self.derived_chart_data, DerivedChartData):
            raise TypeError("derived_chart_data must be DerivedChartData")
        if not isinstance(self.validation, ValidationResult):
            raise TypeError("validation must be ValidationResult")


def _validate_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")


def _validate_activation_column(
    column: PersonalityActivationColumn | DesignActivationColumn,
) -> None:
    for field_name in _PLANETARY_FIELDS:
        value = getattr(column, field_name)
        if value is not None and not isinstance(value, Activation):
            raise TypeError(f"{field_name} must be Activation or None")


def _validate_confidence_value(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{field_name} must be a finite numeric confidence from 0.0 to 1.0"
        )

    confidence = float(value)
    if not math.isfinite(confidence):
        raise ValueError(f"{field_name} must be finite")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return confidence


def _validate_uncertain_field_path(field_path: object) -> None:
    if not isinstance(field_path, str) or not field_path.strip():
        raise ValueError("field_path must be a non-empty raw Vision field path")

    if not _is_raw_vision_field_path(field_path):
        raise ValueError("field_path must point to a raw Vision field")


def _is_raw_vision_field_path(field_path: str) -> bool:
    if "." in field_path:
        container, field_name = field_path.split(".", maxsplit=1)
        return container in {"personality", "design"} and field_name in _PLANETARY_FIELDS

    if "[" in field_path or "]" in field_path:
        return _is_indexed_raw_collection_path(field_path)

    return field_path in _RAW_VISION_COLLECTION_FIELDS


def _is_indexed_raw_collection_path(field_path: str) -> bool:
    if not field_path.endswith("]") or "[" not in field_path:
        return False

    field_name, raw_index = field_path[:-1].split("[", maxsplit=1)
    return field_name in _RAW_VISION_COLLECTION_FIELDS and raw_index.isdecimal()


def _validate_observed_value(value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("observed_value must be str, int, float, or None")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("observed_value float must be finite")


def _typed_tuple(
    values: Iterable[_T],
    item_type: type[_T],
    field_name: str,
) -> tuple[_T, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of {item_type.__name__}")

    try:
        tuple_values = tuple(values)
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be an iterable of {item_type.__name__}"
        ) from exc

    for value in tuple_values:
        if isinstance(value, bool) and item_type is int:
            raise TypeError(f"{field_name} items must be {item_type.__name__}")
        if not isinstance(value, item_type):
            raise TypeError(f"{field_name} items must be {item_type.__name__}")
    return tuple_values


def _coerce_enum(
    value: object,
    enum_type: type[_EnumT],
    field_name: str,
) -> _EnumT:
    """Return a StrEnum member from a member instance or its string value."""
    if isinstance(value, enum_type):
        return value

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a valid {enum_type.__name__}")

    for member in enum_type:
        if member.value == value:
            return member

    raise ValueError(f"{field_name} must be a valid {enum_type.__name__}")


__all__ = [
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
]
