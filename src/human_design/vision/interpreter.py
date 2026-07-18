"""Deterministic BodyGraph interpretation from parsed raw Vision facts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from human_design.vision.constants import (
    ALL_CHANNELS,
    CANONICAL_CENTERS,
    CHANNEL_TO_CENTERS,
    MOTOR_CENTERS,
    PLANETARY_FIELDS,
)
from human_design.vision.models import (
    Activation,
    DerivedBasicInfo,
    DerivedChartData,
    RawVisionExtraction,
    ValidationCode,
    ValidationSource,
    ValidationWarning,
    warning_defaults,
)


_TYPE_TO_LIFE_FIELDS: dict[str, tuple[str, str, str]] = {
    "Generator": ("To Respond", "Frustration", "Satisfaction"),
    "Manifesting Generator": ("To Respond", "Frustration", "Satisfaction"),
    "Projector": ("Wait for the Invitation", "Bitterness", "Success"),
    "Manifestor": ("To Inform", "Anger", "Peace"),
    "Reflector": ("Wait a Lunar Cycle", "Disappointment", "Surprise"),
}

_SELF_PROJECTED_CHANNELS = frozenset({"1-8", "7-31", "10-20", "13-33"})

# Adjacency map between defined centers, built from active channels.
_CenterGraph = dict[str, set[str]]


@dataclass(frozen=True)
class BodyGraphInterpretationResult:
    """Typed output from deterministic BodyGraph interpretation."""

    derived_chart_data: DerivedChartData
    warnings: tuple[ValidationWarning, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.derived_chart_data, DerivedChartData):
            raise TypeError("derived_chart_data must be DerivedChartData")

        warnings = tuple(self.warnings)
        for warning in warnings:
            if not isinstance(warning, ValidationWarning):
                raise TypeError("warnings items must be ValidationWarning")
        object.__setattr__(self, "warnings", warnings)


def interpret_bodygraph(raw_vision: RawVisionExtraction) -> BodyGraphInterpretationResult:
    """Derive chart data from already-normalized raw Vision extraction facts."""
    active_gates = derive_active_gates(raw_vision)
    active_channels = derive_active_channels(active_gates)
    defined_centers = derive_defined_centers(active_channels)
    center_graph = _build_center_graph(defined_centers, active_channels)

    chart_type = _derive_type(defined_centers, center_graph)
    authority, warnings = _derive_authority(
        chart_type,
        defined_centers,
        active_channels,
    )
    strategy, not_self_theme, signature = _TYPE_TO_LIFE_FIELDS[chart_type]

    basic_info = DerivedBasicInfo(
        type=chart_type,
        authority=authority,
        profile=_derive_profile(raw_vision),
        strategy=strategy,
        definition=_derive_definition(center_graph),
        not_self_theme=not_self_theme,
        signature=signature,
    )
    derived_chart_data = DerivedChartData(
        basic_info=basic_info,
        active_gates=active_gates,
        active_channels=active_channels,
        defined_centers=defined_centers,
    )
    return BodyGraphInterpretationResult(
        derived_chart_data=derived_chart_data,
        warnings=warnings,
    )


def derive_active_gates(raw_vision: RawVisionExtraction) -> tuple[int, ...]:
    """Return sorted unique gate numbers from activation columns only.

    Range validity is guaranteed by ``Activation`` at construction time.
    """
    return tuple(sorted({activation.gate for activation in _iter_activations(raw_vision)}))


def derive_active_channels(active_gates: Iterable[int]) -> tuple[str, ...]:
    """Return active canonical channels in ``ALL_CHANNELS`` order."""
    active_gate_set = set(active_gates)
    active_channels: list[str] = []
    for channel in ALL_CHANNELS:
        gate_a, gate_b = _channel_gate_pair(channel)
        if gate_a in active_gate_set and gate_b in active_gate_set:
            active_channels.append(channel)
    return tuple(active_channels)


def derive_defined_centers(active_channels: Iterable[str]) -> tuple[str, ...]:
    """Return canonical-order centers defined by active channel endpoints."""
    defined_centers: set[str] = set()
    for channel in active_channels:
        defined_centers.update(CHANNEL_TO_CENTERS[channel])
    return tuple(center for center in CANONICAL_CENTERS if center in defined_centers)


def _iter_activations(raw_vision: RawVisionExtraction) -> Iterable[Activation]:
    for column in (raw_vision.personality, raw_vision.design):
        for field_name in PLANETARY_FIELDS:
            activation = getattr(column, field_name)
            if activation is not None:
                yield activation


def _channel_gate_pair(channel: str) -> tuple[int, int]:
    gate_a, gate_b = channel.split("-", maxsplit=1)
    return int(gate_a), int(gate_b)


def _build_center_graph(
    defined_centers: tuple[str, ...],
    active_channels: tuple[str, ...],
) -> _CenterGraph:
    """Build the adjacency map between defined centers from active channels."""
    graph: _CenterGraph = {center: set() for center in defined_centers}
    for channel in active_channels:
        center_a, center_b = CHANNEL_TO_CENTERS[channel]
        if center_a in graph and center_b in graph:
            graph[center_a].add(center_b)
            graph[center_b].add(center_a)
    return graph


def _derive_profile(raw_vision: RawVisionExtraction) -> str:
    personality_sun = raw_vision.personality.sun
    design_sun = raw_vision.design.sun
    if personality_sun is None or design_sun is None:
        return "Unknown"
    return f"{personality_sun.line}/{design_sun.line}"


def _derive_definition(center_graph: _CenterGraph) -> str:
    if not center_graph:
        return "No Definition"

    component_count = _count_connected_components(center_graph)
    if component_count == 1:
        return "Single Definition"
    if component_count == 2:
        return "Split Definition"
    if component_count == 3:
        return "Triple Split Definition"
    return "Quadruple Split Definition"


def _count_connected_components(graph: _CenterGraph) -> int:
    visited: set[str] = set()
    component_count = 0

    for center in graph:
        if center in visited:
            continue
        component_count += 1
        stack = [center]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(graph[current] - visited)

    return component_count


def _derive_type(
    defined_centers: tuple[str, ...],
    center_graph: _CenterGraph,
) -> str:
    if not defined_centers:
        return "Reflector"

    # Standard rule: a motor connected to the Throat (directly OR through
    # intermediate centers) is what distinguishes MG/Manifestor charts.
    motor_to_throat = _throat_connected_to_motor(center_graph)
    if "Sacral" in center_graph:
        return "Manifesting Generator" if motor_to_throat else "Generator"
    return "Manifestor" if motor_to_throat else "Projector"


def _throat_connected_to_motor(center_graph: _CenterGraph) -> bool:
    """Return True if any motor center reaches the Throat through the graph."""
    if "Throat" not in center_graph:
        return False

    visited: set[str] = set()
    stack = ["Throat"]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in MOTOR_CENTERS:
            return True
        stack.extend(center_graph[current] - visited)
    return False


def _derive_authority(
    chart_type: str,
    defined_centers: tuple[str, ...],
    active_channels: tuple[str, ...],
) -> tuple[str, tuple[ValidationWarning, ...]]:
    defined_center_set = set(defined_centers)
    if chart_type == "Reflector":
        return "Lunar", ()
    if "Solar Plexus" in defined_center_set:
        return "Emotional", ()
    if "Sacral" in defined_center_set:
        return "Sacral", ()
    if "Spleen" in defined_center_set:
        return "Splenic", ()
    if chart_type == "Manifestor" and "Ego" in defined_center_set:
        return "Ego", ()
    if chart_type == "Projector" and "25-51" in active_channels:
        return "Ego-Projected", ()
    if chart_type == "Projector" and _SELF_PROJECTED_CHANNELS.intersection(
        active_channels
    ):
        return "Self-Projected", ()
    return "Needs Review", (_unsupported_authority_warning(),)


def _unsupported_authority_warning() -> ValidationWarning:
    severity, affects_validity = warning_defaults(ValidationCode.UNSUPPORTED_AUTHORITY)
    return ValidationWarning(
        code=ValidationCode.UNSUPPORTED_AUTHORITY,
        message="Authority could not be derived by Phase 2 v1 rules.",
        severity=severity,
        affects_validity=affects_validity,
        source=ValidationSource.interpreter,
        field_path="derived_chart_data.basic_info.authority",
    )


__all__ = [
    "BodyGraphInterpretationResult",
    "interpret_bodygraph",
    "derive_active_gates",
    "derive_active_channels",
    "derive_defined_centers",
]