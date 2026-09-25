"""Project validated Phase 2 facts without raw visual evidence or private metadata."""

from human_design.reading.models import ChartContext, ChartFact
from human_design.vision.constants import ALL_CHANNELS, CANONICAL_CENTERS, PLANETARY_FIELDS
from human_design.vision.models import BodyGraphExtractionResult


class InvalidChartError(ValueError):
    """Phase 2 validation prevents use of this chart for personalized retrieval."""


def build_chart_context(bodygraph_result: BodyGraphExtractionResult) -> ChartContext:
    if not isinstance(bodygraph_result, BodyGraphExtractionResult):
        raise TypeError("bodygraph_result must be BodyGraphExtractionResult")
    if not bodygraph_result.validation_result.is_valid:
        raise InvalidChartError("Chart failed Phase 2 validation; personalized context is unavailable")
    derived = bodygraph_result.derived_chart_data
    facts = [ChartFact(name, getattr(derived.basic_info, name), f"derived_chart_data.basic_info.{name}")
             for name in ("type", "authority", "profile", "strategy", "definition", "not_self_theme", "signature")]
    facts.extend(ChartFact("active_gate", gate, f"derived_chart_data.active_gates[gate={gate}]")
                 for gate in sorted(set(derived.active_gates)))
    facts.extend(ChartFact("active_channel", channel, f"derived_chart_data.active_channels[channel={channel}]")
                 for channel in ALL_CHANNELS if channel in derived.active_channels)
    for defined in (True, False):
        for center in CANONICAL_CENTERS:
            if (center in derived.defined_centers) == defined:
                selector = "center" if defined else "complement_center"
                facts.append(ChartFact("defined_center" if defined else "undefined_center", center,
                    f"derived_chart_data.defined_centers[{selector}={center}]"))
    for column in ("personality", "design"):
        for planet in PLANETARY_FIELDS:
            activation = getattr(getattr(bodygraph_result.raw_vision, column), planet)
            if activation is not None:
                value = f"{column.title()} {planet.replace('_', ' ').title()} {activation.gate}.{activation.line}"
                facts.append(ChartFact(f"{column}_activation", value, f"raw_vision.{column}.{planet}"))
    # Provider/parser messages may contain observed private data; codes suffice here.
    warnings = tuple(sorted({warning.code.value for warning in bodygraph_result.validation_result.warnings}))
    return ChartContext(tuple(facts), warnings)
