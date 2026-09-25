"""Safe projection of real, offline Phase 2 parser/interpreter output."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from human_design.reading.chart_context import InvalidChartError, build_chart_context
from human_design.vision.constants import ALL_CHANNELS, CANONICAL_CENTERS, PLANETARY_FIELDS
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.models import BodyGraphExtractionResult, UncertainItem, ValidationResult
from human_design.vision.parser import parse_bodygraph_raw_extraction_json
from human_design.vision.validation import validate_bodygraph_extraction


def result_from_payload(payload):
    parsed = parse_bodygraph_raw_extraction_json(json.dumps(payload))
    interpreted = interpret_bodygraph(parsed.raw_vision)
    return BodyGraphExtractionResult(parsed.raw_vision, interpreted.derived_chart_data,
        validate_bodygraph_extraction(parse_result=parsed, interpretation_result=interpreted))


@pytest.fixture
def payload():
    return json.loads((Path(__file__).parent / "fixtures/bodygraph/test1_raw_response.json").read_text())


@pytest.mark.parametrize("field", ["type", "authority", "profile", "strategy", "definition", "not_self_theme", "signature"])
def test_basic_facts_preserve_exact_scalar_provenance(payload, field):
    result = result_from_payload(payload)
    assert result.validation_result.is_valid
    context = build_chart_context(result)
    fact = next(f for f in context.facts if f.field == field)
    assert fact.value == getattr(result.derived_chart_data.basic_info, field)
    assert fact.source_path == f"derived_chart_data.basic_info.{field}"
    assert build_chart_context(result) == context


def test_individual_facts_follow_canonical_order_even_if_input_lists_do_not(payload):
    result = result_from_payload(payload)
    derived = result.derived_chart_data
    result = replace(result, derived_chart_data=replace(derived,
        active_gates=tuple(reversed(derived.active_gates)),
        active_channels=tuple(reversed(derived.active_channels)),
        defined_centers=tuple(reversed(derived.defined_centers))))
    facts = build_chart_context(result).facts
    for field, expected, root, selector in (
        ("active_gate", sorted(derived.active_gates), "active_gates", "gate"),
        ("active_channel", [c for c in ALL_CHANNELS if c in derived.active_channels], "active_channels", "channel"),
        ("defined_center", [c for c in CANONICAL_CENTERS if c in derived.defined_centers], "defined_centers", "center"),
        ("undefined_center", [c for c in CANONICAL_CENTERS if c not in derived.defined_centers], "defined_centers", "complement_center"),
    ):
        selected = [f for f in facts if f.field == field]
        assert [f.value for f in selected] == expected
        assert [f.source_path for f in selected] == [f"derived_chart_data.{root}[{selector}={v}]" for v in expected]


def test_each_activation_has_column_planet_gate_line_and_exact_path(payload):
    result = result_from_payload(payload)
    context = build_chart_context(result)
    for column in ("personality", "design"):
        facts = [f for f in context.facts if f.field == f"{column}_activation"]
        assert [f.source_path for f in facts] == [f"raw_vision.{column}.{p}" for p in PLANETARY_FIELDS]
        for fact, planet in zip(facts, PLANETARY_FIELDS, strict=True):
            activation = getattr(getattr(result.raw_vision, column), planet)
            assert fact.value == f"{column.title()} {planet.replace('_', ' ').title()} {activation.gate}.{activation.line}"
    assert any(f.value == "Design Earth 42.6" and f.source_path == "raw_vision.design.earth" for f in context.facts)


def test_invalid_chart_blocks_before_downstream_callback(payload):
    payload["personality"]["sun"] = None
    result = result_from_payload(payload)
    assert not result.validation_result.is_valid
    retrieve = Mock()
    with pytest.raises(InvalidChartError):
        retrieve(build_chart_context(result))
    retrieve.assert_not_called()


def test_visual_only_evidence_and_private_warning_payloads_never_become_facts(payload):
    payload["visually_defined_centers"] = ["Head", "Solar Plexus"]
    payload["visually_active_gates"] = [64]
    payload["visible_colored_channels"] = ["47-64"]
    result = result_from_payload(payload)
    assert result.validation_result.is_valid
    private = "/private/chart.png name birth_date birth_time birth_place provider api_key base64 raw_response"
    result = replace(result,
        raw_vision=replace(result.raw_vision, uncertain_items=(UncertainItem("design.earth", private, private, 0.1),)),
        validation_result=ValidationResult(tuple(replace(w, message=private) for w in result.validation_result.warnings)))
    context = build_chart_context(result)
    assert not any(f.field == "active_gate" and f.value == 64 for f in context.facts)
    assert not any(f.field == "active_channel" and f.value == "47-64" for f in context.facts)
    assert any(f.field == "undefined_center" and f.value == "Head" for f in context.facts)
    assert context.validation_warnings == tuple(sorted({w.code.value for w in result.validation_result.warnings}))
    serialized = json.dumps(asdict(context))
    for excluded in (private, "visually_active_gates", "visible_colored_channels", "visually_defined_centers",
                     "uncertain_items", "confidence", "birth_date", "base64"):
        assert excluded not in serialized


def test_adapter_does_not_reparse_or_reinterpret(payload, monkeypatch):
    result = result_from_payload(payload)
    def forbidden(*args, **kwargs):
        pytest.fail("adapter must only project the typed result")
    monkeypatch.setattr("human_design.vision.parser.parse_bodygraph_raw_extraction_json", forbidden)
    monkeypatch.setattr("human_design.vision.interpreter.interpret_bodygraph", forbidden)
    assert build_chart_context(result).facts


def test_null_activations_are_not_projected_when_supplied_valid_result(payload):
    result = result_from_payload(payload)
    result = replace(result, raw_vision=replace(result.raw_vision,
        design=replace(result.raw_vision.design, earth=None)))
    assert all(f.source_path != "raw_vision.design.earth" for f in build_chart_context(result).facts)
