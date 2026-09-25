"""Deterministic intent selection from Task 35 facts; no retrieval execution."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from human_design.reading.chart_context import build_chart_context
from human_design.reading.query_builder import build_retrieval_queries
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.models import BodyGraphExtractionResult
from human_design.vision.parser import parse_bodygraph_raw_extraction_json
from human_design.vision.validation import validate_bodygraph_extraction


def basic(*names):
    return {f"derived_chart_data.basic_info.{name}" for name in names}


def gate(number):
    return f"derived_chart_data.active_gates[gate={number}]"


def channel(value):
    return f"derived_chart_data.active_channels[channel={value}]"


GATE42 = {gate(42), "raw_vision.personality.earth", "raw_vision.design.earth"}
CHANNEL42 = {channel("42-53"), gate(42), gate(53)}


@pytest.fixture
def context():
    payload = json.loads((Path(__file__).parent / "fixtures/bodygraph/test1_raw_response.json").read_text())
    payload["personality"]["earth"] = "42.4"
    payload["design"]["sun"] = "53.6"
    parsed = parse_bodygraph_raw_extraction_json(json.dumps(payload))
    interpreted = interpret_bodygraph(parsed.raw_vision)
    validation = validate_bodygraph_extraction(parse_result=parsed, interpretation_result=interpreted)
    assert validation.is_valid
    return build_chart_context(BodyGraphExtractionResult(parsed.raw_vision, interpreted.derived_chart_data, validation))


@pytest.mark.parametrize("query,expected", [
    ("How should I make decisions according to my chart?", basic("type", "authority", "strategy")),
    ("How do I decide?", basic("type", "authority", "strategy")),
    ("我應該如何做決定？", basic("type", "authority", "strategy")),
    ("我该如何做决定？", basic("type", "authority", "strategy")),
    ("What is my Sacral Authority?", basic("authority")),
    ("什麼是情緒權威？", basic("authority")),
    ("How does my authority relate to my type and strategy?", basic("type", "authority", "strategy")),
    ("Explain Manifesting Generator", basic("type", "strategy")),
    ("顯示生產者的策略是什麼？", basic("type", "strategy")),
    ("我的类型与策略？", basic("type", "strategy")),
    ("What is my strategy?", basic("type", "strategy")),
    ("What is Profile 4/6?", basic("profile")),
    ("Explain 4-6 Profile", basic("profile")),
    ("4/6人生角色是什麼？", basic("profile")),
    ("What is Split Definition?", basic("definition")),
    ("我的定義是什麼？", basic("definition")),
    ("Explain my definition and type", basic("definition")),
    ("Explain my Profile and Gate 42", basic("profile") | GATE42),
    ("How does my definition involve Channel 42-53?", basic("definition") | CHANNEL42),
    ("What should I understand about myself?", basic("type", "authority", "profile")),
    ("What should I reflect on in my career?", basic("type", "authority", "profile")),
    ("What can I reflect on about my relationships?", basic("type", "authority", "profile")),
    ("我的職涯有什麼值得反思？", basic("type", "authority", "profile")),
    ("我的關係有什麼值得反思？", basic("type", "authority", "profile")),
])
def test_basic_intents_select_only_required_existing_facts(context, query, expected):
    request, selected = build_retrieval_queries(query, context)
    assert {f.source_path for f in selected} == expected
    assert selected == tuple(f for f in context.facts if f.source_path in expected)
    assert all(any(f is existing for existing in context.facts) for f in selected)
    assert build_retrieval_queries(query, context) == (request, selected)
    assert request.dense_query != request.sparse_query


@pytest.mark.parametrize("query", [
    "What does Gate 42 mean in my chart?", "Explain Hexagram 42", "Explain Gate: 42",
    "Gate #42", "42nd Gate", "42 Gate", "42號閘門代表什麼？", "42号闸门是什么？", "閘門42的意義？",
])
def test_generic_gate_never_adds_channels(context, query):
    request, selected = build_retrieval_queries(query, context)
    assert {f.source_path for f in selected} == GATE42
    assert "Gate 42" in request.sparse_query and "Hexagram 42" in request.sparse_query
    assert all(f.field != "active_channel" for f in selected)


@pytest.mark.parametrize("query", [
    "Is Gate 42 part of an active Channel?", "What does Gate 42 connect to?",
    "Does Gate 42 complete a Channel in my chart?", "How is Gate 42 paired?",
    "42號閘門連接哪個通道？", "42号闸门与什么配对？",
])
def test_connection_gate_selects_only_related_active_channel(context, query):
    _, selected = build_retrieval_queries(query, context)
    assert {f.source_path for f in selected} == GATE42 | {channel("42-53")}


@pytest.mark.parametrize("query", [
    "What does Channel 42-53 mean?", "Explain Channel 53-42", "42-53通道代表什麼？",
    "53與42通道的意義？", "53与42通道是什么？",
])
def test_generic_channel_excludes_planetary_activations(context, query):
    request, selected = build_retrieval_queries(query, context)
    assert {f.source_path for f in selected} == CHANNEL42
    for term in ("Channel 42-53", "Channel 53-42", "Gate 42", "Gate 53"):
        assert term in request.sparse_query


@pytest.mark.parametrize("query,activations", [
    ("What planetary activations form Channel 42-53?", {"raw_vision.personality.earth", "raw_vision.design.earth", "raw_vision.design.sun"}),
    ("Which Design activations form Channel 42-53?", {"raw_vision.design.earth", "raw_vision.design.sun"}),
    ("Which Personality activations form Channel 42-53?", {"raw_vision.personality.earth"}),
    ("Channel 42-53的行星啟動？", {"raw_vision.personality.earth", "raw_vision.design.earth", "raw_vision.design.sun"}),
])
def test_channel_activation_wording_is_explicit_and_endpoint_limited(context, query, activations):
    _, selected = build_retrieval_queries(query, context)
    assert {f.source_path for f in selected} == CHANNEL42 | activations


@pytest.mark.parametrize("query,expected", [
    ("Explain Design Earth 42.6", {gate(42), "raw_vision.design.earth"}),
    ("設計地球42.6的意義？", {gate(42), "raw_vision.design.earth"}),
    ("设计地球42.6是什么？", {gate(42), "raw_vision.design.earth"}),
    ("What is my Personality Earth?", {"raw_vision.personality.earth"}),
    ("What is my Design Sun?", {"raw_vision.design.sun"}),
    ("我的設計太陽？", {"raw_vision.design.sun"}),
    ("What is Earth at Gate 42 Line 4?", {gate(42), "raw_vision.personality.earth"}),
    ("Which Sun activation is in Channel 42-53?", {"raw_vision.design.sun"}),
    ("Explain Personality Earth Gate 64", set()),
])
def test_specific_planetary_selection(context, query, expected):
    request, selected = build_retrieval_queries(query, context)
    assert {f.source_path for f in selected} == expected
    if "42.6" in query:
        for term in ("Design Earth", "42.6", "Gate 42", "Hexagram 42", "Line 6"):
            assert term in request.sparse_query


@pytest.mark.parametrize("alias,canonical,defined", [
    ("G Center", "G", True), ("Self Center", "G", True), ("Identity", "G", True),
    ("G中心", "G", True), ("自我中心", "G", True),
    ("Heart", "Ego", True), ("Will Center", "Ego", True), ("Ego Center", "Ego", True),
    ("意志力中心", "Ego", True), ("心臟中心", "Ego", True),
    ("Solar Plexus", "Solar Plexus", False), ("情緒中心", "Solar Plexus", False),
    ("Sacral Center", "Sacral", True), ("薦骨中心", "Sacral", True),
])
def test_center_aliases_select_only_defined_or_complement_fact(context, alias, canonical, defined):
    request, selected = build_retrieval_queries(f"Explain my {alias}", context)
    selector = "center" if defined else "complement_center"
    assert {f.source_path for f in selected} == {f"derived_chart_data.defined_centers[{selector}={canonical}]"}
    assert canonical in request.sparse_query


def test_center_only_adds_explicit_type_authority(context):
    _, selected = build_retrieval_queries("How does my Solar Plexus Center relate to my type and authority?", context)
    assert {f.source_path for f in selected} == basic("type", "authority") | {
        "derived_chart_data.defined_centers[complement_center=Solar Plexus]"}


def test_decision_queries_are_semantic_and_do_not_dump_entities(context):
    request, selected = build_retrieval_queries("How do I make decisions?", context)
    for term in ("Manifesting Generator", "Sacral", "To Respond", "decisions"):
        assert term in request.dense_query and term in request.sparse_query
    assert all(term not in request.dense_query + request.sparse_query for term in ("Gate 42", "Channel 42-53", "Design Earth", "Profile"))
    assert len(selected) == 3


@pytest.mark.parametrize("query", ["What does Gate 64 mean in my chart?", "Explain Channel 47-64", "Explain Channel 5-7"])
def test_inactive_entities_are_never_invented(context, query):
    request, selected = build_retrieval_queries(query, context)
    assert selected == ()
    assert request.dense_query == query


@pytest.mark.parametrize("query,terms", [
    ("  What does Gate 42 mean?  ", ("Gate 42", "Hexagram 42")),
    ("4-6 Profile", ("Profile 4/6",)), ("4/6人生角色", ("Profile 4/6",)),
    ("Channel 53-42", ("Channel 42-53", "Gate 42", "Gate 53")),
    ("Design Earth 42.6", ("Design Earth", "Gate 42", "Line 6")),
    ("二分人的定義", ("Split Definition",)),
])
def test_knowledge_only_preserves_original_and_uses_sparse_normalization(query, terms):
    request, selected = build_retrieval_queries(query, None)
    assert selected == ()
    assert request.original_query == request.dense_query == query.strip()
    assert all(term in request.sparse_query for term in terms)
    if "Profile" in query or "人生角色" in query:
        assert "Channel" not in request.sparse_query


def test_specific_gate_line_pairs_do_not_form_cross_product(context):
    _, selected = build_retrieval_queries("Explain Earth 42.4 and Sun 53.6", context)
    assert {f.source_path for f in selected} == {
        gate(42), gate(53), "raw_vision.personality.earth", "raw_vision.design.sun"}


@pytest.mark.parametrize("query,expected", [
    ("What is my Personality North Node?", {"raw_vision.personality.north_node"}),
    ("我的設計南交點？", {"raw_vision.design.south_node"}),
    ("我的人格水星？", {"raw_vision.personality.mercury"}),
    ("Explain Design Pluto", {"raw_vision.design.pluto"}),
    ("How does Gate 42 work in Channel 42-53?", CHANNEL42 | GATE42),
    ("Which Line 6 activations form Channel 42-53?", CHANNEL42 | {"raw_vision.design.earth", "raw_vision.design.sun"}),
    ("Gate 42的第6爻", {gate(42), "raw_vision.design.earth"}),
    ("What is Self-Projected Authority?", basic("authority")),
    ("How does my Definition involve the G Center?", basic("definition") | {"derived_chart_data.defined_centers[center=G]"}),
])
def test_additional_explicit_entity_combinations(context, query, expected):
    _, selected = build_retrieval_queries(query, context)
    assert {f.source_path for f in selected} == expected


@pytest.mark.parametrize("phrase", ["To Respond", "Wait for the Invitation", "To Inform", "Wait a Lunar Cycle", "等待回應", "等待邀请"])
def test_exact_strategy_phrases_select_type_and_strategy(context, phrase):
    _, selected = build_retrieval_queries(f"Explain {phrase}", context)
    assert {f.source_path for f in selected} == basic("type", "strategy")


@pytest.mark.parametrize("alias,canonical,selector", [("Splenic", "Spleen", "center"), ("Emotional", "Solar Plexus", "complement_center")])
def test_remaining_phase2_center_aliases(context, alias, canonical, selector):
    query = f"Explain my {alias} Center"
    _, selected = build_retrieval_queries(query, context)
    assert {f.source_path for f in selected} == {f"derived_chart_data.defined_centers[{selector}={canonical}]"}
    knowledge, _ = build_retrieval_queries(query, None)
    assert canonical in knowledge.sparse_query


@pytest.mark.parametrize("query", ["", " ", "\n\t\u3000", "x" * 2001, " " + "x" * 2000])
def test_invalid_query_stops_before_every_downstream_boundary(query, context, monkeypatch):
    from human_design.reading.query_builder import InvalidQuestionError

    normalize = Mock(side_effect=AssertionError("must validate before normalization"))
    monkeypatch.setattr("human_design.reading.query_builder.normalize_sparse_query", normalize)
    dense, sparse, rerank, prompt, generate = [Mock() for _ in range(5)]
    # A caller-order harness, not a Task 40 production pipeline.
    with pytest.raises(InvalidQuestionError):
        request, facts = build_retrieval_queries(query, context)
        candidates = dense(request.dense_query) + sparse(request.sparse_query)
        final = rerank(request.dense_query, candidates)
        generate(prompt(request.original_query, facts, final))
    for boundary in (normalize, dense, sparse, rerank, prompt, generate):
        boundary.assert_not_called()


@pytest.mark.parametrize("character", ["x", "界", "🪐"])
def test_query_limit_is_exactly_2000_python_characters(character):
    from human_design.reading.query_builder import InvalidQuestionError, validate_query

    query = character * 2000
    assert validate_query(query) == query
    request, facts = build_retrieval_queries(query, None)
    assert request.original_query == request.dense_query == query and facts == ()
    with pytest.raises(InvalidQuestionError, match="2,000"):
        validate_query(query + character)


def test_query_validation_preserves_trimming_without_keyword_blocking():
    from human_design.reading.query_builder import validate_query

    for query in ("  Explain Gate 42  ", " Human Design and mental-health diagnosis? ",
                  "Reflect on financial decisions and relationships", "What is death in this book?"):
        assert validate_query(query) == query.strip()
        assert build_retrieval_queries(query, None)[0].original_query == query.strip()


@pytest.mark.parametrize("query", [None, 42, b"question", []])
def test_invalid_query_type_is_a_safe_typed_error(query):
    from human_design.reading.query_builder import InvalidQuestionError, validate_query

    with pytest.raises(InvalidQuestionError):
        validate_query(query)
