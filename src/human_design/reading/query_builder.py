"""Deterministic focused queries over existing Task 35 facts; no retrieval or inference."""

from __future__ import annotations

import re
from collections.abc import Callable

from human_design.rag.bm25 import normalize_sparse_query
from human_design.rag.models import HybridSearchRequest
from human_design.reading.models import ChartContext, ChartFact
from human_design.vision.constants import ALL_CHANNELS, CANONICAL_CENTERS, CENTER_ALIASES, PLANETARY_FIELDS


MAX_QUERY_CHARACTERS = 2000


class InvalidQuestionError(ValueError):
    """Invalid question content, without echoing potentially private input."""


def validate_query(user_query: str) -> str:
    """Require nonblank text, at most 2,000 Python characters before trimming.

    Count code points with len(), not encoded bytes, tokens, or grapheme clusters.
    This boundary must run before query construction or any downstream work.
    """
    if not isinstance(user_query, str):
        raise InvalidQuestionError("Question must be a string")
    if len(user_query) > MAX_QUERY_CHARACTERS:
        raise InvalidQuestionError("Question must contain at most 2,000 Python characters")
    normalized = user_query.strip()
    if not normalized:
        raise InvalidQuestionError("Question must not be empty or whitespace-only")
    return normalized


_INTENTS = {
    "decision": ("decide", "decision", "decisions", "decision-making", "做決定", "做决定", "決策", "决策"),
    "authority": ("Authority", "權威", "权威"),
    "type": ("Type", "類型", "类型", "Generator", "Manifestor", "Projector", "Reflector"),
    "strategy": ("Strategy", "策略"),
    "profile": ("Profile", "人生角色"),
    "definition": ("Definition", "定義", "定义"),
    "connection": ("Channel", "connect", "connects", "connection", "pair", "paired", "pairing", "通道", "連接", "连接", "配對", "配对"),
    "activation": ("Gate", "Hexagram", "Line", "activation", "activations", "planet", "planetary", "Personality", "Design", "閘門", "闸门", "爻", "行星", "啟動", "启动"),
}
_DEFINITION_ALIASES = {
    "Single Definition": ("一分人",), "Split Definition": ("二分人",),
    "Triple Split Definition": ("三分人",), "Quadruple Split Definition": ("四分人",),
    "No Definition": ("無定義", "无定义"),
}
_STRATEGY_ALIASES = {
    "To Respond": ("等待回應", "等待回应"),
    "Wait for the Invitation": ("等待邀請", "等待邀请"),
    "To Inform": ("告知",),
    "Wait a Lunar Cycle": ("等待月亮週期", "等待月亮周期"),
}
_AUTHORITY_PHRASE = re.compile(
    r"\b(?:Sacral|Emotional|Solar\s+Plexus|Splenic|Ego(?:[- ]Projected|[- ]Manifested)?|Heart|Will|"
    r"Self[- ]Projected|Mental|Lunar)\s+Authority\b"
    r"|(?:薦骨|荐骨|情緒|情绪|脾臟|脾脏|直覺|直觉|意志力|自我投射|心智|月亮)(?:權威|权威)", re.I,
)
_GATE_LINE = re.compile(r"(?<![\d.])([1-9]\d?)\.([1-6])(?![\d.])")


def _contains(text: str, phrase: str) -> bool:
    if not phrase.isascii():
        return phrase in text
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I) is not None


def _intent(text: str, name: str) -> bool:
    aliases = _INTENTS[name] + (tuple(_STRATEGY_ALIASES) if name == "strategy" else ())
    return any(_contains(text, alias) for alias in aliases)


def _normalized_query(query: str) -> str:
    # Small syntactic variants precede the shared Task 30 lexical normalizer.
    text = re.sub(r"\b(Gate|Hexagram|Profile)\s*[:：]\s*", r"\1 ", query, flags=re.I)
    text = re.sub(r"(?<!\d)(\d{1,2})\s+Gate\b", r"Gate \1", text, flags=re.I)
    text = re.sub(r"(?:閘門|闸门)\s*(\d{1,2})(?!\d)", r"Gate \1", text)
    text = re.sub(r"第?([1-6])\s*爻", r" Line \1 ", text)
    for canonical, aliases in (_DEFINITION_ALIASES | _STRATEGY_ALIASES).items():
        if any(alias in text for alias in aliases):
            text += f" {canonical}"
    center_text = _AUTHORITY_PHRASE.sub("", text)
    for alias, canonical in CENTER_ALIASES.items():
        if _contains(center_text, alias) and not _contains(text, canonical):
            text += f" {canonical}"
    return normalize_sparse_query(text)


def _entities(text: str) -> tuple[set[int], set[str], set[tuple[int, int]]]:
    gates = {int(g) for g in re.findall(r"\b(?:Gate|Hexagram)\s+(\d{1,2})(?!\d)", text, re.I) if 1 <= int(g) <= 64}
    channels = {"-".join(map(str, sorted((int(a), int(b)))))
                for a, b in re.findall(r"\bChannel\s+(\d{1,2})-(\d{1,2})(?!\d)", text, re.I)}
    gate_lines = {(int(g), int(line)) for g, line in _GATE_LINE.findall(text) if 1 <= int(g) <= 64}
    return gates | {g for g, _ in gate_lines}, channels, gate_lines


def _activation_matches(
    fact: ChartFact, *, gates: set[int], columns: set[str], planets: set[str], lines: set[int],
    gate_lines: set[tuple[int, int]],
) -> bool:
    if fact.field not in {"personality_activation", "design_activation"} or not isinstance(fact.value, str):
        return False
    match = _GATE_LINE.search(fact.value)
    if match is None:
        return False
    column, planet = fact.source_path.split(".")[1:]
    gate, line = map(int, match.groups())
    return (not gates or gate in gates) and (not lines or line in lines) and (
        not columns or column in columns) and (not planets or planet in planets) and (
        not gate_lines or (gate, line) in gate_lines)


def _select_facts(query: str, normalized: str, context: ChartContext) -> tuple[ChartFact, ...]:
    gates, channel_terms, gate_lines = _entities(normalized)
    channels = channel_terms & set(ALL_CHANNELS)
    endpoints = {int(g) for c in channels for g in c.split("-")}
    columns = {c for c in ("personality", "design") if _contains(normalized, c)}
    planets = {p for p in PLANETARY_FIELDS if _contains(normalized, p.replace("_", " "))}
    lines = {int(line) for line in re.findall(r"\bLine\s+([1-6])(?!\d)", normalized, re.I)} | {line for _, line in gate_lines}
    selected: set[str] = set()

    def add(predicate: Callable[[ChartFact], bool]) -> None:
        selected.update(f.source_path for f in context.facts if predicate(f))

    def activations(matching_gates: set[int]) -> None:
        add(lambda f: _activation_matches(f, gates=matching_gates, columns=columns, planets=planets,
                                          lines=lines, gate_lines=gate_lines))

    def basic(*names: str) -> None:
        add(lambda f: f.field in names)

    if planets or gate_lines:
        activations(gates or endpoints)
        add(lambda f: f.field == "active_gate" and f.value in gates)
    else:
        # A center embedded in an authority name is not an explicit center question.
        center_text = _normalized_query(_AUTHORITY_PHRASE.sub("", query))
        centers = {c for c in CANONICAL_CENTERS if _contains(center_text, c)}
        if centers:
            add(lambda f: f.field in {"defined_center", "undefined_center"} and f.value in centers)
            for name in ("type", "authority", "definition"):
                if _intent(normalized, name):
                    basic(name)
        elif _intent(normalized, "definition") and not channels:
            basic("definition")
        else:
            if gates:
                add(lambda f: f.field == "active_gate" and f.value in gates)
                activations(gates)
                if _intent(normalized, "connection") and not channel_terms:
                    add(lambda f: f.field == "active_channel" and bool(gates & set(map(int, str(f.value).split("-")))))
            if channels:
                add(lambda f: (f.field == "active_channel" and f.value in channels)
                    or (f.field == "active_gate" and f.value in endpoints))
                if _intent(normalized, "activation"):
                    activations(gates or endpoints)
            if _intent(normalized, "decision") and not gates and not channel_terms:
                basic("type", "authority", "strategy")
            else:
                if _intent(normalized, "authority"):
                    basic("authority")
                if _intent(normalized, "type") or _intent(normalized, "strategy"):
                    basic("type", "strategy")
                for name in ("profile", "definition"):
                    if _intent(normalized, name):
                        basic(name)
                if not (gates or channel_terms or any(_intent(normalized, name) for name in
                        ("authority", "type", "strategy", "profile", "definition"))):
                    basic("type", "authority", "profile")
    return tuple(f for f in context.facts if f.source_path in selected)


def _fact_sentence(fact: ChartFact) -> str:
    if fact.field == "active_gate":
        return f"Gate {fact.value} is active."
    if fact.field == "active_channel":
        return f"Channel {fact.value} is active."
    if fact.field in {"defined_center", "undefined_center"}:
        return f"The {fact.value} Center is {'defined' if fact.field == 'defined_center' else 'undefined'}."
    if fact.field in {"personality_activation", "design_activation"}:
        return f"The chart includes the activation {fact.value}."
    return f"The chart's {fact.field.replace('_', ' ')} is {fact.value}."


def _fact_terms(fact: ChartFact) -> str:
    if fact.field == "active_gate":
        return f"Gate {fact.value}"
    if fact.field == "active_channel":
        return f"Channel {fact.value}"
    if fact.field in {"defined_center", "undefined_center"}:
        return f"{fact.value} Center {'defined' if fact.field == 'defined_center' else 'undefined'}"
    if fact.field in {"personality_activation", "design_activation"}:
        return str(fact.value)
    if fact.field == "authority":
        return f"{fact.value} Authority"
    return f"{fact.field.replace('_', ' ').title()} {fact.value}"


def build_retrieval_queries(
    user_query: str, chart_context: ChartContext | None,
) -> tuple[HybridSearchRequest, tuple[ChartFact, ...]]:
    original = validate_query(user_query)
    normalized = _normalized_query(original)
    selected = _select_facts(original, normalized, chart_context) if chart_context is not None else ()
    dense = original
    if selected:
        dense += " Relevant chart facts: " + " ".join(_fact_sentence(f) for f in selected)
    sparse = " ".join([normalized, *(_fact_terms(f) for f in selected)]).strip()
    _, channels, gate_lines = _entities(sparse)
    for channel in sorted(channels & set(ALL_CHANNELS)):
        sparse += " " + " ".join(f"Gate {g}" for g in channel.split("-"))
    for gate, line in sorted(gate_lines):
        sparse += f" Gate {gate} Hexagram {gate} Line {line}"
    return HybridSearchRequest(original, dense, normalize_sparse_query(sparse)), selected
