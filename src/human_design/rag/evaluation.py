"""Small deterministic retrieval metrics, independent of CLI and provider construction.

A result is relevant when it matches any labeled entity or source/page target.
Hit@5 and MRR@20 use that relevance. Entity and source/page rates measure the
fraction of distinct labeled targets found in the first 20 results. Unlabeled
rates are None and excluded from macro averages; missing hits score zero.

Regression minimums apply independently to each mode's aggregate: Hit@5 >= 0.80,
MRR@20 >= 0.50, entity coverage >= 0.80, and source/page coverage >= 0.80.
Equality passes. None metrics have no labels and are excluded. Any remaining
metric below its minimum fails the run; individual queries have no pass/fail flag.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from human_design.rag.config import AppConfig
from human_design.rag.hybrid_retriever import HybridRetriever, Retriever
from human_design.rag.models import HybridSearchRequest, RetrievedChunk, _validate_relative_source


REGRESSION_THRESHOLDS = {
    "hit_at_5": 0.80,
    "mrr_at_20": 0.50,
    "exact_entity_hit_rate": 0.80,
    "expected_source_page_hit_rate": 0.80,
}


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    request: HybridSearchRequest
    expected_entities: tuple[str, ...] = ()
    expected_sources: tuple[Mapping[str, str | int], ...] = ()


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_5: float
    mrr_at_20: float
    exact_entity_hit_rate: float | None
    expected_source_page_hit_rate: float | None


@dataclass(frozen=True)
class QueryEvaluation:
    case_id: str
    query: str
    metrics: RetrievalMetrics


@dataclass(frozen=True)
class ModeEvaluation:
    per_query: tuple[QueryEvaluation, ...]
    aggregate: RetrievalMetrics


def threshold_failures(aggregate: RetrievalMetrics) -> tuple[str, ...]:
    """Return below-minimum aggregate metric names in stable order, skipping None."""
    return tuple(
        name for name, minimum in REGRESSION_THRESHOLDS.items()
        if (value := getattr(aggregate, name)) is not None and value < minimum
    )


def _normalize_entity_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold().translate(str.maketrans("‐‑–—", "----"))
    text = " ".join(text.split())
    return re.sub(r"(?<=\d)\s*([/-])\s*(?=\d)", r"\1", text)


def _entity_hit(entity: str, text: str) -> bool:
    label = _normalize_entity_text(entity)
    prefix = r"(?<!manifesting )" if label == "generator" else ""
    return re.search(prefix + r"(?<!\w)" + re.escape(label) + r"(?!\w)", _normalize_entity_text(text)) is not None


def _source_hit(target: Mapping[str, str | int], chunk: RetrievedChunk) -> bool:
    return all(getattr(chunk, name) == value for name, value in target.items())


def score_query(case: EvaluationCase, ranked: Sequence[RetrievedChunk]) -> RetrievalMetrics:
    candidates = ranked[:20]
    relevant = [
        any(_entity_hit(entity, chunk.text) for entity in case.expected_entities)
        or any(_source_hit(target, chunk) for target in case.expected_sources)
        for chunk in candidates
    ]
    entities = {_normalize_entity_text(entity) for entity in case.expected_entities}
    sources = {tuple(sorted(target.items())) for target in case.expected_sources}
    return RetrievalMetrics(
        hit_at_5=float(any(relevant[:5])),
        mrr_at_20=next((1 / rank for rank, hit in enumerate(relevant, 1) if hit), 0.0),
        exact_entity_hit_rate=(sum(any(_entity_hit(entity, c.text) for c in candidates) for entity in entities)
                               / len(entities)) if entities else None,
        expected_source_page_hit_rate=(sum(any(_source_hit(dict(target), c) for c in candidates) for target in sources)
                                      / len(sources)) if sources else None,
    )


def _aggregate(rows: Sequence[QueryEvaluation]) -> RetrievalMetrics:
    averages = {}
    for metric in fields(RetrievalMetrics):
        values = [getattr(row.metrics, metric.name) for row in rows if getattr(row.metrics, metric.name) is not None]
        averages[metric.name] = math.fsum(values) / len(values) if values else None
    return RetrievalMetrics(**averages)


def evaluate_modes(
    cases: Sequence[EvaluationCase], dense: Retriever, sparse: Retriever, config: AppConfig,
) -> dict[str, ModeEvaluation]:
    """Evaluate each mode independently with purpose-specific queries; never open indexes."""
    if not cases:
        raise ValueError("Evaluation requires at least one query case")
    hybrid = HybridRetriever(dense, sparse, config)
    reports = {}
    for mode in ("dense-only", "bm25-only", "rrf-hybrid"):
        rows = []
        for case in cases:
            if mode == "dense-only":
                ranked = dense.retrieve(case.request.dense_query, config.dense_top_k)
            elif mode == "bm25-only":
                ranked = sparse.retrieve(case.request.sparse_query, config.sparse_top_k)
            else:
                ranked = hybrid.retrieve(case.request).candidates
            rows.append(QueryEvaluation(case.case_id, case.request.original_query, score_query(case, ranked)))
        reports[mode] = ModeEvaluation(tuple(rows), _aggregate(rows))
    return reports


def _parse_case(item: object) -> EvaluationCase:
    if not isinstance(item, dict):
        raise ValueError("query case must be an object")
    required = {"case_id", "original_query", "dense_query", "sparse_query"}
    if not required <= item.keys() or item.keys() - required - {"expected_entities", "expected_sources"}:
        raise ValueError("query case has missing or unknown fields")
    for key in required:
        if not isinstance(item[key], str) or not item[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", item["case_id"]):
        raise ValueError("case_id must be a safe identifier")
    entities, sources = item.get("expected_entities", []), item.get("expected_sources", [])
    if not isinstance(entities, list) or any(not isinstance(e, str) or not e.strip() for e in entities):
        raise ValueError("expected_entities must be a list of non-empty literals")
    if not isinstance(sources, list):
        raise ValueError("expected_sources must be a list")
    for source in sources:
        if (not isinstance(source, dict) or not {"source_file", "source_relpath"} & source.keys()
                or source.keys() - {"source_file", "source_relpath", "page_label", "page_number"}):
            raise ValueError("expected_sources requires safe source and optional page fields")
        for key, value in source.items():
            if key == "page_number":
                if type(value) is not int:
                    raise ValueError("page_number must be an integer")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
            elif key.startswith("source_"):
                _validate_relative_source(value, key, basename=key == "source_file")
    if not entities and not sources:
        raise ValueError("query case requires entity or source/page labels")
    request = HybridSearchRequest(*(item[key] for key in ("original_query", "dense_query", "sparse_query")))
    return EvaluationCase(item["case_id"], request, tuple(entities), tuple(dict(source) for source in sources))


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    """Read sanitized labels; schema errors identify the case without echoing local paths."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("Cannot read evaluation queries JSON") from None
    if (not isinstance(payload, dict) or set(payload) != {"schema_version", "cases"}
            or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
            or not isinstance(payload["cases"], list) or not payload["cases"]):
        raise ValueError("Evaluation queries require schema_version 1 and a non-empty cases list")
    cases, seen = [], set()
    for index, item in enumerate(payload["cases"], 1):
        identifier = item.get("case_id") if isinstance(item, dict) else None
        label = identifier if isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9_-]+", identifier) else f"#{index}"
        try:
            case = _parse_case(item)
            if case.case_id in seen:
                raise ValueError("duplicate case_id")
        except ValueError as exc:
            raise ValueError(f"Evaluation query {label}: {exc}") from None
        seen.add(case.case_id)
        cases.append(case)
    return tuple(cases)
