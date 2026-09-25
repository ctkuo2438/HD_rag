"""Small retrieval regression metrics and offline CLI contracts."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from human_design.rag import evaluation
from human_design.rag.config import load_config
from human_design.rag.evaluation import (
    EvaluationCase,
    RetrievalMetrics,
    evaluate_modes,
    load_evaluation_cases,
    score_query,
)
from human_design.rag.models import HybridSearchRequest, RetrievedChunk


FIXTURE = Path(__file__).parent / "fixtures/rag/hybrid_queries.example.json"
REQUEST = HybridSearchRequest("What is Gate 42?", "Explain completion in Gate 42", "Gate 42 Hexagram 42")


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_hybrid_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_hybrid_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(text="Unrelated synthetic passage", **fields):
    return RetrievedChunk("canonical-id", text, "reference.pdf", **fields)


def case(**changes):
    return replace(EvaluationCase("gate42", REQUEST, ("Gate 42",)), **changes)


@pytest.mark.parametrize("rank,hit,rr", [(1, 1, 1), (5, 1, 1/5), (6, 0, 1/6), (20, 0, 1/20), (21, 0, 0)])
def test_rank_boundaries(rank, hit, rr):
    ranked = [chunk() for _ in range(rank - 1)] + [chunk("Gate 42 is a synthetic entity label.")]
    metrics = score_query(case(), ranked)
    assert metrics.hit_at_5 == hit
    assert metrics.mrr_at_20 == rr
    assert metrics.exact_entity_hit_rate == (1 if rank <= 20 else 0)
    assert metrics.expected_source_page_hit_rate is None


def test_empty_results_and_missing_hit():
    for ranked in ([], [chunk()]):
        metrics = score_query(case(), ranked)
        assert (metrics.hit_at_5, metrics.mrr_at_20, metrics.exact_entity_hit_rate) == (0, 0, 0)


@pytest.mark.parametrize("entity,text,expected", [
    ("Gate 4", "Gate 42", 0), ("Gate 42", "Gate 420", 0),
    ("Channel 42-53", "Channel 42-530", 0), ("Profile 4/6", "Profile 4/60", 0),
    ("Generator", "Regenerator", 0), ("Generator", "Manifesting Generator", 0),
    ("Gate 42", "(gAtE   42).", 1), ("Channel 42-53", "Channel 42–53.", 1),
    ("Profile 4/6", "Profile 4 / 6.", 1), ("G", "G中心", 0),
    ("G", "The G Center", 1), ("Ego", "Ego Center", 1),
])
def test_exact_normalized_entity_matching(entity, text, expected):
    assert score_query(case(expected_entities=(entity,)), [chunk(text)]).exact_entity_hit_rate == expected


def test_entity_rate_measures_labeled_entity_coverage():
    metrics = score_query(case(expected_entities=("Gate 42", "Gate 53")), [chunk("Gate 42"), chunk("Gate 42")])
    assert metrics.exact_entity_hit_rate == 0.5


def test_source_page_labels_match_safe_public_fields_together():
    labeled = case(expected_entities=(), expected_sources=({"source_relpath": "books/reference.pdf", "page_number": 7},))
    for candidate in (
        chunk(source_relpath="other/reference.pdf", page_number=7),
        chunk(source_relpath="books/reference.pdf", page_number=8),
        chunk(metadata={"source_relpath": "books/reference.pdf", "page_number": 7}),
    ):
        assert score_query(labeled, [candidate]).expected_source_page_hit_rate == 0
    metrics = score_query(labeled, [chunk(source_relpath="books/reference.pdf", page_number=7)])
    assert metrics.expected_source_page_hit_rate == 1
    assert metrics.hit_at_5 == metrics.mrr_at_20 == 1
    assert metrics.exact_entity_hit_rate is None


class FakeRetriever:
    ingestion_id = "same-canonical-build"

    def __init__(self, backend, text):
        self.backend, self.text, self.calls = backend, text, []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return [replace(chunk(self.text), chunk_id=self.backend, **{f"{self.backend}_rank": 1})]


def test_modes_use_separate_queries_and_separate_aggregates():
    dense, sparse = FakeRetriever("dense", "Gate 42"), FakeRetriever("sparse", "Gate 4")
    settings = replace(load_config(env={}), dense_top_k=8, sparse_top_k=11)
    cases = [case(), case(case_id="gate4", expected_entities=("Gate 4",))]
    reports = evaluate_modes(cases, dense, sparse, settings)
    assert list(reports) == ["dense-only", "bm25-only", "rrf-hybrid"]
    assert dense.calls == [(REQUEST.dense_query, 8)] * 4
    assert sparse.calls == [(REQUEST.sparse_query, 11)] * 4
    assert reports["dense-only"].aggregate.hit_at_5 == 0.5
    assert reports["bm25-only"].aggregate.hit_at_5 == 0.5
    assert reports["rrf-hybrid"].aggregate.hit_at_5 == 1
    assert reports["rrf-hybrid"].aggregate.mrr_at_20 == 0.75
    assert [row.metrics.hit_at_5 for row in reports["dense-only"].per_query] == [1, 0]
    assert reports["dense-only"].aggregate.expected_source_page_hit_rate is None
    assert evaluate_modes(cases, dense, sparse, settings) == reports


def payload():
    return {"schema_version": 1, "cases": [{
        "case_id": "gate42", "original_query": REQUEST.original_query,
        "dense_query": REQUEST.dense_query, "sparse_query": REQUEST.sparse_query,
        "expected_entities": ["Gate 42"], "expected_sources": [],
    }]}


@pytest.mark.parametrize("field,value", [
    ("dense_query", ""), ("sparse_query", None), ("expected_entities", "Gate 42"),
    ("expected_entities", [""]), ("expected_sources", [{"source_file": "/private/reference.pdf"}]),
    ("expected_sources", [{"source_relpath": "../reference.pdf"}]),
    ("expected_sources", [{"source_file": "reference.pdf", "page_number": True}]),
    ("expected_sources", [{"source_path": "/private/reference.pdf"}]),
])
def test_fixture_schema_errors_name_case(tmp_path, field, value):
    data = payload()
    data["cases"][0][field] = value
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="gate42") as caught:
        load_evaluation_cases(path)
    assert "/private/" not in str(caught.value)


@pytest.mark.parametrize("problem", ["duplicate", "no-labels", "missing-query", "bad-root", "bad-json"])
def test_malformed_fixtures_fail(tmp_path, problem):
    data = payload()
    if problem == "duplicate":
        data["cases"].append(data["cases"][0])
    elif problem == "no-labels":
        data["cases"][0]["expected_entities"] = []
    elif problem == "missing-query":
        del data["cases"][0]["dense_query"]
    elif problem == "bad-root":
        data = []
    path = tmp_path / "queries.json"
    path.write_text("{" if problem == "bad-json" else json.dumps(data))
    with pytest.raises(ValueError):
        load_evaluation_cases(path)


def test_committed_fixture_has_required_sanitized_coverage():
    cases = load_evaluation_cases(FIXTURE)
    assert 10 <= len(cases) <= 15
    entities = {entity for query in cases for entity in query.expected_entities}
    assert {"Generator", "Sacral Authority", "Gate 42", "Channel 42-53", "Profile 4/6",
            "Split Definition", "Solar Plexus", "Emotional Authority", "G", "Ego"} <= entities
    queries = " ".join(query.request.original_query for query in cases)
    assert all(term in queries for term in ("情緒權威", "42號閘門", "4/6人生角色"))


def test_cli_injected_retrievers_never_open_storage_or_print_chunks(tmp_path, monkeypatch, capsys, cli):
    monkeypatch.chdir(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("offline CLI must not open storage or construct providers")
    monkeypatch.setattr(cli, "load_dense_retriever", forbidden)
    monkeypatch.setattr(cli, "load_bm25_retriever", forbidden)
    private_text = "Full synthetic passage that must not appear in output. Gate 42"
    dense, sparse = FakeRetriever("dense", private_text), FakeRetriever("sparse", private_text)
    assert cli.main(["--queries", str(FIXTURE)], retrievers=(dense, sparse), config=load_config(env={})) == 1
    output = capsys.readouterr().out
    assert all(mode in output for mode in ("dense-only", "bm25-only", "rrf-hybrid"))
    assert "Hit@5" in output and "MRR@20" in output and "aggregate" in output
    assert "gate42-en" in output
    assert private_text not in output


@pytest.mark.parametrize("enabled,key", [(False, None), (True, None)])
def test_cli_real_dense_gate_before_storage(tmp_path, monkeypatch, capsys, enabled, key, cli):
    monkeypatch.chdir(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("must gate real dense evaluation before opening any index")
    monkeypatch.setattr(cli, "load_dense_retriever", forbidden)
    monkeypatch.setattr(cli, "load_bm25_retriever", forbidden)
    settings = replace(load_config(env={}), real_embeddings=enabled, openai_api_key=key)
    assert cli.main(["--queries", str(FIXTURE)], config=settings) == 2
    assert ("OPENAI_API_KEY" if enabled else "HD_RAG_REAL_EMBEDDINGS") in capsys.readouterr().err


def test_cli_existing_loaders_receive_shared_identity(tmp_path, monkeypatch, capsys, cli):
    dense, sparse = FakeRetriever("dense", "Gate 42"), FakeRetriever("sparse", "Gate 42")
    settings = replace(load_config(env={}), index_dir=tmp_path, real_embeddings=True, openai_api_key="fake-key")
    monkeypatch.setattr(cli, "load_dense_retriever", lambda config: dense)
    calls = []
    def load_sparse(path, identity):
        calls.append((path, identity))
        return sparse
    monkeypatch.setattr(cli, "load_bm25_retriever", load_sparse)
    assert cli.main(["--queries", str(FIXTURE)], config=settings) == 1
    assert calls == [(tmp_path / "bm25", dense.ingestion_id)]
    assert list(tmp_path.iterdir()) == []
    assert "fake-key" not in capsys.readouterr().out


@pytest.mark.parametrize("metric,minimum", [
    ("hit_at_5", 0.8), ("mrr_at_20", 0.5),
    ("exact_entity_hit_rate", 0.8), ("expected_source_page_hit_rate", 0.8),
])
def test_aggregate_threshold_boundaries(metric, minimum):
    strong = RetrievalMetrics(1, 1, 1, 1)
    assert evaluation.threshold_failures(replace(strong, **{metric: minimum})) == ()
    assert evaluation.threshold_failures(replace(strong, **{metric: minimum - 0.001})) == (metric,)


@pytest.mark.parametrize("metric", ["exact_entity_hit_rate", "expected_source_page_hit_rate"])
def test_unlabeled_aggregate_metric_is_excluded_from_thresholds(metric):
    aggregate = replace(RetrievalMetrics(1, 1, 1, 1), **{metric: None})
    assert evaluation.threshold_failures(aggregate) == ()
    assert evaluation.threshold_failures(replace(aggregate, **{metric: 0})) == (metric,)


@pytest.mark.parametrize("dense_text,sparse_text,exit_code,failed_modes", [
    ("Gate 42", "Gate 42", 0, ()),
    ("Unrelated synthetic passage", "Unrelated synthetic passage", 1, ("dense-only", "bm25-only", "rrf-hybrid")),
    ("Gate 42", "Unrelated synthetic passage", 1, ("bm25-only",)),
    ("Unrelated synthetic passage", "Gate 42", 1, ("dense-only",)),
])
def test_cli_threshold_exit_codes_and_independent_mode_failures(
    tmp_path, capsys, cli, dense_text, sparse_text, exit_code, failed_modes,
):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload()))
    dense, sparse = FakeRetriever("dense", dense_text), FakeRetriever("sparse", sparse_text)
    assert cli.main(["--queries", str(path)], retrievers=(dense, sparse), config=load_config(env={})) == exit_code
    output = capsys.readouterr().out
    assert "source/page=n/a" in output  # absent source labels must not cause failure
    assert ("Regression thresholds FAILED" if failed_modes else "Regression thresholds passed") in output
    for mode in ("dense-only", "bm25-only", "rrf-hybrid"):
        assert (f"{mode}: threshold failure" in output) == (mode in failed_modes)
    assert "expected_source_page_hit_rate" not in output
    assert "Unrelated synthetic passage" not in output
    assert dense.calls == [(REQUEST.dense_query, 20)] * 2
    assert sparse.calls == [(REQUEST.sparse_query, 20)] * 2


def test_cli_thresholds_use_aggregates_without_per_query_failures(tmp_path, capsys, cli):
    data = payload()
    original = data["cases"][0]
    data["cases"] = [dict(original, case_id=f"gate-{i}") for i in range(5)]
    data["cases"][-1]["expected_entities"] = ["Gate 4"]
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(data))
    dense, sparse = FakeRetriever("dense", "Gate 42"), FakeRetriever("sparse", "Gate 42")
    assert cli.main(["--queries", str(path)], retrievers=(dense, sparse), config=load_config(env={})) == 0
    output = capsys.readouterr().out
    assert "Hit@5=0.0000" in output  # one failed query does not fail a passing aggregate
    assert "aggregate: Hit@5=0.8000" in output
    assert "Regression thresholds passed" in output


def test_cli_invalid_input_still_returns_two_before_retrieval(tmp_path, capsys, cli):
    path = tmp_path / "queries.json"
    path.write_text("{")
    dense, sparse = FakeRetriever("dense", "Gate 42"), FakeRetriever("sparse", "Gate 42")
    assert cli.main(["--queries", str(path)], retrievers=(dense, sparse), config=load_config(env={})) == 2
    output = capsys.readouterr()
    assert "Cannot evaluate retrieval" in output.err
    assert "Regression thresholds" not in output.out
    assert not dense.calls and not sparse.calls


def test_cli_index_setup_error_still_returns_two(tmp_path, monkeypatch, capsys, cli):
    def broken_index(config):
        raise ValueError("Phase 3 index is missing")
    monkeypatch.setattr(cli, "load_dense_retriever", broken_index)
    settings = replace(load_config(env={}), index_dir=tmp_path, real_embeddings=True, openai_api_key="fake-key")
    assert cli.main(["--queries", str(FIXTURE)], config=settings) == 2
    output = capsys.readouterr()
    assert "Phase 3 index is missing" in output.err
    assert "Regression thresholds" not in output.out


def test_canonical_channel_alone_earns_full_fixture_entity_credit():
    channel_case = next(case for case in load_evaluation_cases(FIXTURE) if case.case_id == "channel-en")
    metrics = score_query(channel_case, [chunk("A synthetic passage about Channel 42-53.")])
    assert metrics.exact_entity_hit_rate == 1.0
    assert channel_case.expected_entities == ("Channel 42-53",)
    assert "Channel 53-42" in channel_case.request.sparse_query
