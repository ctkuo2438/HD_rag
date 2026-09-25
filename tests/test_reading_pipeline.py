"""Offline orchestration: real query/RRF/prompt logic and injected provider boundaries."""

import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from human_design.rag.config import load_config
from human_design.rag.hybrid_index import HybridIndexError
from human_design.rag.models import RetrievedChunk
from human_design.rag.reranker import NoOpReranker, create_reranker
from human_design.reading import ReadingPipeline
from human_design.reading import pipeline as module
from human_design.reading.models import (
    AnswerResult, AnswerStatus, ChartImageQuestionRequest, ChartQuestionRequest, KnowledgeQuestionRequest,
)
from human_design.reading.prompt import build_source_citations
from human_design.reading.query_builder import InvalidQuestionError
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.models import BodyGraphExtractionResult
from human_design.vision.parser import parse_bodygraph_raw_extraction_json
from human_design.vision.validation import validate_bodygraph_extraction


@pytest.fixture
def chart():
    raw = (Path(__file__).parent / "fixtures/bodygraph/test1_raw_response.json").read_text()
    parsed = parse_bodygraph_raw_extraction_json(raw)
    derived = interpret_bodygraph(parsed.raw_vision)
    return BodyGraphExtractionResult(parsed.raw_vision, derived.derived_chart_data,
        validate_bodygraph_extraction(parse_result=parsed, interpretation_result=derived))


def settings(**changes):
    return replace(load_config(env={}), final_top_k=2, **changes)


def chunks(backend):
    return [RetrievedChunk(f"chunk-{i}", f"Synthetic passage number {i}.", "reference.pdf",
                           **{f"{backend}_rank": i, f"{backend}_score": 0.7}) for i in range(1, 4)]


class FakeRetriever:
    ingestion_id = "same-canonical-ingestion"

    def __init__(self, backend, events, values=None):
        self.backend, self.events = backend, events
        self.values = chunks(backend) if values is None else values

    def retrieve(self, query, top_k):
        self.events.append((self.backend, query, top_k))
        return self.values


def service(events, *, config=None, values=None, reranker=None, **kwargs):
    captured = []

    def generate(prompt, config):
        events.append(("generate",))
        captured.append(prompt)
        return AnswerResult(AnswerStatus.OK, "Grounded explanation [S1].",
                            citations=build_source_citations(prompt)[:1], chart_facts_used=prompt.chart_facts[:1])

    return ReadingPipeline(config=config or settings(),
        dense_retriever=FakeRetriever("dense", events, values),
        sparse_retriever=FakeRetriever("sparse", events, values),
        reranker=reranker, generator=generate, **kwargs), captured


def record_function(monkeypatch, target, name, events):
    original = getattr(target, name)
    def record(*args, **kwargs):
        events.append((name,))
        return original(*args, **kwargs)
    monkeypatch.setattr(target, name, record)


def test_exact_knowledge_order_and_purpose_specific_queries(monkeypatch):
    from human_design.rag import hybrid_retriever

    events = []
    for name in ("validate_query", "build_retrieval_queries", "build_prompt_context"):
        record_function(monkeypatch, module, name, events)
    record_function(monkeypatch, hybrid_retriever, "fuse_rrf", events)
    rerank = SimpleNamespace(rerank=lambda q, c, n: (events.append(("rerank", q, n)) or list(c)[:n]))
    pipeline, captured = service(events, reranker=rerank)
    answer = pipeline.answer_knowledge_question(KnowledgeQuestionRequest("  Explain Hexagram 42  "))
    assert [e[0] for e in events] == ["validate_query", "build_retrieval_queries", "dense", "sparse",
                                    "fuse_rrf", "rerank", "build_prompt_context", "generate"]
    assert events[2][1] == "Explain Hexagram 42"
    assert "Gate 42" in events[3][1] and events[3][1] != events[2][1]
    assert events[5][1] == events[2][1]
    assert captured[0].chart_facts == ()
    assert [(s.citation_id, s.chunk.chunk_id) for s in captured[0].sources] == [("S1", "chunk-1"), ("S2", "chunk-2")]
    assert answer.status is AnswerStatus.OK
    serialized = json.dumps(asdict(answer))
    for term in ("dense_score", "sparse_rank", "rrf_score", "rerank_score", "Synthetic passage", "chunk-3"):
        assert term not in serialized


def test_chart_adaptation_and_fact_selection_before_retrieval_without_image_access(chart, monkeypatch):
    events = []
    for name in ("build_chart_context", "build_retrieval_queries"):
        record_function(monkeypatch, module, name, events)
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=AssertionError("core must never read images")))
    pipeline, captured = service(events)
    answer = pipeline.answer_chart_question(ChartQuestionRequest("Explain my Profile", chart))
    assert [e[0] for e in events][:4] == ["build_chart_context", "build_retrieval_queries", "dense", "sparse"]
    assert [f.field for f in captured[0].chart_facts] == ["profile"]
    assert answer.chart_facts_used == captured[0].chart_facts
    assert "Profile" in events[2][1]


@pytest.mark.parametrize("kind", ["knowledge", "chart", "image"])
@pytest.mark.parametrize("query", ["", " \n\t", "x" * 2001], ids=["empty", "whitespace", "oversized"])
def test_invalid_query_precedes_every_dependency(kind, query, chart, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("invalid input must stop first"))
    for name in ("load_config", "build_chart_context", "build_retrieval_queries", "load_dense_retriever",
                 "load_bm25_retriever", "create_reranker", "build_prompt_context", "generate_answer"):
        monkeypatch.setattr(module, name, forbidden)
    pipeline = ReadingPipeline(extractor=forbidden)
    requests = {"knowledge": KnowledgeQuestionRequest(query), "chart": ChartQuestionRequest(query, chart),
                "image": ChartImageQuestionRequest(query, Path("unread.png"))}
    with pytest.raises(InvalidQuestionError):
        getattr(pipeline, f"answer_{'chart_image' if kind == 'image' else kind}_question")(requests[kind])
    forbidden.assert_not_called()


def test_invalid_chart_short_circuits_before_configuration_and_retrieval(chart, monkeypatch):
    from human_design.vision.models import ParseResult

    raw = replace(chart.raw_vision, personality=replace(chart.raw_vision.personality, sun=None))
    interpreted = interpret_bodygraph(raw)
    invalid = BodyGraphExtractionResult(raw, interpreted.derived_chart_data,
        validate_bodygraph_extraction(parse_result=ParseResult(raw), interpretation_result=interpreted))
    assert not invalid.validation_result.is_valid
    forbidden = Mock(side_effect=AssertionError("invalid chart must stop first"))
    monkeypatch.setattr(module, "load_config", forbidden)
    pipeline = ReadingPipeline(generator=forbidden)
    result = pipeline.answer_chart_question(ChartQuestionRequest("Explain Profile", invalid))
    assert result.status is AnswerStatus.INVALID_CHART
    assert result.citations == result.chart_facts_used == result.warnings == ()
    forbidden.assert_not_called()


@pytest.mark.parametrize("with_chart", [False, True])
def test_no_fused_evidence_never_reranks_renders_or_generates(with_chart, chart, monkeypatch):
    events = []
    forbidden = Mock(side_effect=AssertionError("no evidence must stop first"))
    for name in ("create_reranker", "build_prompt_context"):
        monkeypatch.setattr(module, name, forbidden)
    pipeline, captured = service(events, values=[])
    result = pipeline.answer_chart_question(ChartQuestionRequest("Explain Profile", chart)) if with_chart else (
        pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Gate 42")))
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.citations == result.chart_facts_used == ()
    assert [e[0] for e in events] == ["dense", "sparse"] and not captured
    forbidden.assert_not_called()


def test_empty_reranking_result_never_builds_prompt(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("no final evidence"))
    monkeypatch.setattr(module, "build_prompt_context", forbidden)
    pipeline, captured = service([], reranker=SimpleNamespace(rerank=lambda *args: []))
    assert pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Gate 42")).status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert not captured


def test_noop_and_fake_cohere_use_existing_adapters_and_only_final_sources(monkeypatch):
    for provider in ("none", "cohere"):
        calls = []
        config = settings(rerank_provider=provider, real_rerank_api=provider == "cohere",
                          cohere_api_key="fake-key", rerank_model="fake-model")
        client = SimpleNamespace(rerank=Mock(return_value=SimpleNamespace(results=[
            SimpleNamespace(index=2, relevance_score=0.5), SimpleNamespace(index=0, relevance_score=0.1)])))
        def factory(c):
            calls.append(c.rerank_provider)
            return create_reranker(c, client=client)
        monkeypatch.setattr(module, "create_reranker", factory)
        pipeline, captured = service([], config=config)
        pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Hexagram 42"))
        assert calls == [provider]
        expected = ["chunk-1", "chunk-2"] if provider == "none" else ["chunk-3", "chunk-1"]
        assert [s.chunk.chunk_id for s in captured[0].sources] == expected
        assert [s.citation_id for s in captured[0].sources] == ["S1", "S2"]
        if provider == "none":
            client.rerank.assert_not_called()
        else:
            assert client.rerank.call_args.kwargs["query"] == "Hexagram 42"
            assert all(s.chunk.rrf_score is not None for s in captured[0].sources)


def test_pipeline_enforces_final_top_k_even_for_injected_reranker():
    pipeline, captured = service([], reranker=SimpleNamespace(rerank=lambda q, c, n: list(reversed(c))))
    pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Gate 42"))
    assert len(captured[0].sources) == 2


def test_identity_mismatch_prevents_both_queries():
    events = []
    pipeline, _ = service(events)
    pipeline.sparse_retriever.ingestion_id = "other-ingestion"
    with pytest.raises(HybridIndexError):
        pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Gate 42"))
    assert not events


@pytest.mark.parametrize("missing", ["dense_retriever", "sparse_retriever"])
def test_incomplete_injected_pair_never_falls_back(missing):
    events = []
    pipeline, _ = service(events)
    pipeline = replace(pipeline, **{missing: None})
    with pytest.raises(HybridIndexError):
        pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Gate 42"))
    assert not events


def test_production_pair_load_is_lazy_and_requires_bm25_before_dense_construction(tmp_path, monkeypatch):
    config = settings(index_dir=tmp_path, real_embeddings=True, openai_api_key="fake-key")
    manifest = SimpleNamespace(ingestion_id="same-canonical-ingestion")
    monkeypatch.setattr(module, "load_manifest", Mock(return_value=manifest))
    monkeypatch.setattr(module, "load_nodes", Mock(return_value=[]))
    sparse = Mock(side_effect=HybridIndexError("Missing BM25"))
    dense = Mock(side_effect=AssertionError("no dense-only output"))
    monkeypatch.setattr(module, "load_bm25_retriever", sparse)
    monkeypatch.setattr(module, "load_dense_retriever", dense)
    pipeline = ReadingPipeline(config=config)
    sparse.assert_not_called()
    with pytest.raises(HybridIndexError):
        pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Gate 42"))
    sparse.assert_called_once_with(tmp_path / "bm25", manifest.ingestion_id)
    dense.assert_not_called()


def test_production_pair_validates_canonical_ids_and_identity(tmp_path, monkeypatch):
    config = settings(index_dir=tmp_path, real_embeddings=True, openai_api_key="fake-key")
    events = []
    dense, sparse = FakeRetriever("dense", events), FakeRetriever("sparse", events)
    sparse.chunk_ids = ("chunk-1", "chunk-2", "chunk-3")
    monkeypatch.setattr(module, "load_manifest", Mock(return_value=SimpleNamespace(ingestion_id=dense.ingestion_id)))
    monkeypatch.setattr(module, "load_nodes", Mock(return_value=[SimpleNamespace(node_id=i) for i in sparse.chunk_ids]))
    monkeypatch.setattr(module, "load_bm25_retriever", Mock(return_value=sparse))
    monkeypatch.setattr(module, "load_dense_retriever", Mock(return_value=dense))
    generator = Mock(return_value=AnswerResult(AnswerStatus.OK, "Answer [S1]."))
    pipeline = ReadingPipeline(config=config, generator=generator)
    pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Gate 42"))
    assert [e[0] for e in events] == ["dense", "sparse"]
    sparse.chunk_ids = ("different-id",)
    events.clear()
    with pytest.raises(HybridIndexError):
        pipeline.answer_knowledge_question(KnowledgeQuestionRequest("Gate 42"))
    assert not events


def test_image_facade_passes_path_only_to_phase2_extractor_then_typed_core(chart, monkeypatch):
    path = Path("/private/synthetic-chart.png")
    extractor = Mock(return_value=chart)
    core = Mock(return_value=AnswerResult(AnswerStatus.INSUFFICIENT_EVIDENCE, "No evidence."))
    monkeypatch.setattr(ReadingPipeline, "answer_chart_question", core)
    pipeline = ReadingPipeline(extractor=extractor)
    pipeline.answer_chart_image_question(ChartImageQuestionRequest("Explain Profile", path))
    extractor.assert_called_once_with(path)
    core.assert_called_once_with(ChartQuestionRequest("Explain Profile", chart))


def test_production_image_extractor_reuses_phase2_flow(chart, monkeypatch):
    from human_design.vision import pipeline as phase2
    from human_design.vision.config import load_vision_config

    raw = (Path(__file__).parent / "fixtures/bodygraph/test1_raw_response.json").read_text()
    events = []
    vision_config = load_vision_config(env={"HD_VISION_REAL_API": "1", "OPENAI_API_KEY": "fake-key"})
    monkeypatch.setattr(module, "load_vision_config", lambda: vision_config)
    extraction = Mock(return_value=raw)
    monkeypatch.setattr(phase2, "extract_bodygraph_raw_json", extraction)
    for name in ("parse_bodygraph_raw_extraction_json", "interpret_bodygraph", "validate_bodygraph_extraction"):
        record_function(monkeypatch, phase2, name, events)
    core = Mock(return_value=AnswerResult(AnswerStatus.INSUFFICIENT_EVIDENCE, "No evidence."))
    monkeypatch.setattr(ReadingPipeline, "answer_chart_question", core)
    path = Path("not-read.png")
    ReadingPipeline().answer_chart_image_question(ChartImageQuestionRequest("Profile", path))
    assert extraction.call_args.kwargs["image_path"] == path
    assert [e[0] for e in events] == ["parse_bodygraph_raw_extraction_json", "interpret_bodygraph", "validate_bodygraph_extraction"]
    assert core.call_args.args[0].bodygraph_result == chart


@pytest.mark.parametrize("stage", ["extractor", "reranker", "generator"])
def test_operational_failures_are_sanitized(stage, chart, capsys):
    private = "fake-key /private/chart.png base64-marker birth-date-1901 complete private passage"
    bad = Mock(side_effect=RuntimeError(private))
    pipeline, _ = service([], extractor=bad if stage == "extractor" else None,
                          reranker=SimpleNamespace(rerank=bad) if stage == "reranker" else NoOpReranker())
    if stage == "generator":
        pipeline = replace(pipeline, generator=bad)
    with pytest.raises(module.ReadingPipelineError) as caught:
        if stage == "extractor":
            pipeline.answer_chart_image_question(ChartImageQuestionRequest("Profile", Path("not-read.png")))
        else:
            pipeline.answer_chart_question(ChartQuestionRequest("Profile", chart))
    assert private not in "".join(traceback.format_exception(caught.value)) + str(capsys.readouterr())


@pytest.mark.parametrize("query", [
    "Give me a complete reading", "Read my entire chart", "Explain everything in my chart",
    "Give me a complete reading of my chart", "Give me a complete reading of my chart.",
    "Give me a full reading", "Give me a full reading of my chart", "Give me a full reading of my chart.",
    "  GIVE\tme a FULL reading\nof my CHART?!  ",
    "Tell me everything about my Human Design", "完整解讀我的人類圖", "幫我看完整張圖", "全部都解釋給我",
    "  GIVE me a COMPLETE reading!!! ", "Read\tmy entire\nchart?", "Please, read my entire chart.",
    "Tell me everything about my Human Design！", " 完整 解讀 我的 人類圖。 ", "幫我看完整張圖？！",
    "完整解读我的人类图", "帮我看完整张图", "全部都解释给我",
])
@pytest.mark.parametrize("kind", ["knowledge", "chart", "image"])
def test_broad_reading_stops_before_every_dependency(query, kind, chart, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("scope guard must stop first"))
    for name in ("load_config", "load_vision_config", "extract_bodygraph", "build_chart_context",
                 "build_retrieval_queries", "load_manifest", "load_nodes", "load_dense_retriever",
                 "load_bm25_retriever", "create_reranker", "build_prompt_context", "generate_answer"):
        monkeypatch.setattr(module, name, forbidden)
    retriever = SimpleNamespace(ingestion_id="unused", retrieve=forbidden)
    pipeline = ReadingPipeline(extractor=forbidden, dense_retriever=retriever, sparse_retriever=retriever,
                               reranker=SimpleNamespace(rerank=forbidden), generator=forbidden)
    requests = {"knowledge": KnowledgeQuestionRequest(query), "chart": ChartQuestionRequest(query, chart),
                "image": ChartImageQuestionRequest(query, Path("/private/unread.png"))}
    result = getattr(pipeline, f"answer_{'chart_image' if kind == 'image' else kind}_question")(requests[kind])
    assert result == AnswerResult(AnswerStatus.NEEDS_FOCUS,
        "Please choose a specific topic, such as Authority, Profile, a Gate, a Channel, a Center, Type, Strategy, or Definition.")
    assert result.citations == result.chart_facts_used == result.warnings == ()
    forbidden.assert_not_called()


@pytest.mark.parametrize("query", [
    "Gate 42", "Channel 42-53", "Profile 4/6", "Sacral Authority", "Solar Plexus Center",
    "Generator Type", "Strategy", "Split Definition", "How does Gate 42 complete Channel 42-53?",
    "Explain everything about Gate 42", "Give me a complete explanation of Sacral Authority",
    "Tell me everything about my Profile", "完整解釋我的薦骨權威", "說明42號閘門如何完成通道",
])
def test_focused_questions_and_complete_everything_counterexamples_continue(query):
    events = []
    pipeline, captured = service(events)
    assert pipeline.answer_knowledge_question(KnowledgeQuestionRequest(query)).status is AnswerStatus.OK
    assert [event[0] for event in events] == ["dense", "sparse", "generate"]
    assert len(captured) == 1


def test_validation_precedes_broad_scope_matching(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("validation must happen first"))
    monkeypatch.setattr(module, "_needs_focus", forbidden)
    for query in ("", "Give me a complete reading" + " " * 2000):
        with pytest.raises(InvalidQuestionError):
            ReadingPipeline().answer_knowledge_question(KnowledgeQuestionRequest(query))
    forbidden.assert_not_called()
