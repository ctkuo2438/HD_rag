"""Offline contracts for focused Phase 3 Q&A models."""

import json
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import get_type_hints

import pytest

import human_design.reading as reading
from human_design.rag.models import (
    HybridRetrievalResult,
    HybridSearchRequest,
    RetrievedChunk,
)
from human_design.reading.models import (
    AnswerResult,
    AnswerStatus,
    ChartContext,
    ChartFact,
    ChartImageQuestionRequest,
    ChartQuestionRequest,
    KnowledgeQuestionRequest,
    PromptContext,
    PromptSource,
    SourceCitation,
)
from human_design.vision.models import (
    BodyGraphExtractionResult,
    DerivedBasicInfo,
    DerivedChartData,
    DesignActivationColumn,
    PersonalityActivationColumn,
    RawVisionExtraction,
    ValidationResult,
)


@pytest.fixture
def chart_result() -> BodyGraphExtractionResult:
    """Synthetic typed result; no image, extraction client, or storage needed."""
    return BodyGraphExtractionResult(
        raw_vision=RawVisionExtraction(
            personality=PersonalityActivationColumn(),
            design=DesignActivationColumn(),
        ),
        derived_chart_data=DerivedChartData(
            basic_info=DerivedBasicInfo(
                type="Reflector",
                authority="Lunar",
                profile="Unknown",
                strategy="Wait a Lunar Cycle",
                definition="No Definition",
                not_self_theme="Disappointment",
                signature="Surprise",
            ),
            active_gates=(),
            active_channels=(),
            defined_centers=(),
        ),
        validation_result=ValidationResult(),
    )


@pytest.mark.parametrize(
    ("model", "hints"),
    [
        (KnowledgeQuestionRequest, {"query": str}),
        (ChartImageQuestionRequest, {"query": str, "bodygraph_image": Path}),
        (
            ChartQuestionRequest,
            {"query": str, "bodygraph_result": BodyGraphExtractionResult},
        ),
    ],
)
def test_request_contracts(model: type, hints: dict[str, type]) -> None:
    assert is_dataclass(model)
    assert model.__dataclass_params__.frozen
    assert {item.name for item in fields(model)} == set(hints)
    assert get_type_hints(model) == hints
    assert getattr(reading, model.__name__) is model


def test_requests_preserve_outer_image_and_typed_core_boundaries(
    chart_result: BodyGraphExtractionResult,
) -> None:
    requests = (
        KnowledgeQuestionRequest(query="What is Lunar Authority?"),
        ChartImageQuestionRequest(query="How do I decide?", bodygraph_image=Path("chart.png")),
        ChartQuestionRequest(query="How do I decide?", bodygraph_result=chart_result),
    )
    assert requests[2].bodygraph_result is chart_result
    for request in requests:
        with pytest.raises(FrozenInstanceError):
            request.query = "changed"
        assert (Path in get_type_hints(type(request)).values()) == isinstance(
            request, ChartImageQuestionRequest
        )


@pytest.mark.parametrize("query", [None, 42, [], {}])
def test_requests_reject_wrong_query_types(
    query: object, chart_result: BodyGraphExtractionResult,
) -> None:
    with pytest.raises(TypeError, match="query"):
        KnowledgeQuestionRequest(query=query)
    with pytest.raises(TypeError, match="query"):
        ChartImageQuestionRequest(query=query, bodygraph_image=Path("chart.png"))
    with pytest.raises(TypeError, match="query"):
        ChartQuestionRequest(query=query, bodygraph_result=chart_result)


@pytest.mark.parametrize("value", [None, object(), {}, Path("chart.png")])
def test_chart_core_requires_phase2_result(value: object) -> None:
    with pytest.raises(TypeError, match="bodygraph_result"):
        ChartQuestionRequest(query="How do I decide?", bodygraph_result=value)


@pytest.mark.parametrize("value", [None, "chart.png", object()])
def test_image_entry_requires_path(value: object) -> None:
    with pytest.raises(TypeError, match="bodygraph_image"):
        ChartImageQuestionRequest(query="How do I decide?", bodygraph_image=value)


def test_request_content_validation_is_deferred() -> None:
    assert KnowledgeQuestionRequest(query="").query == ""
    assert KnowledgeQuestionRequest(query="  ").query == "  "


def _chunk(**overrides: object) -> RetrievedChunk:
    values = {"chunk_id": "chunk-42", "text": "Synthetic Gate 42 passage.",
              "source_file": "reference.pdf"}
    return RetrievedChunk(**(values | overrides))


def _fact(**overrides: object) -> ChartFact:
    values = {"field": "active_gate", "value": 42,
              "source_path": "derived_chart_data.active_gates[gate=42]"}
    return ChartFact(**(values | overrides))


def _retrieval(**overrides: object) -> HybridRetrievalResult:
    values = {
        "request": HybridSearchRequest("What is Gate 42?", "Gate 42 meaning", "Gate 42"),
        "candidates": (), "ingestion_version": "v1", "ingestion_id": "corpus-identity",
        "dense_top_k": 20, "sparse_top_k": 20, "fusion_top_k": 20,
        "final_top_k": 4, "rrf_k": 60,
    }
    return HybridRetrievalResult(**(values | overrides))


@pytest.mark.parametrize("model", [
    HybridSearchRequest, RetrievedChunk, HybridRetrievalResult, ChartFact,
    ChartContext, PromptSource, PromptContext, SourceCitation, AnswerResult,
])
def test_phase3_models_are_frozen_and_fully_typed(model: type) -> None:
    assert is_dataclass(model)
    assert model.__dataclass_params__.frozen
    assert set(get_type_hints(model)) == {item.name for item in fields(model)}


def test_hybrid_request_preserves_separate_queries() -> None:
    request = HybridSearchRequest("什麼是薦骨權威？", "What is Sacral Authority?", "Sacral Authority")
    assert asdict(request) == {
        "original_query": "什麼是薦骨權威？",
        "dense_query": "What is Sacral Authority?",
        "sparse_query": "Sacral Authority",
    }


@pytest.mark.parametrize("name", ["original_query", "dense_query", "sparse_query"])
def test_hybrid_request_rejects_non_string_queries(name: str) -> None:
    values = dict.fromkeys(("original_query", "dense_query", "sparse_query"), "query")
    values[name] = None
    with pytest.raises(TypeError, match=name):
        HybridSearchRequest(**values)


@pytest.mark.parametrize("provenance", [
    {}, {"dense_rank": 1, "dense_score": 0.8},
    {"sparse_rank": 2, "sparse_score": 4.2},
    {"dense_rank": 3, "dense_score": 0.0, "sparse_rank": 1, "sparse_score": 7.0},
    {"dense_rank": 1}, {"sparse_rank": 1},
])
def test_optional_retriever_provenance_has_no_sentinels(provenance: dict) -> None:
    chunk = _chunk(**provenance)
    for name in ("dense_rank", "dense_score", "sparse_rank", "sparse_score",
                 "rrf_score", "rerank_score"):
        assert getattr(chunk, name) == provenance.get(name)
    for name in ("source_relpath", "document_title", "page_label", "page_number"):
        assert getattr(chunk, name) is None


@pytest.mark.parametrize("rank", [0, -1, 1.5, "1", True, False])
@pytest.mark.parametrize("name", ["dense_rank", "sparse_rank"])
def test_rank_requires_positive_non_boolean_integer(name: str, rank: object) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        _chunk(**{name: rank})


@pytest.mark.parametrize("name", ["dense_score", "sparse_score", "rrf_score", "rerank_score"])
@pytest.mark.parametrize("score", [True, "0.5", float("nan"), float("inf")])
def test_scores_must_be_finite_numbers_when_present(name: str, score: object) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        _chunk(**{name: score})


def test_fused_result_preserves_order_and_index_provenance() -> None:
    candidates = [_chunk(chunk_id="second", sparse_rank=1, rrf_score=0.02),
                  _chunk(chunk_id="first", dense_rank=2, rrf_score=0.01)]
    result = _retrieval(candidates=candidates)
    candidates.clear()
    assert tuple(item.chunk_id for item in result.candidates) == ("second", "first")
    assert result.ingestion_version == "v1"
    assert result.ingestion_id == "corpus-identity"
    assert (result.dense_top_k, result.sparse_top_k, result.fusion_top_k,
            result.final_top_k, result.rrf_k) == (20, 20, 20, 4, 60)
    assert result.rerank_provider is None
    assert result.rerank_model is None
    reranked = replace(result, rerank_provider="cohere", rerank_model="fixture-rerank")
    assert reranked.rerank_provider == "cohere"


def test_fused_result_rejects_unfused_candidates() -> None:
    with pytest.raises(ValueError, match="rrf_score"):
        _retrieval(candidates=(_chunk(dense_rank=1),))


def test_metadata_is_json_safe_and_copied() -> None:
    metadata = {"chapter": "Gates", "details": {"pages": [1, None, True, 2.5]}}
    chunk = _chunk(metadata=metadata)
    assert json.loads(json.dumps(asdict(chunk), allow_nan=False))["metadata"] == metadata
    metadata["details"]["pages"].append(3)
    assert chunk.metadata["details"]["pages"] == [1, None, True, 2.5]
    first, second = _chunk(), _chunk()
    assert first.metadata == second.metadata == {}
    assert first.metadata is not second.metadata


@pytest.mark.parametrize("metadata", [
    [], {1: "non-string key"}, {"bad": Path("reference.pdf")},
    {"bad": {1, 2}}, {"bad": object()}, {"bad": float("nan")},
    {"nested": {"bad": float("inf")}}, {"bad": (1, 2)},
])
def test_metadata_rejects_non_json_values(metadata: object) -> None:
    with pytest.raises((TypeError, ValueError), match="metadata"):
        _chunk(metadata=metadata)


@pytest.mark.parametrize("value", ["Sacral", 42, 0.5, True, None])
def test_chart_fact_accepts_json_scalars(value: object) -> None:
    assert _fact(value=value).value == value


@pytest.mark.parametrize("value", [{}, [], (42,), object(), float("nan"), float("inf")])
def test_chart_fact_rejects_containers_and_non_json_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="value"):
        _fact(value=value)


@pytest.mark.parametrize("path", ["", "  ", "/private/chart.png", "C:\\private\\chart.png"])
def test_chart_fact_requires_a_model_field_path(path: str) -> None:
    with pytest.raises(ValueError, match="source_path"):
        _fact(source_path=path)


def test_chart_context_rejects_ambiguous_fact_paths() -> None:
    with pytest.raises(ValueError, match="source_path"):
        ChartContext(facts=(_fact(), _fact(value=43)))
    facts = [_fact(), _fact(value=43, source_path="derived_chart_data.active_gates[gate=43]")]
    context = ChartContext(facts=facts, validation_warnings=["SAFE_WARNING"])
    facts.clear()
    assert len(context.facts) == 2
    assert context.validation_warnings == ("SAFE_WARNING",)


def test_prompt_contains_only_question_selected_facts_and_final_sources() -> None:
    source = PromptSource(citation_id="S1", chunk=_chunk(source_relpath="books/reference.pdf"))
    context = PromptContext(user_question="What is Gate 42?", chart_facts=[_fact()], sources=[source])
    assert {item.name for item in fields(context)} == {"user_question", "chart_facts", "sources"}
    assert context.chart_facts == (_fact(),)
    assert context.sources == (source,)
    assert source.chunk.source_relpath == "books/reference.pdf"


@pytest.mark.parametrize("path", [
    "/private/reference.pdf", "C:\\private\\reference.pdf", "\\\\host\\share\\book.pdf",
    "file:///private/reference.pdf",
])
@pytest.mark.parametrize("name", ["source_file", "document_title", "page_label"])
def test_citations_reject_absolute_paths(name: str, path: str) -> None:
    values = {"citation_id": "S1", "chunk_id": "chunk-42", "source_file": "reference.pdf"}
    with pytest.raises(ValueError, match=name):
        SourceCitation(**(values | {name: path}))


@pytest.mark.parametrize("path", ["/private/book.pdf", "C:\\private\\book.pdf", "../book.pdf"])
def test_source_relpath_must_be_safe_relative_provenance(path: str) -> None:
    with pytest.raises(ValueError, match="source_relpath"):
        _chunk(source_relpath=path)


def test_prompt_source_rejects_internal_absolute_metadata() -> None:
    chunk = _chunk(metadata={"local": {"source_path": "/private/reference.pdf"}})
    with pytest.raises(ValueError, match="absolute|path"):
        PromptSource(citation_id="S1", chunk=chunk)


@pytest.mark.parametrize("path", [
    "/private/reference.pdf", "C:\\private\\reference.pdf", "\\\\host\\share\\book.pdf",
    "file:///private/reference.pdf",
])
def test_provenance_rejects_paths_but_prompt_data_accepts_path_like_text(path: str) -> None:
    text = f"Local source ({path})"
    with pytest.raises(ValueError, match="document_title"):
        SourceCitation("S1", "chunk-42", "reference.pdf", document_title=text)
    source = PromptSource("S1", _chunk(text=text))
    context = PromptContext(user_question=text, sources=(source,))
    assert context.user_question == context.sources[0].chunk.text == text
    with pytest.raises(ValueError, match="value"):
        _fact(value=text)


def test_path_guard_accepts_profiles_urls_and_normal_prose() -> None:
    fact = _fact(value="1/3", source_path="derived_chart_data.basic_info.profile")
    text = "The 1/3 profile. Reference: https://example.com/guide/profile"
    prompt = PromptContext(text, chart_facts=(fact,), sources=(PromptSource("S1", _chunk(text=text)),))
    assert prompt.chart_facts == (fact,)


def test_citation_and_answer_exclude_internal_provenance() -> None:
    citation = SourceCitation("S1", "chunk-42", "reference.pdf", "Synthetic Reference", "7", 7)
    assert {item.name for item in fields(citation)} == {
        "citation_id", "chunk_id", "source_file", "document_title", "page_label", "page_number",
    }
    answer = AnswerResult(
        status=AnswerStatus.OK, answer_markdown="Synthetic explanation [S1].",
        citations=[citation], chart_facts_used=[_fact()], warnings=["SAFE_WARNING"],
    )
    output = json.loads(json.dumps(asdict(answer), allow_nan=False))
    assert set(output) == {"status", "answer_markdown", "citations", "chart_facts_used", "warnings"}
    assert output["status"] == "ok"
    assert output["citations"][0] == asdict(citation)
    assert output["chart_facts_used"][0]["source_path"] == _fact().source_path
    assert answer.citations == (citation,)
    assert answer.chart_facts_used == (_fact(),)
    assert answer.warnings == ("SAFE_WARNING",)


def test_answer_status_has_exactly_the_six_contract_values() -> None:
    expected = {"OK": "ok", "INSUFFICIENT_EVIDENCE": "insufficient_evidence",
                "INVALID_CHART": "invalid_chart", "NEEDS_FOCUS": "needs_focus",
                "REFUSED": "refused", "ERROR": "error"}
    assert {status.name: status.value for status in AnswerStatus} == expected
    for status in AnswerStatus:
        answer = AnswerResult(status=status, answer_markdown="")
        assert json.loads(json.dumps(asdict(answer)))["status"] == expected[status.name]


def test_empty_collection_defaults_and_model_immutability() -> None:
    for model in (ChartContext,):
        assert model().facts == model().validation_warnings == ()
    first = AnswerResult(status=AnswerStatus.NEEDS_FOCUS, answer_markdown="Focus the question.")
    second = AnswerResult(status=AnswerStatus.ERROR, answer_markdown="")
    assert first.citations == second.citations == ()
    assert first.chart_facts_used == second.chart_facts_used == ()
    assert first.warnings == second.warnings == ()
    assert PromptContext(user_question="Question").chart_facts == ()
    assert PromptContext(user_question="Question").sources == ()
    with pytest.raises(FrozenInstanceError):
        first.status = AnswerStatus.OK
