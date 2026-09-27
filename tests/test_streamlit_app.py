"""Offline UI contracts using Streamlit's real AppTest and injected pipelines."""

import importlib.util
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from human_design.rag.config import load_config
from human_design.rag.models import RetrievedChunk
from human_design.rag.reranker import NoOpReranker
from human_design.reading.models import (
    AnswerResult, AnswerStatus, ChartFact, ChartImageQuestionRequest,
    KnowledgeQuestionRequest, SourceCitation,
)
from human_design.reading.pipeline import ReadingPipeline, ReadingPipelineError
from human_design.reading.prompt import build_source_citations
from human_design.reading.query_builder import InvalidQuestionError
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.models import BodyGraphExtractionResult, ParseResult
from human_design.vision.parser import parse_bodygraph_raw_extraction_json
from human_design.vision.validation import validate_bodygraph_extraction


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/streamlit_app.py"
OFFLINE = {
    "HD_RAG_REAL_EMBEDDINGS": "0", "HD_RAG_REAL_GENERATION": "0",
    "HD_RAG_RERANK_PROVIDER": "none", "HD_RAG_REAL_RERANK_API": "0",
    "HD_VISION_REAL_API": "0",
}
PRIVATE = "fake-key base64-private birth-date-1901 /private/chart.png full-private-passage"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    for key, value in OFFLINE.items():
        monkeypatch.setenv(key, value)
    for key in ("OPENAI_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    forbidden = Mock(side_effect=AssertionError("No providers or production storage in UI tests"))
    for name in ("load_dense_retriever", "load_bm25_retriever", "load_manifest",
                 "load_nodes", "extract_bodygraph", "create_reranker", "generate_answer"):
        monkeypatch.setattr("human_design.reading.pipeline." + name, forbidden)
    yield forbidden
    forbidden.assert_not_called()


def result(status=AnswerStatus.OK):
    if status is not AnswerStatus.OK:
        return AnswerResult(status, "Please choose a focused topic or check the available evidence.")
    return AnswerResult(
        status, "A reflective answer with a useful citation. [S1]",
        citations=(SourceCitation("S1", "canonical-test-id", "reference.pdf", page_number=7),),
        chart_facts_used=(ChartFact("authority", "Sacral", "derived_chart_data.basic_info.authority"),),
        warnings=("This is reflective information.",),
    )


@pytest.fixture
def fake(monkeypatch):
    service = SimpleNamespace(
        answer_knowledge_question=Mock(return_value=result()),
        answer_chart_image_question=Mock(return_value=result()),
    )
    constructor = Mock(return_value=service)
    monkeypatch.setattr("human_design.reading.ReadingPipeline", constructor)
    return service, constructor


def app():
    return AppTest.from_file(SCRIPT, default_timeout=10).run()


def app_module():
    spec = importlib.util.spec_from_file_location("web_app", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submit(at, query="How should I make decisions?"):
    at.text_area(key="query").set_value(query)
    return at.button(key="submit").click().run()


def visible(at):
    return "\n".join(str(element.value) for kind in
                     ("markdown", "text", "caption", "info", "warning", "error", "exception")
                     for element in at.get(kind))


def test_initial_load_and_editing_do_not_construct_a_pipeline(fake):
    service, constructor = fake
    at = app()
    assert not at.exception
    assert len(at.text_area) == len(at.file_uploader) == 1
    assert at.text_area[0].max_chars == 2000
    at.text_area[0].set_value("What is Authority?").run()
    at.file_uploader[0].set_value(("chart.png", b"synthetic-image", "image/png")).run()
    constructor.assert_not_called()
    service.answer_knowledge_question.assert_not_called()
    service.answer_chart_image_question.assert_not_called()


def test_knowledge_answer_renders_only_public_results_and_does_not_repeat_on_rerun(fake):
    service, constructor = fake
    at = submit(app(), "  What is Sacral Authority?  ")
    assert not at.exception and not at.error
    service.answer_knowledge_question.assert_called_once_with(
        KnowledgeQuestionRequest("What is Sacral Authority?"))
    service.answer_chart_image_question.assert_not_called()
    text = visible(at)
    for expected in ("A reflective answer", "[S1]", "reference.pdf", "7", "Sacral", "reflective information"):
        assert expected in text
    for forbidden in ("canonical-test-id", "derived_chart_data", "dense_score", "rrf_score"):
        assert forbidden not in text
    at.run()
    constructor.assert_called_once()
    service.answer_knowledge_question.assert_called_once()
    assert "A reflective answer" in visible(at)


@pytest.mark.parametrize("failure", [False, True])
def test_chart_upload_uses_safe_temporary_path_and_always_cleans_it(fake, tmp_path, failure):
    service, _ = fake
    paths = []

    def answer_image(request, *, on_validated_chart=None):
        assert isinstance(request, ChartImageQuestionRequest)
        path = request.bodygraph_image
        paths.append(path)
        assert path.parent.parent == tmp_path
        assert path.name == "bodygraph.png"
        assert path.read_bytes() == b"synthetic-image"
        if failure:
            raise RuntimeError(PRIVATE)
        return result()

    service.answer_chart_image_question.side_effect = answer_image
    at = app()
    at.file_uploader[0].set_value(("../../private-name.png", b"synthetic-image", "image/png"))
    submit(at)
    assert not at.exception
    service.answer_chart_image_question.assert_called_once()
    service.answer_knowledge_question.assert_not_called()
    assert paths and not paths[0].exists() and not paths[0].parent.exists()
    assert bool(at.error) is failure
    text = visible(at)
    for value in (*PRIVATE.split(), str(paths[0]), "private-name"):
        assert value not in text


@pytest.mark.parametrize("query", ["", " \n "], ids=["empty", "blank"])
def test_invalid_query_stops_before_pipeline_or_upload_processing(fake, query):
    service, constructor = fake
    at = app()
    at.file_uploader[0].set_value(("chart.png", b"synthetic-image", "image/png"))
    submit(at, query)
    assert at.error and not at.exception
    constructor.assert_not_called()
    service.answer_chart_image_question.assert_not_called()


def test_query_limit_is_enforced_before_reading_upload_even_without_widget_limit(fake):
    _, constructor = fake
    upload = SimpleNamespace(getvalue=Mock(side_effect=AssertionError("must not read")))
    with pytest.raises(InvalidQuestionError):
        app_module()._answer("界" * 2001, upload)
    upload.getvalue.assert_not_called()
    constructor.assert_not_called()


def test_exactly_2000_characters_can_be_submitted(fake):
    service, _ = fake
    at = submit(app(), "界" * 2000)
    assert not at.exception and not at.error
    service.answer_knowledge_question.assert_called_once_with(KnowledgeQuestionRequest("界" * 2000))


@pytest.mark.parametrize("name,size", [
    ("private.txt", 10), ("chart.png", 0), ("chart.png", 20 * 1024 * 1024 + 1),
], ids=["unsupported-type", "empty-file", "over-limit"])
def test_invalid_upload_stops_before_reading_bytes_or_constructing_pipeline(fake, name, size):
    _, constructor = fake
    upload = SimpleNamespace(name=name, size=size,
                             getvalue=Mock(side_effect=AssertionError("must not read")))
    module = app_module()
    with pytest.raises(module.InvalidUploadError) as caught:
        module._answer("What is Authority?", upload)
    assert name not in str(caught.value)
    upload.getvalue.assert_not_called()
    constructor.assert_not_called()


@pytest.mark.parametrize("query", [
    "Give me a complete reading of my chart.", "完整解讀我的人類圖",
])
@pytest.mark.parametrize("with_image", [False, True])
def test_broad_question_uses_existing_guard_without_providers(query, with_image):
    at = app()
    if with_image:
        at.file_uploader[0].set_value(("chart.png", b"synthetic-image", "image/png"))
    submit(at, query)
    assert not at.exception and not at.error
    assert at.info and "specific topic" in visible(at)


@pytest.mark.parametrize("status", [
    AnswerStatus.INVALID_CHART, AnswerStatus.INSUFFICIENT_EVIDENCE, AnswerStatus.NEEDS_FOCUS,
])
def test_normal_non_answer_results_are_shown_without_tracebacks(fake, status):
    service, _ = fake
    service.answer_knowledge_question.return_value = result(status)
    at = submit(app())
    assert not at.exception and not at.error
    assert at.info and "focused topic" in visible(at)
    assert "reference.pdf" not in visible(at)


def test_new_failure_clears_the_previous_answer_and_hides_private_error(fake, caplog):
    service, _ = fake
    at = submit(app())
    service.answer_knowledge_question.side_effect = RuntimeError(PRIVATE)
    submit(at, "What is Gate 42?")
    assert not at.exception and at.error
    text = visible(at) + caplog.text
    assert "A reflective answer" not in text
    assert all(value not in text for value in PRIVATE.split())


def test_safe_operational_error_keeps_actionable_configuration_hint(fake):
    service, _ = fake
    service.answer_knowledge_question.side_effect = ReadingPipelineError(
        "Enable HD_RAG_REAL_GENERATION=1 for real generation")
    at = submit(app())
    assert at.error and not at.exception
    assert "HD_RAG_REAL_GENERATION" in visible(at)


@pytest.mark.parametrize("with_image,variable", [
    (False, "HD_RAG_REAL_EMBEDDINGS"), (True, "HD_VISION_REAL_API"),
])
def test_process_offline_flags_override_dotenv_opt_ins(monkeypatch, tmp_path, with_image, variable):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=fake-key\nCOHERE_API_KEY=fake-cohere-key\n"
        "HD_RAG_REAL_EMBEDDINGS=1\nHD_RAG_REAL_GENERATION=1\n"
        "HD_RAG_RERANK_PROVIDER=cohere\nHD_RAG_REAL_RERANK_API=1\nHD_VISION_REAL_API=1\n")
    monkeypatch.chdir(tmp_path)
    at = app()
    if with_image:
        at.file_uploader[0].set_value(("chart.png", b"synthetic-image", "image/png"))
    submit(at)
    assert at.error and not at.exception
    assert variable in visible(at)
    assert "fake-key" not in visible(at) and "fake-cohere-key" not in visible(at)


def test_answer_markdown_never_enables_raw_html(fake):
    service, _ = fake
    service.answer_knowledge_question.return_value = AnswerResult(
        AnswerStatus.OK, '<img src="x" onerror="alert(1)">Reflective answer [S1]',
        citations=result().citations,
    )
    at = submit(app())
    assert not at.exception
    assert all(not element.proto.allow_html for element in at.markdown)


@pytest.fixture
def chart_flow(monkeypatch):
    raw = (Path(__file__).parent / "fixtures/bodygraph/test1_raw_response.json").read_text()
    parsed = parse_bodygraph_raw_extraction_json(raw)
    derived = interpret_bodygraph(parsed.raw_vision)
    chart = BodyGraphExtractionResult(parsed.raw_vision, derived.derived_chart_data,
        validate_bodygraph_extraction(parse_result=parsed, interpretation_result=derived))
    assert chart.validation_result.is_valid
    extractor = Mock(return_value=chart)

    def retriever(backend):
        return SimpleNamespace(ingestion_id="same-test-ingestion", retrieve=Mock(return_value=[
            RetrievedChunk("test-chunk", "Synthetic reference passage.", "reference.pdf",
                           **{f"{backend}_rank": 1, f"{backend}_score": 0.7})]))

    def generate(prompt, config):
        return AnswerResult(AnswerStatus.OK, "A reflective answer. [S1]",
                            citations=build_source_citations(prompt), chart_facts_used=prompt.chart_facts)

    dense, sparse = retriever("dense"), retriever("sparse")
    generator = Mock(side_effect=generate)
    service = ReadingPipeline(config=load_config(env={}), extractor=extractor,
        dense_retriever=dense, sparse_retriever=sparse, reranker=NoOpReranker(), generator=generator)
    monkeypatch.setattr("human_design.reading.ReadingPipeline", Mock(return_value=service))
    return SimpleNamespace(chart=chart, extractor=extractor, generator=generator,
                           dense=dense, sparse=sparse)


def test_same_image_reuses_validated_chart_but_reselects_facts_and_answers(chart_flow, monkeypatch):
    at = app()
    at.file_uploader[0].set_value(("chart.png", b"synthetic-image", "image/png"))
    submit(at)
    assert not at.error and not at.exception
    first_prompt = chart_flow.generator.call_args.args[0]
    assert {fact.field for fact in first_prompt.chart_facts} == {"type", "authority", "strategy"}
    # A cache hit must use the typed chart core without creating another image file.
    monkeypatch.setattr("tempfile.TemporaryDirectory", Mock(side_effect=AssertionError("no image copy")))
    submit(at, "Explain my Profile")
    assert not at.error and not at.exception
    chart_flow.extractor.assert_called_once()
    assert chart_flow.generator.call_count == 2
    assert chart_flow.dense.retrieve.call_count == chart_flow.sparse.retrieve.call_count == 2
    assert [f.field for f in chart_flow.generator.call_args.args[0].chart_facts] == ["profile"]
    digest = sha256(b"synthetic-image").hexdigest()
    assert at.session_state["_validated_charts"] == {digest: chart_flow.chart}
    assert at.session_state["_validated_charts"][digest] is chart_flow.chart
    text = visible(at)
    for private in (digest, "Synthetic reference passage.", "raw_vision", "derived_chart_data"):
        assert private not in text


def test_image_cache_uses_content_not_filename_and_can_revisit_an_image(chart_flow):
    at = app()
    uploads = [
        ("chart.png", b"image-a", 1),
        ("renamed.png", b"image-a", 1),
        ("renamed.png", b"image-b", 2),
        ("chart.png", b"image-a", 2),
    ]
    for name, content, expected_extractions in uploads:
        at.file_uploader[0].set_value((name, content, "image/png"))
        submit(at)
        assert not at.error and not at.exception
        assert chart_flow.extractor.call_count == expected_extractions
    assert chart_flow.generator.call_count == 4


def test_chart_cache_is_not_shared_between_browser_sessions(chart_flow):
    for _ in range(2):
        at = app()
        at.file_uploader[0].set_value(("chart.png", b"same-image", "image/png"))
        submit(at)
        submit(at, "Explain my Profile")
        assert not at.error and not at.exception
    assert chart_flow.extractor.call_count == 2
    assert chart_flow.generator.call_count == 4


@pytest.mark.parametrize("failure", ["invalid-chart", "extraction-error"])
def test_failed_or_invalid_extractions_are_not_cached(chart_flow, failure, caplog):
    raw = replace(chart_flow.chart.raw_vision,
                  personality=replace(chart_flow.chart.raw_vision.personality, sun=None))
    derived = interpret_bodygraph(raw)
    invalid = BodyGraphExtractionResult(raw, derived.derived_chart_data,
        validate_bodygraph_extraction(parse_result=ParseResult(raw), interpretation_result=derived))
    assert not invalid.validation_result.is_valid
    chart_flow.extractor.side_effect = [
        invalid if failure == "invalid-chart" else RuntimeError(PRIVATE), chart_flow.chart,
    ]
    at = app()
    at.file_uploader[0].set_value(("chart.png", b"retry-image", "image/png"))
    submit(at)
    assert not at.exception
    assert "_validated_charts" not in at.session_state or not at.session_state["_validated_charts"]
    chart_flow.generator.assert_not_called()
    chart_flow.dense.retrieve.assert_not_called()
    chart_flow.sparse.retrieve.assert_not_called()
    if failure == "invalid-chart":
        assert at.session_state["answer"].status is AnswerStatus.INVALID_CHART
    else:
        assert at.error
    submit(at)
    submit(at, "Explain my Profile")
    assert not at.error and not at.exception
    assert chart_flow.extractor.call_count == 2
    assert chart_flow.generator.call_count == 2
    assert all(value not in visible(at) + caplog.text for value in PRIVATE.split())


def test_valid_extraction_remains_cached_when_later_generation_fails(chart_flow, caplog):
    generate = chart_flow.generator.side_effect

    def fail_once(prompt, config):
        if chart_flow.generator.call_count == 1:
            raise RuntimeError(PRIVATE)
        return generate(prompt, config)

    chart_flow.generator.side_effect = fail_once
    at = app()
    at.file_uploader[0].set_value(("chart.png", b"synthetic-image", "image/png"))
    submit(at)
    assert at.error and not at.exception
    submit(at, "Explain my Profile")
    assert not at.error and not at.exception
    chart_flow.extractor.assert_called_once()
    assert chart_flow.generator.call_count == 2
    assert all(value not in visible(at) + caplog.text for value in PRIVATE.split())


@pytest.mark.parametrize("query", ["", "Give me a complete reading of my chart."])
def test_cached_chart_does_not_bypass_query_or_scope_guards(chart_flow, query, monkeypatch):
    at = app()
    at.file_uploader[0].set_value(("chart.png", b"synthetic-image", "image/png"))
    submit(at)
    forbidden = Mock(side_effect=AssertionError("input guards must run first"))
    monkeypatch.setattr("human_design.reading.pipeline.build_chart_context", forbidden)
    submit(at, query)
    assert not at.exception
    forbidden.assert_not_called()
    chart_flow.extractor.assert_called_once()
    chart_flow.generator.assert_called_once()
    chart_flow.dense.retrieve.assert_called_once()
    chart_flow.sparse.retrieve.assert_called_once()
    if query:
        assert at.session_state["answer"].status is AnswerStatus.NEEDS_FOCUS
    else:
        assert at.error


def test_removing_upload_returns_to_knowledge_mode_without_cached_chart_facts(chart_flow):
    at = app()
    at.file_uploader[0].set_value(("chart.png", b"synthetic-image", "image/png"))
    submit(at)
    at.file_uploader[0].set_value(None)
    submit(at, "What is Sacral Authority?")
    assert not at.error and not at.exception
    chart_flow.extractor.assert_called_once()
    assert chart_flow.generator.call_count == 2
    assert chart_flow.generator.call_args.args[0].chart_facts == ()
