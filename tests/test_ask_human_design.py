"""Minimal CLI contract, safe output, and explicit offline process-environment precedence."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from human_design.rag.config import load_config
from human_design.rag.models import RetrievedChunk
from human_design.reading import ReadingPipeline
from human_design.reading.models import (
    AnswerResult, AnswerStatus, ChartImageQuestionRequest, KnowledgeQuestionRequest, SourceCitation,
)
from human_design.vision.config import load_vision_config


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ask_human_design.py"
OFFLINE = {"HD_RAG_REAL_EMBEDDINGS": "0", "HD_RAG_REAL_GENERATION": "0",
           "HD_RAG_RERANK_PROVIDER": "none", "HD_RAG_REAL_RERANK_API": "0", "HD_VISION_REAL_API": "0"}
PRIVATE = "fake-key base64-private-marker birth-date-1901 /private/chart.png complete-private-passage"


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("ask_human_design_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answer(status=AnswerStatus.OK):
    if status is AnswerStatus.OK:
        return AnswerResult(status, "A reflective explanation [S1].",
                            citations=(SourceCitation("S1", "canonical-id", "reference.pdf", page_number=7),),
                            warnings=("Evidence is limited.",))
    return AnswerResult(status, "Please choose a focused topic or provide sufficient valid evidence.")


def fake_pipeline(result):
    return SimpleNamespace(answer_knowledge_question=Mock(return_value=result),
                           answer_chart_image_question=Mock(return_value=result),
                           private_payload=PRIVATE)


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("chart_mode", [False, True])
def test_modes_construct_only_intended_request_types(cli, json_mode, chart_mode, capsys):
    pipeline = fake_pipeline(answer())
    path = Path("/private/chart.png")
    argv = ["What is Sacral Authority?"] + (["--bodygraph", str(path)] if chart_mode else []) + (["--json"] if json_mode else [])
    assert cli.main(argv, pipeline=pipeline) == 0
    if chart_mode:
        pipeline.answer_chart_image_question.assert_called_once_with(ChartImageQuestionRequest(argv[0], path))
        pipeline.answer_knowledge_question.assert_not_called()
    else:
        pipeline.answer_knowledge_question.assert_called_once_with(KnowledgeQuestionRequest(argv[0]))
        pipeline.answer_chart_image_question.assert_not_called()
    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    if json_mode:
        payload = json.loads(stdout)
        assert payload == json.loads(json.dumps(asdict(answer())))
        assert payload["status"] == "ok" and "private_payload" not in payload
    else:
        for text in ("ok", "A reflective explanation [S1].", "reference.pdf", "7", "Evidence is limited."):
            assert text in stdout
    for value in PRIVATE.split():
        assert value not in stdout + stderr


@pytest.mark.parametrize("status", [AnswerStatus.INVALID_CHART, AnswerStatus.NEEDS_FOCUS, AnswerStatus.INSUFFICIENT_EVIDENCE])
@pytest.mark.parametrize("json_mode", [False, True])
def test_short_circuit_statuses_are_normal_results(cli, status, json_mode, capsys):
    assert cli.main(["Question"] + (["--json"] if json_mode else []), pipeline=fake_pipeline(answer(status))) == 0
    stdout, stderr = capsys.readouterr()
    assert not stderr and status.value in stdout and "Traceback" not in stdout


def test_cli_exposes_exact_flags_without_abbreviations(cli, capsys):
    parser = cli._build_parser()
    options = {flag for action in parser._actions for flag in action.option_strings}
    assert options == {"-h", "--help", "--bodygraph", "--json"}
    assert [a.dest for a in parser._actions if not a.option_strings] == ["query"]
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--bodygraph" in help_text and "--json" in help_text
    assert "--mock" not in help_text and "--dump" not in help_text


@pytest.mark.parametrize("flag", ["--mock-retrieval", "--mock-generation", "--provider-debug", "--dump-prompt",
                                 "--dump-chunks", "--model", "--provider", "--bod", "--js"])
def test_unsupported_arguments_do_not_echo_values(cli, flag, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["Question", flag, PRIVATE])
    assert caught.value.code == 2
    output = str(capsys.readouterr())
    assert "--help" in output
    assert all(value not in output for value in PRIVATE.split())


@pytest.mark.parametrize("error", [RuntimeError(PRIVATE), ValueError(PRIVATE), OSError(PRIVATE)])
def test_unexpected_errors_never_print_private_provider_details(cli, error, capsys):
    pipeline = fake_pipeline(answer())
    pipeline.answer_knowledge_question.side_effect = error
    assert cli.main(["Question"], pipeline=pipeline) == 1
    stdout, stderr = capsys.readouterr()
    assert not stdout and "failed" in stderr.lower() and "Traceback" not in stderr
    assert all(value not in stderr for value in PRIVATE.split())


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("failure,code", [
    ("request", "provider_request"), ("json", "structured_output"),
    ("incomplete", "incomplete_response"), ("citation", "unknown_citation"),
    ("chart_fact", "unknown_chart_fact"), ("private", "private_output"),
])
def test_real_generator_diagnostics_reach_cli_without_private_output(cli, failure, code, json_mode, monkeypatch, capsys):
    config = replace(load_config(env={}), real_generation=True,
                     generation_model="fake-model", openai_api_key="fake-key")
    payload = {"answer_markdown": "Reflective explanation [S1].", "used_source_ids": ["S1"],
               "used_chart_fact_paths": [], "limitations": []}
    if failure == "citation":
        payload["answer_markdown"] = "Explanation [S99]."
    elif failure == "chart_fact":
        payload["used_chart_fact_paths"] = [PRIVATE]
    elif failure == "private":
        payload["answer_markdown"] = PRIVATE + " [S1]"
    create = Mock(side_effect=RuntimeError(PRIVATE) if failure == "request" else None,
                  return_value=SimpleNamespace(status="incomplete" if failure == "incomplete" else "completed",
                      output_text=PRIVATE if failure == "json" else json.dumps(payload)))
    client = SimpleNamespace(responses=SimpleNamespace(create=create), close=Mock())
    constructor = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))

    def retriever(backend):
        return SimpleNamespace(ingestion_id="same", retrieve=lambda *args: [RetrievedChunk(
            "canonical", "complete-private-passage", "reference.pdf", **{backend + "_rank": 1})])

    pipeline = ReadingPipeline(config=config, dense_retriever=retriever("dense"), sparse_retriever=retriever("sparse"))
    assert cli.main(["Gate 42"] + (["--json"] if json_mode else []), pipeline=pipeline) == 2
    stdout, stderr = capsys.readouterr()
    assert stdout == "" and f"[{code}]" in stderr and "Traceback" not in stderr
    for value in PRIVATE.split():
        assert value not in stderr
    constructor.assert_called_once()
    assert constructor.call_args.kwargs["max_retries"] == 0
    create.assert_called_once()
    assert create.call_args.kwargs["store"] is False
    assert create.call_args.kwargs["input"] not in stderr
    client.close.assert_called_once()


def test_invalid_question_returns_safe_usage_error(cli, capsys):
    assert cli.main([" "], pipeline=ReadingPipeline()) == 2
    stdout, stderr = capsys.readouterr()
    assert not stdout and "Question" in stderr and "Traceback" not in stderr


def test_broad_cli_request_does_not_construct_dependencies(cli, monkeypatch, capsys):
    forbidden = Mock(side_effect=AssertionError("broad request must stop first"))
    for name in ("load_config", "build_chart_context", "build_retrieval_queries", "load_dense_retriever",
                 "load_bm25_retriever", "load_vision_config", "create_reranker", "generate_answer"):
        monkeypatch.setattr("human_design.reading.pipeline." + name, forbidden)
    assert cli.main(["Read my entire chart", "--bodygraph", "/private/not-read.png", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "needs_focus"
    forbidden.assert_not_called()


@pytest.mark.parametrize("provider,expected", [("embeddings", "HD_RAG_REAL_EMBEDDINGS"),
    ("generation", "HD_RAG_REAL_GENERATION"), ("vision", "HD_VISION_REAL_API"), ("cohere", "HD_RAG_REAL_RERANK_API")])
def test_disabled_provider_errors_name_opt_in_without_secret_values(cli, provider, expected, monkeypatch, capsys):
    config = load_config(env={})
    def retriever(backend):
        return SimpleNamespace(ingestion_id="same", retrieve=lambda *args: [RetrievedChunk(
            "canonical", "complete-private-passage", "reference.pdf", **{backend + "_rank": 1})])
    kwargs = {} if provider == "embeddings" else {
        "dense_retriever": retriever("dense"), "sparse_retriever": retriever("sparse")}
    if provider == "cohere":
        config = replace(config, rerank_provider="cohere", cohere_api_key="fake-key", rerank_model="fake-model")
    pipeline = ReadingPipeline(config=config, **kwargs)
    if provider == "vision":
        monkeypatch.setattr("human_design.reading.pipeline.load_vision_config", lambda: load_vision_config(env={}))
    argv = ["Gate 42"] + (["--bodygraph", "/private/not-read.png"] if provider == "vision" else [])
    assert cli.main(argv, pipeline=pipeline) == 2
    stdout, stderr = capsys.readouterr()
    assert not stdout and expected in stderr
    for value in ("fake-key", "complete-private-passage", "/private/not-read.png"):
        assert value not in stderr


def opt_in_dotenv(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=fake-key\nCOHERE_API_KEY=fake-cohere-key\nHD_RAG_REAL_EMBEDDINGS=1\n"
        "HD_RAG_REAL_GENERATION=1\nHD_RAG_RERANK_PROVIDER=cohere\nHD_RAG_REAL_RERANK_API=1\n"
        "HD_VISION_REAL_API=1\n", encoding="utf-8")


def test_explicit_process_flags_override_all_dotenv_opt_ins(cli, tmp_path, monkeypatch, capsys):
    opt_in_dotenv(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    for key, value in OFFLINE.items():
        monkeypatch.setenv(key, value)
    config = load_config()
    assert not config.real_embeddings and not config.real_generation and not config.real_rerank_api
    assert config.rerank_provider == "none"  # no Cohere model validation despite .env opt-in
    assert not load_vision_config().real_api_enabled
    assert cli.main(["Gate 42"]) == 2
    output = str(capsys.readouterr())
    assert "HD_RAG_REAL_EMBEDDINGS" in output and "fake-key" not in output
    assert not (tmp_path / "storage").exists()


@pytest.mark.parametrize("changes,required", [
    ({"HD_RAG_RERANK_PROVIDER": "cohere", "HD_RAG_REAL_RERANK_API": "0"}, "HD_RAG_REAL_RERANK_API"),
    ({"HD_RAG_REAL_GENERATION": "1", "HD_RAG_GENERATION_MODEL": "", "OPENAI_API_KEY": "fake-key"},
     "HD_RAG_GENERATION_MODEL"),
])
def test_configuration_gate_names_the_required_variable(cli, changes, required, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for key, value in (OFFLINE | changes).items():
        monkeypatch.setenv(key, value)
    assert cli.main(["Gate 42"]) == 2
    output = str(capsys.readouterr())
    assert required in output and "fake-key" not in output and str(tmp_path) not in output


@pytest.mark.parametrize("argv,code,expected", [
    (["--help"], 0, "--bodygraph"),
    (["Gate 42"], 2, "HD_RAG_REAL_EMBEDDINGS"),
    (["Profile", "--bodygraph", "/private/not-read.png"], 2, "HD_VISION_REAL_API"),
    (["Read my entire chart", "--json"], 0, '"needs_focus"'),
])
def test_safe_subprocess_from_temporary_cwd(argv, code, expected, tmp_path):
    assert SCRIPT.is_file()
    opt_in_dotenv(tmp_path)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HD_RAG_", "HD_VISION_")) and key not in ("OPENAI_API_KEY", "COHERE_API_KEY")}
    env.update(OFFLINE)
    process = subprocess.run([sys.executable, str(SCRIPT), *argv], cwd=tmp_path, env=env,
                             text=True, capture_output=True, timeout=30, check=False)
    assert process.returncode == code
    output = process.stdout + process.stderr
    assert expected in output and "Traceback" not in output
    for value in ("fake-key", "fake-cohere-key", "/private/not-read.png", str(tmp_path)):
        assert value not in output
    assert not (tmp_path / "storage").exists()
