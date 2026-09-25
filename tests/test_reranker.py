"""Reranking contracts with fake clients only; no optional dependency required."""

import builtins
import importlib
import sys
import traceback
from dataclasses import replace
from types import SimpleNamespace

import pytest

from human_design.rag.config import load_config
from human_design.rag.models import HybridSearchRequest, RetrievedChunk
from human_design.rag.reranker import CohereReranker, NoOpReranker, create_reranker


def candidates():
    return [RetrievedChunk(
        f"canonical-{i}", f"Synthetic complete passage {i}", "reference.pdf", page_number=i + 1,
        dense_rank=i + 1, dense_score=0.4 - i, sparse_rank=3 - i, sparse_score=40 + i,
        rrf_score=1 / (61 + i) + 1 / (63 - i), metadata={"chunk_id": f"canonical-{i}"},
    ) for i in range(3)]


def settings(**changes):
    return replace(load_config(env={}), rerank_provider="cohere", real_rerank_api=True,
                   rerank_model="fake-rerank-model", cohere_api_key="fake-secret-key", **changes)


class FakeClient:
    def __init__(self, results=(), error=None):
        self.results, self.error, self.calls = list(results), error, []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(results=self.results)


def result(index, score=0.001):
    return SimpleNamespace(index=index, relevance_score=score)


def block_optional_import(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "cohere" or name.startswith("cohere.") or "cohere_rerank" in name:
            raise AssertionError("Cohere must not be imported")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


def test_none_imports_nothing_and_preserves_exact_order(monkeypatch):
    block_optional_import(monkeypatch)
    import human_design.rag.reranker as module
    importlib.reload(module)
    chunks = candidates()
    reranker = module.create_reranker(load_config(env={}))
    assert type(reranker).__name__ == "NoOpReranker"
    actual = reranker.rerank("dense query", chunks, 2)
    assert actual == chunks[:2]
    assert all(a is b for a, b in zip(actual, chunks))
    assert chunks == candidates()


def test_cohere_receives_only_semantic_query_text_and_required_parameters():
    chunks = candidates()
    request = HybridSearchRequest("original", "semantic dense question", "sparse exact terms")
    client = FakeClient([result(2, 0.001), result(0, 0.0001), result(1, 0.9)])
    actual = CohereReranker(settings(), client=client).rerank(request.dense_query, chunks, 2)
    assert client.calls == [{
        "model": "fake-rerank-model", "query": request.dense_query,
        "documents": [chunk.text for chunk in chunks], "top_n": 2,
    }]
    assert actual == [replace(chunks[2], rerank_score=0.001), replace(chunks[0], rerank_score=0.0001)]
    assert request.sparse_query not in repr(client.calls)
    assert chunks == candidates()  # frozen provenance was not overwritten


@pytest.mark.parametrize("kind", ["none", "cohere"])
def test_output_respects_configured_final_top_k(kind):
    chunks = candidates()
    client = FakeClient([result(2), result(1), result(0)])
    config = replace(settings(), rerank_provider=kind, final_top_k=1)
    reranker = create_reranker(config, client=client)
    assert len(reranker.rerank("dense", chunks, 99)) == 1
    if kind == "cohere":
        assert client.calls[0]["top_n"] == 1
    else:
        assert not client.calls


@pytest.mark.parametrize("kind", ["none", "cohere"])
def test_empty_input_never_calls_provider(kind):
    client = FakeClient(error=AssertionError("must not call"))
    reranker = create_reranker(replace(settings(), rerank_provider=kind), client=client)
    assert reranker.rerank("dense", [], 4) == []
    assert not client.calls


@pytest.mark.parametrize("top_n", [0, -1, True, 1.5])
@pytest.mark.parametrize("kind", ["none", "cohere"])
def test_invalid_top_n_fails_before_call(kind, top_n):
    client = FakeClient()
    reranker = create_reranker(replace(settings(), rerank_provider=kind), client=client)
    with pytest.raises(ValueError, match="positive integer"):
        reranker.rerank("dense", candidates(), top_n)
    assert not client.calls


@pytest.mark.parametrize("indexes", [[-1], [3], [99], [True], [1.5], ["canonical-0"], [None], [0, 0], [2, 1, 1]])
def test_invalid_duplicate_and_unknown_indexes_rejected_even_after_top_n(indexes):
    client = FakeClient([result(index) for index in indexes])
    with pytest.raises(ValueError, match="index"):
        CohereReranker(settings(), client=client).rerank("dense", candidates(), 1)


@pytest.mark.parametrize("entry", [SimpleNamespace(), SimpleNamespace(index=0), result(0, float("nan")), result(0, "secret")])
def test_malformed_provider_results_are_safe(entry):
    with pytest.raises(ValueError, match="Cohere") as caught:
        CohereReranker(settings(), client=FakeClient([entry])).rerank("dense", candidates(), 2)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("change,message", [
    ({"rerank_provider": "none"}, "cohere"),
    ({"real_rerank_api": False}, "HD_RAG_REAL_RERANK_API"),
    ({"rerank_model": None}, "HD_RAG_RERANK_MODEL"),
    ({"rerank_model": " "}, "HD_RAG_RERANK_MODEL"),
    ({"cohere_api_key": None}, "COHERE_API_KEY"),
    ({"cohere_api_key": " "}, "COHERE_API_KEY"),
])
def test_gates_before_optional_import_or_provider_construction(monkeypatch, change, message):
    block_optional_import(monkeypatch)
    with pytest.raises(ValueError, match=message):
        CohereReranker(replace(settings(), **change))


def test_unknown_provider_rejected_without_import(monkeypatch):
    block_optional_import(monkeypatch)
    with pytest.raises(ValueError, match="none or cohere"):
        create_reranker(replace(settings(), rerank_provider="other"))


def test_missing_optional_extra_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cohere", None)
    with pytest.raises(ImportError, match="uv sync --extra rerank"):
        create_reranker(settings())


def test_lazy_sdk_construction_uses_configured_key_only(monkeypatch):
    constructions = []
    client = FakeClient([result(0)])
    def constructor(**kwargs):
        constructions.append(kwargs)
        return client
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=constructor))
    reranker = create_reranker(settings())
    assert constructions == [{"api_key": "fake-secret-key"}]
    assert reranker.rerank("dense", candidates(), 1)[0].chunk_id == "canonical-0"
    assert "fake-secret-key" not in repr(reranker)


@pytest.mark.parametrize("during", ["construction", "call"])
def test_provider_exceptions_do_not_expose_key_or_passages(monkeypatch, caplog, capsys, during):
    secret = "fake-secret-key " + candidates()[0].text
    client = FakeClient(error=RuntimeError(secret))
    def constructor(**kwargs):
        raise RuntimeError(secret)
    if during == "construction":
        monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=constructor))
    with pytest.raises(ValueError, match="Cohere") as caught:
        reranker = create_reranker(settings()) if during == "construction" else CohereReranker(settings(), client=client)
        reranker.rerank("dense", candidates(), 2)
    output = "".join(traceback.format_exception(caught.value)) + caplog.text + capsys.readouterr().out
    assert "fake-secret-key" not in output
    assert candidates()[0].text not in output


def test_noop_larger_top_n_returns_all_available_candidates():
    assert NoOpReranker().rerank("dense", candidates(), 4) == candidates()
