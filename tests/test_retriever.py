from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from human_design.rag.config import load_config
from human_design.rag import retriever


def _load_query_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "query_kb.py"
    spec = importlib.util.spec_from_file_location("query_kb_script", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load script spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path):
    return replace(
        load_config(env={}),
        chroma_dir=tmp_path / "chroma",
        collection_name="human_design_test",
        embedding_model="text-embedding-3-small",
        openai_api_key=None,
    )


def test_load_existing_chroma_index_uses_config_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_create_chroma_vector_store(**kwargs: object) -> str:
        calls.append(
            "vector_store:"
            f"{kwargs['chroma_dir']}:"
            f"{kwargs['collection_name']}:"
            f"{kwargs['embedding_model']}"
        )
        return "vector-store"

    def fake_create_embedding_model_from_config(received_config: object) -> str:
        assert received_config is config
        calls.append("embedding-model")
        return "embed-model"

    class FakeVectorStoreIndex:
        @classmethod
        def from_vector_store(
            cls,
            *,
            vector_store: object,
            embed_model: object,
        ) -> str:
            calls.append(f"from_vector_store:{vector_store}:{embed_model}")
            return "index"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("retriever reload must not construct an index from nodes")

    monkeypatch.setattr(retriever, "create_chroma_vector_store", fake_create_chroma_vector_store)
    monkeypatch.setattr(
        retriever,
        "create_openai_embedding_model_from_config",
        fake_create_embedding_model_from_config,
    )
    monkeypatch.setattr(retriever, "VectorStoreIndex", FakeVectorStoreIndex)

    index = retriever.load_existing_chroma_index(config)

    assert index == "index"
    assert calls == [
        f"vector_store:{config.chroma_dir}:human_design_test:text-embedding-3-small",
        "embedding-model",
        "from_vector_store:vector-store:embed-model",
    ]


def test_create_retriever_from_config_passes_similarity_top_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    class FakeIndex:
        def as_retriever(self, *, similarity_top_k: int) -> str:
            return f"retriever:{similarity_top_k}"

    monkeypatch.setattr(
        retriever,
        "load_existing_chroma_index",
        lambda received_config: FakeIndex(),
    )

    result = retriever.create_retriever_from_config(config, similarity_top_k=7)

    assert result == "retriever:7"


def test_embedding_model_mismatch_raises_rebuild_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    def fake_create_chroma_vector_store(**kwargs: object) -> object:
        raise ValueError("Embedding model mismatch")

    monkeypatch.setattr(retriever, "create_chroma_vector_store", fake_create_chroma_vector_store)

    with pytest.raises(ValueError, match="rebuild the Chroma collection"):
        retriever.load_existing_chroma_index(config)


def test_retriever_does_not_load_pdfs_or_chunk_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    from human_design.rag import chunking, ingestion

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("retriever reload must not ingest or chunk documents")

    monkeypatch.setattr(ingestion, "load_pdf", fail_if_called)
    monkeypatch.setattr(ingestion, "load_pdfs", fail_if_called)
    monkeypatch.setattr(ingestion, "build_text_extraction_report", fail_if_called)
    monkeypatch.setattr(chunking, "chunk_documents", fail_if_called)
    monkeypatch.setattr(retriever, "create_chroma_vector_store", lambda **kwargs: "vector-store")
    monkeypatch.setattr(
        retriever,
        "create_openai_embedding_model_from_config",
        lambda received_config: "embed-model",
    )

    class FakeVectorStoreIndex:
        @classmethod
        def from_vector_store(
            cls,
            *,
            vector_store: object,
            embed_model: object,
        ) -> str:
            return "index"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("retriever reload must not persist nodes")

    monkeypatch.setattr(retriever, "VectorStoreIndex", FakeVectorStoreIndex)

    assert retriever.load_existing_chroma_index(config) == "index"


def test_retriever_default_tests_do_not_require_openai_api_key_or_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(retriever, "create_chroma_vector_store", lambda **kwargs: "vector-store")
    monkeypatch.setattr(
        retriever,
        "create_openai_embedding_model_from_config",
        lambda received_config: "embed-model",
    )

    class FakeVectorStoreIndex:
        @classmethod
        def from_vector_store(
            cls,
            *,
            vector_store: object,
            embed_model: object,
        ) -> str:
            return "index"

    monkeypatch.setattr(retriever, "VectorStoreIndex", FakeVectorStoreIndex)

    assert retriever.load_existing_chroma_index(config) == "index"
    assert not (tmp_path / "storage" / "chroma").exists()


def test_retriever_does_not_append_or_delete_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(retriever, "create_chroma_vector_store", lambda **kwargs: "vector-store")
    monkeypatch.setattr(
        retriever,
        "create_openai_embedding_model_from_config",
        lambda received_config: "embed-model",
    )

    class FakeVectorStoreIndex:
        @classmethod
        def from_vector_store(
            cls,
            *,
            vector_store: object,
            embed_model: object,
        ) -> str:
            return "index"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("retriever reload must not append vectors")

        def delete_ref_doc(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("retriever reload must not delete vectors")

    monkeypatch.setattr(retriever, "VectorStoreIndex", FakeVectorStoreIndex)

    assert retriever.load_existing_chroma_index(config) == "index"


def test_query_script_requires_query_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    query_script = _load_query_script()

    with pytest.raises(SystemExit):
        query_script.main([])

    assert "usage:" in capsys.readouterr().err


def test_query_script_requires_real_retrieval_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_script = _load_query_script()
    monkeypatch.delenv("HD_RAG_REAL_EMBEDDINGS", raising=False)

    with pytest.raises(SystemExit, match="Real retrieval is disabled"):
        query_script.main(["Generator"])


def test_query_script_passes_query_and_top_k_to_retriever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_script = _load_query_script()
    config = _config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    calls: list[str] = []

    class FakeRetriever:
        def retrieve(self, query: str) -> list[object]:
            calls.append(f"retrieve:{query}")
            return []

    monkeypatch.setattr(query_script, "load_config", lambda: calls.append("load_config") or config)
    monkeypatch.setattr(
        query_script,
        "create_retriever_from_config",
        lambda received_config, similarity_top_k: calls.append(
            f"create_retriever:{received_config.collection_name}:{similarity_top_k}"
        )
        or FakeRetriever(),
    )

    query_script.main(["What is a Generator type?", "--top-k", "3"])

    assert calls == [
        "load_config",
        "create_retriever:human_design_test:3",
        "retrieve:What is a Generator type?",
    ]


def test_query_script_prints_rank_snippet_score_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    query_script = _load_query_script()
    config = _config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    long_text = "A" * 520 + "FULL_CHUNK_TAIL"

    class FakeNode:
        metadata = {
            "source_file": "book.pdf",
            "source_path": "data/pdfs/book.pdf",
            "page_label": "iv",
            "page_number": 4,
        }

        def get_content(self) -> str:
            return long_text

    class FakeRetriever:
        def retrieve(self, query: str) -> list[object]:
            return [SimpleNamespace(node=FakeNode(), score=0.87)]

    monkeypatch.setattr(query_script, "load_config", lambda: config)
    monkeypatch.setattr(
        query_script,
        "create_retriever_from_config",
        lambda received_config, similarity_top_k: FakeRetriever(),
    )

    query_script.main(["Generator", "--snippet-chars", "20"])

    output = capsys.readouterr().out
    assert "Result 1" in output
    assert "Score: 0.87" in output
    assert "Source file: book.pdf" in output
    assert "Source path: data/pdfs/book.pdf" in output
    assert "Page label: iv" in output
    assert "Page number: 4" in output
    assert "Snippet: AAAAAAAAAAAAAAAAAAAA" in output
    assert "FULL_CHUNK_TAIL" not in output


def test_query_script_truncates_long_chunks_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    query_script = _load_query_script()
    config = _config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    long_text = "x" * 520 + "TAIL_SHOULD_NOT_PRINT"

    class FakeRetriever:
        def retrieve(self, query: str) -> list[object]:
            node = SimpleNamespace(text=long_text, metadata={"source_file": "book.pdf"})
            return [SimpleNamespace(node=node, score=None)]

    monkeypatch.setattr(query_script, "load_config", lambda: config)
    monkeypatch.setattr(
        query_script,
        "create_retriever_from_config",
        lambda received_config, similarity_top_k: FakeRetriever(),
    )

    query_script.main(["Generator"])

    output = capsys.readouterr().out
    assert "x" * 500 in output
    assert "TAIL_SHOULD_NOT_PRINT" not in output


def test_query_script_prints_clear_message_when_no_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    query_script = _load_query_script()
    config = _config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")

    class FakeRetriever:
        def retrieve(self, query: str) -> list[object]:
            return []

    monkeypatch.setattr(query_script, "load_config", lambda: config)
    monkeypatch.setattr(
        query_script,
        "create_retriever_from_config",
        lambda received_config, similarity_top_k: FakeRetriever(),
    )

    query_script.main(["Generator"])

    assert "No results found." in capsys.readouterr().out


def test_query_script_shows_clear_message_when_retriever_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_script = _load_query_script()
    config = _config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    monkeypatch.setattr(query_script, "load_config", lambda: config)
    monkeypatch.setattr(
        query_script,
        "create_retriever_from_config",
        lambda received_config, similarity_top_k: (_ for _ in ()).throw(
            ValueError("Please rebuild the Chroma collection")
        ),
    )

    with pytest.raises(SystemExit, match="Run or rebuild ingestion"):
        query_script.main(["Generator"])


def test_query_script_does_not_load_pdfs_or_chunk_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_script = _load_query_script()
    config = _config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")

    from human_design.rag import chunking, ingestion

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("query script must not ingest or chunk documents")

    monkeypatch.setattr(ingestion, "load_pdf", fail_if_called)
    monkeypatch.setattr(ingestion, "load_pdfs", fail_if_called)
    monkeypatch.setattr(ingestion, "build_text_extraction_report", fail_if_called)
    monkeypatch.setattr(chunking, "chunk_documents", fail_if_called)
    monkeypatch.setattr(query_script, "load_config", lambda: config)
    monkeypatch.setattr(
        query_script,
        "create_retriever_from_config",
        lambda received_config, similarity_top_k: SimpleNamespace(
            retrieve=lambda query: []
        ),
    )

    query_script.main(["Generator"])


def test_query_script_does_not_require_openai_key_or_create_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_script = _load_query_script()
    config = _config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(query_script, "load_config", lambda: config)
    monkeypatch.setattr(
        query_script,
        "create_retriever_from_config",
        lambda received_config, similarity_top_k: SimpleNamespace(
            retrieve=lambda query: []
        ),
    )

    query_script.main(["Generator"])

    assert not (tmp_path / "storage" / "chroma").exists()
