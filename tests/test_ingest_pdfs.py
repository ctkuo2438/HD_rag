import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding

from human_design.rag import vector_store
from human_design.rag.ingestion import NoPdfFilesFoundError
from human_design.rag.models import ChunkingResult, TextExtractionReport


def _load_ingest_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "ingest_pdfs.py"
    spec = importlib.util.spec_from_file_location("ingest_pdfs_script", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load script spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        pdf_dir=tmp_path / "pdfs",
        chroma_dir=tmp_path / "chroma",
        collection_name="human_design_test",
        embedding_model="text-embedding-3-small",
        openai_api_key="test-api-key",
        chunk_size=800,
        chunk_overlap=80,
        ingestion_version="v1",
    )


def _fake_report(tmp_path: Path) -> TextExtractionReport:
    return TextExtractionReport(
        pdf_count=2,
        document_count=3,
        source_files=(tmp_path / "pdfs" / "book-a.pdf", tmp_path / "pdfs" / "book-b.pdf"),
        total_text_characters=1234,
        low_text_threshold=100,
        low_text_document_count=0,
        low_text_documents=(),
    )


@pytest.mark.parametrize("opt_in", [None, "0", "false", "true", "01", " 1"])
def test_main_exits_clearly_when_real_embeddings_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opt_in: str | None,
) -> None:
    ingest_script = _load_ingest_script()
    if opt_in is None:
        monkeypatch.delenv("HD_RAG_REAL_EMBEDDINGS", raising=False)
    else:
        monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", opt_in)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("pipeline should not run when real embeddings are disabled")

    monkeypatch.setattr(ingest_script, "load_config", fail_if_called)
    monkeypatch.setattr(ingest_script, "load_pdfs", fail_if_called)
    monkeypatch.setattr(ingest_script, "create_openai_embedding_model_from_config", fail_if_called)
    monkeypatch.setattr(ingest_script, "create_chroma_vector_store", fail_if_called)
    monkeypatch.setattr(ingest_script, "VectorStoreIndex", fail_if_called)
    monkeypatch.setattr(vector_store, "create_chroma_client", fail_if_called)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="Real embedding ingestion is disabled") as exc_info:
        ingest_script.main()

    assert exc_info.value.code == ingest_script.DISABLED_MESSAGE
    assert not (tmp_path / "storage" / "chroma").exists()


@pytest.mark.parametrize("stored_model", [None, "text-embedding-3-small", "other-model"])
def test_main_refuses_populated_collection_before_loading_or_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stored_model: str | None,
) -> None:
    ingest_script = _load_ingest_script()
    config = _fake_config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    monkeypatch.setattr(ingest_script, "load_config", lambda: config)
    client = vector_store.create_chroma_client(config.chroma_dir)
    metadata = {"synthetic_marker": "keep-existing-metadata"}
    if stored_model is not None:
        metadata[vector_store.EMBEDDING_MODEL_METADATA_KEY] = stored_model
    collection = client.create_collection(name=config.collection_name, metadata=metadata)
    collection.add(
        ids=["existing-chunk"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Synthetic existing document."],
        metadatas=[{"source_file": "synthetic.pdf"}],
    )
    before = collection.get(include=["documents", "metadatas", "embeddings"])
    forbidden_calls = []
    for name in (
        "load_pdfs", "chunk_documents", "create_openai_embedding_model_from_config",
        "create_chroma_vector_store", "VectorStoreIndex",
    ):
        forbidden = Mock(side_effect=AssertionError(f"{name} must not run for a populated target"))
        monkeypatch.setattr(ingest_script, name, forbidden)
        forbidden_calls.append(forbidden)

    with pytest.raises(SystemExit, match="refuses to append to a populated collection") as exc_info:
        ingest_script.main()

    message = str(exc_info.value)
    assert "duplicate content" in message
    assert "fresh/empty collection" in message
    assert str(tmp_path) not in message
    assert config.openai_api_key not in message
    for forbidden in forbidden_calls:
        forbidden.assert_not_called()
    reopened = client.get_collection(name=config.collection_name)
    after = reopened.get(include=["documents", "metadatas", "embeddings"])
    assert reopened.count() == 1
    assert reopened.metadata == metadata
    assert after.pop("embeddings").tolist() == before.pop("embeddings").tolist()
    assert after == before
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "initial_state", ["new_directory", "new_collection", "empty", "empty_unlabeled"],
)
def test_main_ingests_new_or_empty_target_then_refuses_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_state: str,
) -> None:
    ingest_script = _load_ingest_script()
    config = _fake_config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    monkeypatch.setattr(ingest_script, "load_config", lambda: config)
    if initial_state != "new_directory":
        client = vector_store.create_chroma_client(config.chroma_dir)
        if initial_state == "new_collection":
            unrelated = client.create_collection(name="unrelated_collection")
            unrelated.add(ids=["keep"], embeddings=[[0.1, 0.2, 0.3]])
        else:
            client.create_collection(
                name=config.collection_name,
                metadata={vector_store.EMBEDDING_MODEL_METADATA_KEY: config.embedding_model}
                if initial_state == "empty" else None,
            )
    documents = [Document(
        text="Synthetic PDF text for offline ingestion.",
        metadata={"file_name": "synthetic.pdf"},
    )]
    pdf_loader = Mock(return_value=documents)
    embedding_factory = Mock(return_value=MockEmbedding(embed_dim=3))
    monkeypatch.setattr(ingest_script, "load_pdfs", pdf_loader)
    monkeypatch.setattr(
        ingest_script, "build_text_extraction_report",
        lambda *args, **kwargs: _fake_report(tmp_path),
    )
    monkeypatch.setattr(ingest_script, "create_openai_embedding_model_from_config", embedding_factory)

    ingest_script.main()

    pdf_loader.assert_called_once_with(config.pdf_dir)
    embedding_factory.assert_called_once_with(config)
    client = vector_store.create_chroma_client(config.chroma_dir)
    collection = client.get_collection(name=config.collection_name)
    assert collection.count() == 1
    assert collection.metadata == {vector_store.EMBEDDING_MODEL_METADATA_KEY: config.embedding_model}
    saved = collection.get(include=["documents", "metadatas"])
    assert saved["documents"] == [documents[0].text]
    assert saved["metadatas"][0]["chunk_size"] == 800
    assert saved["metadatas"][0]["chunk_overlap"] == 80
    assert saved["metadatas"][0]["embedding_model"] == config.embedding_model
    if initial_state == "new_collection":
        assert client.get_collection(name="unrelated_collection").get()["ids"] == ["keep"]

    with pytest.raises(SystemExit, match="refuses to append to a populated collection"):
        ingest_script.main()

    pdf_loader.assert_called_once()
    embedding_factory.assert_called_once()
    assert collection.count() == 1
    assert collection.get(include=["documents", "metadatas"]) == saved


def test_main_preserves_empty_collection_embedding_model_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest_script = _load_ingest_script()
    config = _fake_config(tmp_path)
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    monkeypatch.setattr(ingest_script, "load_config", lambda: config)
    client = vector_store.create_chroma_client(config.chroma_dir)
    collection = client.create_collection(
        name=config.collection_name,
        metadata={vector_store.EMBEDDING_MODEL_METADATA_KEY: "other-model"},
    )
    monkeypatch.setattr(
        ingest_script, "load_pdfs", lambda _: [Document(text="Synthetic PDF text.")],
    )
    monkeypatch.setattr(
        ingest_script, "build_text_extraction_report",
        lambda *args, **kwargs: _fake_report(tmp_path),
    )
    monkeypatch.setattr(
        ingest_script, "create_openai_embedding_model_from_config",
        lambda _: MockEmbedding(embed_dim=3),
    )
    index = Mock(side_effect=AssertionError("mismatched collection must never be indexed"))
    monkeypatch.setattr(ingest_script, "VectorStoreIndex", index)

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        ingest_script.main()

    index.assert_not_called()
    assert collection.count() == 0
    assert collection.metadata == {vector_store.EMBEDDING_MODEL_METADATA_KEY: "other-model"}


def test_main_runs_pipeline_when_real_embeddings_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ingest_script = _load_ingest_script()
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    config = _fake_config(tmp_path)
    documents = [SimpleNamespace(text="full document text should not be printed")]
    nodes = [SimpleNamespace(text="full chunk text should not be printed")]
    calls: list[str] = []

    monkeypatch.setattr(
        ingest_script,
        "load_config",
        lambda: calls.append("load_config") or config,
    )
    monkeypatch.setattr(
        ingest_script,
        "load_pdfs",
        lambda pdf_dir: calls.append(f"load_pdfs:{pdf_dir}") or documents,
    )
    monkeypatch.setattr(
        ingest_script,
        "build_text_extraction_report",
        lambda pdf_dir, **kwargs: calls.append(
            "build_text_extraction_report:"
            f"{pdf_dir}:"
            f"{kwargs['documents'] is documents}"
        )
        or _fake_report(tmp_path),
    )
    monkeypatch.setattr(
        ingest_script,
        "chunk_documents",
        lambda docs, **kwargs: calls.append(
            "chunk_documents:"
            f"{kwargs['chunk_size']}:"
            f"{kwargs['chunk_overlap']}:"
            f"{kwargs['embedding_model']}:"
            f"{kwargs['ingestion_version']}"
        )
        or (
            nodes,
            ChunkingResult(
                document_count=len(docs),
                chunk_count=len(nodes),
                chunk_size=kwargs["chunk_size"],
                chunk_overlap=kwargs["chunk_overlap"],
            ),
        ),
    )
    monkeypatch.setattr(
        ingest_script,
        "create_openai_embedding_model_from_config",
        lambda cfg: calls.append(f"create_openai_embedding_model_from_config:{cfg.embedding_model}")
        or "embed-model",
    )
    monkeypatch.setattr(
        ingest_script,
        "create_chroma_vector_store",
        lambda **kwargs: calls.append(
            "create_chroma_vector_store:"
            f"{kwargs['chroma_dir']}:"
            f"{kwargs['collection_name']}:"
            f"{kwargs['embedding_model']}"
        )
        or "vector-store",
    )

    class FakeStorageContext:
        @staticmethod
        def from_defaults(*, vector_store: object) -> str:
            calls.append(f"storage_context:{vector_store}")
            return "storage-context"

    class FakeVectorStoreIndex:
        def __init__(
            self,
            indexed_nodes: list[object],
            *,
            storage_context: object,
            embed_model: object,
        ) -> None:
            calls.append(
                f"vector_store_index:{len(indexed_nodes)}:{storage_context}:{embed_model}"
            )

    monkeypatch.setattr(ingest_script, "StorageContext", FakeStorageContext)
    monkeypatch.setattr(ingest_script, "VectorStoreIndex", FakeVectorStoreIndex)

    ingest_script.main()

    assert calls == [
        "load_config",
        f"load_pdfs:{config.pdf_dir}",
        f"build_text_extraction_report:{config.pdf_dir}:True",
        "chunk_documents:800:80:text-embedding-3-small:v1",
        "create_openai_embedding_model_from_config:text-embedding-3-small",
        f"create_chroma_vector_store:{config.chroma_dir}:human_design_test:text-embedding-3-small",
        "storage_context:vector-store",
        "vector_store_index:1:storage-context:embed-model",
    ]
    output = capsys.readouterr().out
    assert "PDFs found: 2" in output
    assert "Documents/pages loaded: 3" in output
    assert "Chunks/nodes created: 1" in output
    assert "Chunks persisted: 1" in output
    assert "Collection name: human_design_test" in output
    assert f"Chroma directory: {config.chroma_dir}" in output
    assert "Embedding model: text-embedding-3-small" in output
    assert "Ingestion version: v1" in output
    assert "full document text should not be printed" not in output
    assert "full chunk text should not be printed" not in output


def test_main_handles_no_pdf_errors_with_clear_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest_script = _load_ingest_script()
    monkeypatch.setenv("HD_RAG_REAL_EMBEDDINGS", "1")
    config = _fake_config(tmp_path)
    monkeypatch.setattr(ingest_script, "load_config", lambda: config)
    monkeypatch.setattr(
        ingest_script,
        "load_pdfs",
        lambda pdf_dir: (_ for _ in ()).throw(
            NoPdfFilesFoundError(f"No PDF files found in: {pdf_dir}")
        ),
    )

    with pytest.raises(SystemExit, match="No PDF files found"):
        ingest_script.main()


def test_main_does_not_require_openai_api_key_in_default_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest_script = _load_ingest_script()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HD_RAG_REAL_EMBEDDINGS", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="Real embedding ingestion is disabled"):
        ingest_script.main()

    assert not (tmp_path / "storage" / "chroma").exists()
