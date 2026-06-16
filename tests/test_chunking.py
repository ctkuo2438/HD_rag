from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from human_design.rag import chunking
from human_design.rag.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INGESTION_VERSION,
)
from human_design.rag.models import ChunkingResult


def test_chunk_documents_uses_default_sentence_splitter_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_settings: dict[str, int] = {}

    class FakeSentenceSplitter:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            captured_settings["chunk_size"] = chunk_size
            captured_settings["chunk_overlap"] = chunk_overlap

        def get_nodes_from_documents(self, documents: list[Document]) -> list[TextNode]:
            return [
                TextNode(text=document.text, metadata=dict(document.metadata))
                for document in documents
            ]

    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSentenceSplitter)

    nodes, result = chunking.chunk_documents([Document(text="sample text")])

    assert captured_settings == {
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    }
    assert len(nodes) == 1
    assert isinstance(result, ChunkingResult)
    assert result.document_count == 1
    assert result.chunk_count == 1
    assert result.chunk_size == DEFAULT_CHUNK_SIZE
    assert result.chunk_overlap == DEFAULT_CHUNK_OVERLAP


def test_chunk_documents_accepts_config_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_settings: dict[str, int] = {}

    class FakeSentenceSplitter:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            captured_settings["chunk_size"] = chunk_size
            captured_settings["chunk_overlap"] = chunk_overlap

        def get_nodes_from_documents(self, documents: list[Document]) -> list[TextNode]:
            return [TextNode(text="node", metadata=dict(documents[0].metadata))]

    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSentenceSplitter)

    nodes, result = chunking.chunk_documents(
        [Document(text="source text", metadata={"file_name": "book.pdf"})],
        chunk_size=120,
        chunk_overlap=20,
        embedding_model="test-embedding-model",
        ingestion_version="test-v2",
    )

    assert captured_settings == {"chunk_size": 120, "chunk_overlap": 20}
    assert result.chunk_size == 120
    assert result.chunk_overlap == 20

    [node] = nodes
    assert node.metadata["chunk_size"] == 120
    assert node.metadata["chunk_overlap"] == 20
    assert node.metadata["embedding_model"] == "test-embedding-model"
    assert node.metadata["ingestion_version"] == "test-v2"


def test_chunk_documents_returns_nodes_from_in_memory_documents() -> None:
    documents = [
        Document(
            text="Human Design type strategy authority profile. " * 80,
            metadata={"file_name": "book.pdf"},
        )
    ]

    nodes, result = chunking.chunk_documents(documents)

    assert nodes
    assert result.document_count == 1
    assert result.chunk_count == len(nodes)
    assert all(node.get_content() for node in nodes)


def test_chunk_documents_adds_default_metadata_to_each_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSentenceSplitter:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            pass

        def get_nodes_from_documents(self, documents: list[Document]) -> list[TextNode]:
            return [
                TextNode(text="first node", metadata=dict(documents[0].metadata)),
                TextNode(text="second node", metadata=dict(documents[0].metadata)),
            ]

    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSentenceSplitter)

    nodes, _result = chunking.chunk_documents(
        [
            Document(
                text="source text",
                metadata={
                    "source_path": "data/pdfs/book.pdf",
                    "file_name": "book.pdf",
                    "page_label": "iv",
                    "page_number": 4,
                    "document_title": "Human Design Book",
                    "custom_key": "preserved",
                },
            )
        ]
    )

    for node in nodes:
        assert node.metadata["chunk_size"] == DEFAULT_CHUNK_SIZE
        assert node.metadata["chunk_overlap"] == DEFAULT_CHUNK_OVERLAP
        assert node.metadata["embedding_model"] == DEFAULT_EMBEDDING_MODEL
        assert node.metadata["ingestion_version"] == DEFAULT_INGESTION_VERSION
        assert node.metadata["source_path"] == "data/pdfs/book.pdf"
        assert node.metadata["source_file"] == "book.pdf"
        assert node.metadata["page_label"] == "iv"
        assert node.metadata["page_number"] == 4
        assert node.metadata["document_title"] == "Human Design Book"
        assert node.metadata["custom_key"] == "preserved"


def test_chunk_documents_uses_source_metadata_as_page_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSentenceSplitter:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            pass

        def get_nodes_from_documents(self, documents: list[Document]) -> list[TextNode]:
            return [TextNode(text="node", metadata=dict(documents[0].metadata))]

    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSentenceSplitter)

    [node], _result = chunking.chunk_documents(
        [Document(text="source text", metadata={"source": "12"})]
    )

    assert node.metadata["page_label"] == "12"
    assert node.metadata["page_number"] == 12


def test_chunk_documents_ignores_non_int_like_source_for_page_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSentenceSplitter:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            pass

        def get_nodes_from_documents(self, documents: list[Document]) -> list[TextNode]:
            return [TextNode(text="node", metadata=dict(documents[0].metadata))]

    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSentenceSplitter)

    [node], _result = chunking.chunk_documents(
        [Document(text="source text", metadata={"source": "cover"})]
    )

    assert node.metadata["page_label"] == "cover"
    assert "page_number" not in node.metadata


def test_chunk_documents_does_not_call_pdf_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from human_design.rag import ingestion

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("chunk_documents must not load PDFs")

    monkeypatch.setattr(ingestion, "load_pdf", fail_if_called)
    monkeypatch.setattr(ingestion, "load_pdfs", fail_if_called)

    nodes, result = chunking.chunk_documents([Document(text="in-memory text")])

    assert nodes
    assert result.document_count == 1


def test_chunk_documents_does_not_require_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    nodes, result = chunking.chunk_documents([Document(text="in-memory text")])

    assert nodes
    assert result.chunk_count == len(nodes)


def test_chunk_documents_does_not_create_storage_or_raw_text_dumps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    chunking.chunk_documents([Document(text="full text remains in memory")])

    assert list(tmp_path.rglob("*")) == []
    assert not (tmp_path / "storage").exists()