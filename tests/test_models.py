from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest

from human_design.rag.models import (
    ChunkingResult,
    IngestionResult,
    PdfLoadResult,
    RetrievalResult,
    VectorStoreResult,
)


@pytest.mark.parametrize(
    "model",
    [
        PdfLoadResult,
        ChunkingResult,
        IngestionResult,
        VectorStoreResult,
        RetrievalResult,
    ],
)
def test_models_are_frozen_dataclasses(model: type[object]) -> None:
    assert is_dataclass(model)
    assert model.__dataclass_params__.frozen is True


def test_pdf_load_result_uses_tuple_of_paths() -> None:
    result = PdfLoadResult(
        pdf_count=2,
        document_count=14,
        source_files=(Path("data/pdfs/book-a.pdf"), Path("data/pdfs/book-b.pdf")),
    )

    assert result.pdf_count == 2
    assert result.document_count == 14
    assert result.source_files == (
        Path("data/pdfs/book-a.pdf"),
        Path("data/pdfs/book-b.pdf"),
    )
    assert isinstance(result.source_files, tuple)
    assert all(isinstance(source_file, Path) for source_file in result.source_files)


def test_chunking_result_exposes_chunk_counts_and_settings() -> None:
    result = ChunkingResult(
        document_count=14,
        chunk_count=92,
        chunk_size=800,
        chunk_overlap=80,
    )

    assert result.document_count == 14
    assert result.chunk_count == 92
    assert result.chunk_size == 800
    assert result.chunk_overlap == 80


def test_ingestion_result_exposes_counts_and_chroma_path() -> None:
    result = IngestionResult(
        pdf_count=2,
        document_count=14,
        chunk_count=92,
        persisted_count=92,
        collection_name="human_design",
        chroma_dir=Path("storage/chroma"),
    )

    assert result.pdf_count == 2
    assert result.document_count == 14
    assert result.chunk_count == 92
    assert result.persisted_count == 92
    assert result.collection_name == "human_design"
    assert result.chroma_dir == Path("storage/chroma")
    assert isinstance(result.chroma_dir, Path)


def test_vector_store_result_exposes_collection_path_and_count() -> None:
    result = VectorStoreResult(
        collection_name="human_design",
        chroma_dir=Path("storage/chroma"),
        persisted_count=92,
    )

    assert result.collection_name == "human_design"
    assert result.chroma_dir == Path("storage/chroma")
    assert isinstance(result.chroma_dir, Path)
    assert result.persisted_count == 92


def test_retrieval_result_exposes_common_source_metadata() -> None:
    result = RetrievalResult(
        text="Generators have sustainable life force energy.",
        score=0.87,
        source_file="book-a.pdf",
        page_label="12",
        page_number=12,
        metadata={"document_title": "Human Design Basics", "chapter": "Types"},
    )

    assert result.text == "Generators have sustainable life force energy."
    assert result.score == 0.87
    assert result.source_file == "book-a.pdf"
    assert result.page_label == "12"
    assert result.page_number == 12
    assert result.metadata == {
        "document_title": "Human Design Basics",
        "chapter": "Types",
    }


def test_retrieval_result_supports_missing_optional_metadata() -> None:
    result = RetrievalResult(
        text="A short retrieved snippet.",
        score=None,
        source_file=None,
        page_label=None,
        page_number=None,
        metadata={},
    )

    assert result.score is None
    assert result.source_file is None
    assert result.page_label is None
    assert result.page_number is None
    assert result.metadata == {}


def test_model_field_type_hints_are_present() -> None:
    for model in (
        PdfLoadResult,
        ChunkingResult,
        IngestionResult,
        VectorStoreResult,
        RetrievalResult,
    ):
        assert all(field.type is not None for field in fields(model))


def test_models_are_immutable() -> None:
    result = ChunkingResult(
        document_count=1,
        chunk_count=2,
        chunk_size=800,
        chunk_overlap=80,
    )

    with pytest.raises(FrozenInstanceError):
        result.chunk_count = 3


def test_models_do_not_require_openai_key_or_create_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    PdfLoadResult(pdf_count=0, document_count=0, source_files=())
    VectorStoreResult(
        collection_name="human_design",
        chroma_dir=Path("storage/chroma"),
        persisted_count=0,
    )

    assert not Path("data/pdfs").exists()
    assert not Path("storage").exists()
