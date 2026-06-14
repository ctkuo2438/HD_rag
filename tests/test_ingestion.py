from pathlib import Path

import pytest
from llama_index.core import Document

from human_design.rag import ingestion


def test_discover_pdfs_returns_sorted_pdf_paths(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    third = pdf_dir / "c.pdf"
    first = pdf_dir / "a.pdf"
    second = pdf_dir / "b.pdf"
    third.touch()
    first.touch()
    second.touch()

    assert ingestion.discover_pdfs(pdf_dir) == [first, second, third]


def test_discover_pdfs_ignores_non_pdf_files(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "book.pdf"
    pdf_path.touch()
    (pdf_dir / "notes.txt").touch()
    (pdf_dir / "image.png").touch()

    assert ingestion.discover_pdfs(pdf_dir) == [pdf_path]


def test_discover_pdfs_raises_clear_error_if_directory_does_not_exist(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(ingestion.NoPdfFilesFoundError, match="PDF directory not found"):
        ingestion.discover_pdfs(missing_dir)


def test_discover_pdfs_raises_clear_error_if_path_is_not_directory(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.touch()

    with pytest.raises(ingestion.NoPdfFilesFoundError, match="not a directory"):
        ingestion.discover_pdfs(file_path)


def test_discover_pdfs_raises_clear_error_if_directory_has_no_pdfs(
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "notes.txt").touch()

    with pytest.raises(ingestion.NoPdfFilesFoundError, match="No PDF files found"):
        ingestion.discover_pdfs(pdf_dir)


def test_load_pdf_rejects_non_pdf_paths(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.touch()

    with pytest.raises(ValueError, match="Expected a PDF file"):
        ingestion.load_pdf(text_path)


def test_load_pdf_rejects_missing_files(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        ingestion.load_pdf(missing_pdf)


def test_load_pdfs_uses_pymupdf_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    first = pdf_dir / "a.pdf"
    second = pdf_dir / "b.pdf"
    first.touch()
    second.touch()
    loaded_paths: list[Path] = []

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            loaded_paths.append(Path(file_path))
            return [Document(text=f"loaded {Path(file_path).name}")]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    documents = ingestion.load_pdfs(pdf_dir)

    assert loaded_paths == [first, second]
    assert [document.text for document in documents] == ["loaded a.pdf", "loaded b.pdf"]


def test_loaded_documents_preserve_source_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "book.pdf"
    pdf_path.touch()

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [Document(text="page text", metadata={"reader": "pymupdf"})]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    [document] = ingestion.load_pdf(pdf_path)

    assert document.metadata["source_path"] == str(pdf_path)
    assert document.metadata["file_name"] == "book.pdf"
    assert document.metadata["reader"] == "pymupdf"


def test_page_level_metadata_is_preserved_if_reader_returns_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "book.pdf"
    pdf_path.touch()

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [
                Document(
                    text="page text",
                    metadata={"page_label": "iv", "page_number": 4},
                )
            ]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    [document] = ingestion.load_pdf(pdf_path)

    assert document.metadata["page_label"] == "iv"
    assert document.metadata["page_number"] == 4


def test_ingestion_functions_do_not_require_openai_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "book.pdf"
    pdf_path.touch()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [Document(text="page text")]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    assert ingestion.load_pdf(pdf_path)


def test_ingestion_functions_do_not_create_storage_or_chroma_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    pdf_dir = tmp_path / "data" / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "book.pdf").touch()

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [Document(text="page text")]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    ingestion.load_pdfs(pdf_dir)

    assert not (tmp_path / "storage").exists()
