from pathlib import Path

import pytest
from llama_index.core import Document

from human_design.rag import ingestion
from human_design.rag.models import TextExtractionReport


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


def test_build_text_extraction_report_returns_report_with_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    first = pdf_dir / "a.pdf"
    second = pdf_dir / "b.pdf"
    first.touch()
    second.touch()

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [Document(text=f"text from {Path(file_path).name}")]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    report = ingestion.build_text_extraction_report(pdf_dir, low_text_threshold=5)

    assert isinstance(report, TextExtractionReport)
    assert report.pdf_count == 2
    assert report.document_count == 2
    assert report.source_files == (first, second)
    assert report.total_text_characters == len("text from a.pdf") + len("text from b.pdf")
    assert report.low_text_threshold == 5


def test_build_text_extraction_report_reuses_preloaded_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    first = pdf_dir / "a.pdf"
    second = pdf_dir / "b.pdf"
    first.touch()
    second.touch()
    documents = [
        Document(
            text="enough text",
            metadata={
                "source_path": str(first),
                "file_name": "a.pdf",
                "page_label": "1",
                "page_number": 1,
            },
        ),
        Document(
            text="few",
            metadata={
                "source_path": str(second),
                "file_name": "b.pdf",
                "source": "2",
            },
        ),
    ]
    original_metadata = [dict(document.metadata) for document in documents]

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("preloaded report must not load PDFs")

    monkeypatch.setattr(ingestion, "load_pdf", fail_if_called)
    monkeypatch.setattr(ingestion, "load_pdfs", fail_if_called)
    monkeypatch.setattr(ingestion, "PyMuPDFReader", fail_if_called)

    report = ingestion.build_text_extraction_report(
        pdf_dir,
        low_text_threshold=5,
        documents=documents,
    )

    assert report.pdf_count == 2
    assert report.document_count == 2
    assert report.source_files == (first, second)
    assert report.total_text_characters == len("enough text") + len("few")
    assert report.low_text_document_count == 1
    [low_text_document] = report.low_text_documents
    assert low_text_document.source_path == str(second)
    assert low_text_document.file_name == "b.pdf"
    assert low_text_document.page_label == "2"
    assert low_text_document.page_number == 2
    assert [dict(document.metadata) for document in documents] == original_metadata


def test_build_text_extraction_report_detects_low_text_documents_with_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "book.pdf"
    pdf_path.touch()

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [
                Document(
                    text="  few  ",
                    metadata={"page_label": "iv", "page_number": "4"},
                ),
                Document(
                    text="This page has enough embedded text to exceed threshold.",
                    metadata={"page_label": "v", "page_number": 5},
                ),
            ]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    report = ingestion.build_text_extraction_report(pdf_dir, low_text_threshold=10)

    assert report.low_text_document_count == 1
    assert len(report.low_text_documents) == 1
    low_text_document = report.low_text_documents[0]
    assert low_text_document.source_path == str(pdf_path)
    assert low_text_document.file_name == "book.pdf"
    assert low_text_document.page_label == "iv"
    assert low_text_document.page_number == 4
    assert low_text_document.text_length == len("few")


def test_build_text_extraction_report_has_no_low_text_documents_when_above_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "book.pdf").touch()

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [Document(text="enough embedded text")]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    report = ingestion.build_text_extraction_report(pdf_dir, low_text_threshold=10)

    assert report.low_text_document_count == 0
    assert report.low_text_documents == ()


def test_build_text_extraction_report_does_not_require_openai_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "book.pdf").touch()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [Document(text="embedded text")]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    assert ingestion.build_text_extraction_report(pdf_dir)


def test_build_text_extraction_report_does_not_create_storage_or_text_dumps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    pdf_dir = tmp_path / "data" / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "book.pdf").touch()

    class FakePyMuPDFReader:
        def load(self, file_path: Path, metadata: bool = True) -> list[Document]:
            return [Document(text="copyrighted full text should stay in memory only")]

    monkeypatch.setattr(ingestion, "PyMuPDFReader", FakePyMuPDFReader)

    ingestion.build_text_extraction_report(pdf_dir)

    written_files = sorted(
        path.relative_to(tmp_path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert written_files == [Path("data/pdfs/book.pdf")]
    assert not (tmp_path / "storage").exists()
