"""PDF discovery and loading for the local Human Design RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader

from human_design.rag.models import LowTextDocument, TextExtractionReport


DEFAULT_PDF_DIR = Path("data/pdfs")


class NoPdfFilesFoundError(FileNotFoundError):
    """Raised when a PDF directory cannot provide local PDF files."""


def discover_pdfs(pdf_dir: Path = DEFAULT_PDF_DIR) -> list[Path]:
    """
    Return sorted local PDF paths from a directory.
    """
    if not pdf_dir.exists():
        raise NoPdfFilesFoundError(f"PDF directory not found: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NoPdfFilesFoundError(f"PDF path is not a directory: {pdf_dir}")

    pdf_paths = sorted(
        path for path in pdf_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise NoPdfFilesFoundError(f"No PDF files found in: {pdf_dir}")
    return pdf_paths


def load_pdf(pdf_path: Path) -> list[Document]:
    """
    Load one PDF with PyMuPDFReader and attach source metadata.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF path is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    documents = PyMuPDFReader().load(file_path=pdf_path, metadata=True)
    for document in documents:
        document.metadata = {
            **document.metadata,
            "source_path": str(pdf_path),
            "file_name": pdf_path.name,
        }
    return documents


def load_pdfs(pdf_dir: Path = DEFAULT_PDF_DIR) -> list[Document]:
    """
    Load every discovered PDF and return a combined document list.
    """
    documents: list[Document] = []
    for pdf_path in discover_pdfs(pdf_dir):
        documents.extend(load_pdf(pdf_path))
    return documents


def build_text_extraction_report(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    low_text_threshold: int = 100,
) -> TextExtractionReport:
    """
    Build an in-memory report for embedded text extracted from local PDFs.
    """
    pdf_paths = discover_pdfs(pdf_dir)
    documents: list[Document] = []
    for pdf_path in pdf_paths:
        documents.extend(load_pdf(pdf_path))

    total_text_characters = 0
    low_text_documents: list[LowTextDocument] = []
    for document in documents:
        text = document.text or ""
        stripped_text = text.strip() # only non-whitespace characters
        text_length = len(stripped_text) # character count, not token count
        total_text_characters += text_length # sum of text lengths across all documents

        if text_length < low_text_threshold:
            low_text_documents.append(
                LowTextDocument(
                    source_path=_metadata_str(document, "source_path"),
                    file_name=_metadata_str(document, "file_name"),
                    page_label=_metadata_str(document, "page_label") or _metadata_str(document, "source"),
                    page_number=_metadata_int(document, "page_number") or _metadata_int(document, "source"),
                    text_length=text_length,
                )
            )

    return TextExtractionReport(
        pdf_count=len(pdf_paths),
        document_count=len(documents),
        source_files=tuple(pdf_paths),
        total_text_characters=total_text_characters,
        low_text_threshold=low_text_threshold,
        low_text_document_count=len(low_text_documents),
        low_text_documents=tuple(low_text_documents),
    )


def _metadata_str(document: Document, key: str) -> str | None:
    # load_pdf() will return list[Document]
    '''
    Document looks like this:
        Document(
        text="words on page 12 ...",
        metadata={
            "source_path": "data/pdfs/book.pdf",
            "file_name": "book.pdf",
            "page_label": "12",
            "page_number": 12,
        }
    )
    so value = document.metadata.get(key) will be like "data/pdfs/book.pdf" or "12"
    '''
    value = document.metadata.get(key)
    if value is None:
        return None
    return str(value)


def _metadata_int(document: Document, key: str) -> int | None:
    value = document.metadata.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
