"""PDF discovery and loading for the local Human Design RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader


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
