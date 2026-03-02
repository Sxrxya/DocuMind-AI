"""
DocuMind-AI — Text Extraction

Extracts raw UTF-8 text from PDF, DOCX, and TXT files.
"""

import os
from pathlib import Path

from PyPDF2 import PdfReader
from docx import Document

from backend.core.logging import get_logger

logger = get_logger(__name__)


def extract_text(file_path: str) -> str:
    """
    Extract text from a document file.

    Supported formats: .pdf, .docx, .txt
    Raises ValueError for unsupported file types.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    logger.info(f"Extracting text from '{path.name}' (type: {ext})")

    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext == ".docx":
        return _extract_docx(path)
    elif ext == ".txt":
        return _extract_txt(path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: .pdf, .docx, .txt"
        )


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF file using PyPDF2."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    full_text = "\n\n".join(pages)
    logger.info(f"Extracted {len(reader.pages)} pages, {len(full_text)} chars")
    return full_text


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    logger.info(f"Extracted {len(paragraphs)} paragraphs, {len(full_text)} chars")
    return full_text


def _extract_txt(path: Path) -> str:
    """Read a plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        full_text = f.read()
    logger.info(f"Read {len(full_text)} chars from text file")
    return full_text
