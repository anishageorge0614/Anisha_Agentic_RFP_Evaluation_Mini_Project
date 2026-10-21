"""Document Tool: extract clean text from a PDF."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader


class DocumentError(Exception):
    pass


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(source: bytes | BinaryIO, min_chars: int = 200) -> list[DocumentPage]:
    """Extract cleaned page text while preserving page numbers for retrieval citations."""
    data = source if isinstance(source, (bytes, bytearray)) else source.read()
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [DocumentPage(i, clean_text(p.extract_text() or "")) for i, p in enumerate(reader.pages, 1)]
    except Exception as exc:  # pypdf raises many error types
        raise DocumentError(f"Could not read PDF: {exc}") from exc
    text = "\n\n".join(page.text for page in pages)
    if len(re.sub(r"\[Page \d+\]|\s", "", text)) < min_chars:
        raise DocumentError("PDF contains little or no extractable text (scanned image or empty file?)")
    return pages


def extract_pdf_text(source: bytes | BinaryIO, min_chars: int = 200) -> tuple[str, int]:
    """Return (page-labelled text, page_count). Raises DocumentError for unreadable/empty PDFs."""
    pages = extract_pdf_pages(source, min_chars=min_chars)
    text = clean_text("\n\n".join(f"[Page {page.page_number}]\n{page.text}" for page in pages))
    return text, len(pages)
