"""Document Tool: extract clean text from a PDF."""
from __future__ import annotations

import io
import re
from typing import BinaryIO

from pypdf import PdfReader


class DocumentError(Exception):
    pass


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(source: bytes | BinaryIO, min_chars: int = 200) -> tuple[str, int]:
    """Return (clean_text, page_count). Raises DocumentError for unreadable/empty PDFs."""
    data = source if isinstance(source, (bytes, bytearray)) else source.read()
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception as exc:  # pypdf raises many error types
        raise DocumentError(f"Could not read PDF: {exc}") from exc
    text = clean_text("\n\n".join(f"[Page {i}]\n{t}" for i, t in enumerate(pages, 1)))
    if len(re.sub(r"\[Page \d+\]|\s", "", text)) < min_chars:
        raise DocumentError("PDF contains little or no extractable text (scanned image or empty file?)")
    return text, len(pages)
