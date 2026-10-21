"""Local BM25-style retrieval with stable, page-level source references."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .document_tool import DocumentPage

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "with",
}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOP_WORDS]


def _chunks(pages: list[DocumentPage], max_chars: int = 1200, overlap: int = 160) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page in pages:
        text = page.text.strip()
        start = 0
        index = 1
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
            excerpt = text[start:end].strip()
            if excerpt:
                chunks.append(
                    {
                        "source_ref": f"P{page.page_number}-C{index}",
                        "page_number": page.page_number,
                        "text": excerpt,
                    }
                )
                index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def retrieve(query: str, pages: list[DocumentPage], top_k: int = 4) -> list[dict[str, Any]]:
    """Return relevant chunks ranked by BM25; ties always resolve by page and source reference."""
    if top_k <= 0 or not query.strip():
        return []
    chunks = _chunks(pages)
    query_terms = sorted(set(_tokens(query)))
    if not chunks or not query_terms:
        return []

    tokenized = [_tokens(chunk["text"]) for chunk in chunks]
    frequencies = [Counter(tokens) for tokens in tokenized]
    document_frequency = Counter(term for term in query_terms for counts in frequencies if counts[term])
    avg_length = sum(map(len, tokenized)) / len(tokenized)
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk, tokens, counts in zip(chunks, tokenized, frequencies):
        score = 0.0
        length_norm = 1.2 * (1 - 0.75 + 0.75 * len(tokens) / max(avg_length, 1.0))
        for term in query_terms:
            term_frequency = counts[term]
            if term_frequency:
                idf = math.log(1 + (len(chunks) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                score += idf * term_frequency * 2.2 / (term_frequency + length_norm)
        if score:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]["page_number"], item[1]["source_ref"]))
    return [{**chunk, "relevance": round(score, 6)} for score, chunk in scored[:top_k]]


def retrieve_for_criteria(
    criteria: list[dict[str, Any]], pages: list[DocumentPage], top_k: int = 3
) -> dict[int, list[dict[str, Any]]]:
    return {
        criterion["criterion_id"]: retrieve(
            f"{criterion['name']} {criterion['description']}", pages, top_k=top_k
        )
        for criterion in criteria
    }