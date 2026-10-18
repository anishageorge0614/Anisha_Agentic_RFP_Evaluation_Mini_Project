"""Offline mock "LLM" so the app can be demoed and tested without any API key.

It is a transparent keyword heuristic over the document text and returns the same JSON shape a real model
must return. It is NOT a substitute for a real model: it only exists to exercise the pipeline.
"""
from __future__ import annotations

import re
from typing import Any

from .validation import NOT_FOUND

# (term, points). Negative points flag vague / missing information.
RULES: dict[str, dict[str, Any]] = {
    "technical": {
        "base": 5.0, "scale": 0.45,
        "terms": [
            ("microservices", 1.2), ("autoscaling", 1.0), ("load-tested", 1.0), ("odata", 1.0), ("idoc", 0.8),
            ("certified connectors", 0.8), ("saml", 0.6), ("openid", 0.4), ("api-first", 0.5), ("sandbox", 0.7),
            ("scales to", 0.8), ("modular", 0.4), ("kubernetes", 0.5), ("10,000 concurrent", 0.5),
            ("rest", 0.3), ("webhook", 0.3), ("pre-built", 0.6),
            ("csv", -0.5), ("not included", -0.5), ("to be confirmed", -0.3), ("suitable approaches", -0.3),
            ("finalised in the discovery", -0.2),
        ],
    },
    "implementation": {
        "base": 4.6, "scale": 0.4,
        "terms": [
            ("milestone", 0.8), ("raci", 1.0), ("risk register", 1.0), ("gated", 0.8), ("acceptance", 0.5),
            ("pilot", 0.8), ("hypercare", 0.5), ("uat", 0.5), ("named roles", 1.0), ("phases", 0.7),
            ("dedicated project manager", 0.8), ("train-the-trainer", 0.6), ("change manager", 0.6),
            ("communication plan", 0.6), ("fte", 0.5), ("discovery", 0.3), ("team of", 0.4),
            ("weeks 1-", 0.4), ("week 3", 0.4),
            ("to be confirmed", -0.5), ("is not part of", -1.0), ("exact numbers", -0.5),
        ],
    },
    "commercial": {
        "base": 0.5, "scale": 0.85,
        "terms": [
            ("fixed price", 1.5), ("total three-year cost", 1.0), ("assumptions", 1.0),
            ("milestone-based payments", 1.0), ("prices valid", 0.7), ("change requests at", 0.8), ("usd", 0.3),
            ("charged separately", -0.5), ("may be adjusted", -0.5), ("estimated", -0.25),
        ],
        "price_bonus": True,
    },
    "security": {
        "base": 2.0, "scale": 0.7,
        "terms": [
            ("iso 27001", 1.5), ("soc 2 type ii", 1.5), ("soc 2 type i ", 0.8), ("aes-256", 1.0),
            ("tls 1.3", 0.5), ("encrypted", 0.5), ("mfa", 0.7), ("role-based", 0.6), ("role based", 0.4),
            ("audit log", 0.8), ("immutable", 0.8), ("penetration test", 1.0), ("gdpr", 0.8),
            ("data processing agreement", 0.5), ("data residency", 0.3), ("customer-managed keys", 0.5),
            ("in progress", -1.0), ("on request", -0.75), ("best practices", -0.5),
        ],
    },
    "support": {
        "base": 1.5, "scale": 1.0,
        "terms": [
            ("24x7", 1.5), ("named", 0.7), ("p1", 0.4), ("30-minute", 1.0), ("quarterly service reviews", 0.7),
            ("references:", 1.0), ("twelve", 2.0), ("five comparable", 1.5), ("three comparable", 1.0),
            ("since 2017", 0.5), ("case study", 1.0), ("agreed to take reference calls", 1.5),
            ("food and consumer goods", 0.5), ("8x5", 0.3), ("customer success manager", 0.5),
            ("references are available on request", -0.5), ("founded in 2022", -0.25),
            ("will be agreed after", -0.5), ("one comparable", 0.3), ("ticket portal", 1.0), ("working hours", 0.5),
        ],
    },
}

CATEGORY_KEYWORDS = {
    "technical": ["technical", "architecture", "capability"],
    "implementation": ["implementation", "delivery", "plan"],
    "commercial": ["commercial", "price", "pricing", "cost", "value"],
    "security": ["security", "compliance", "privacy"],
    "support": ["support", "experience", "reference"],
}


def _category(name: str) -> str | None:
    low = name.lower()
    for cat, words in CATEGORY_KEYWORDS.items():
        if any(w in low for w in words):
            return cat
    return None


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", re.sub(r"\[Page \d+\]", " ", text))
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+", flat) if s.strip()]


def _total_price(text: str) -> float | None:
    m = re.search(r"total three-year cost\s*([\d,]+)", re.sub(r"\s+", " ", text), re.IGNORECASE)
    return float(m.group(1).replace(",", "")) if m else None


def _score_category(rule: dict[str, Any], text: str, sents: list[str]) -> tuple[float, list[str], list[str], str]:
    low = re.sub(r"\s+", " ", text.lower()) + " "
    pos, neg, gain, loss = [], [], 0.0, 0.0
    best_term, best_abs = None, 0.0
    for term, pts in rule["terms"]:
        if term in low:
            gain, loss = (gain + pts, loss) if pts > 0 else (gain, loss + pts)
            (pos if pts > 0 else neg).append(term.strip())
            if abs(pts) > best_abs and (pts > 0 or best_term is None or best_term[1] < 0):
                best_term, best_abs = (term.strip(), pts), abs(pts)
    if rule.get("price_bonus"):
        total = _total_price(text)
        if total is not None:
            gain += max(0.0, min(5.0, (520_000 - total) / 50_000))
    points = rule["base"] + gain * rule.get("scale", 1.0) + loss
    evidence = NOT_FOUND
    if best_term:
        evidence = next((s for s in sents if best_term[0] in s.lower()), NOT_FOUND)
    return points, pos, neg, evidence


def mock_evaluate(criteria: list[dict[str, Any]], supplier_name: str, text: str) -> dict[str, Any]:
    sents = _sentences(text)
    results, strengths, weaknesses = [], [], []
    for c in criteria:
        cat = _category(c["name"])
        if cat:
            points, pos, neg, evidence = _score_category(RULES[cat], text, sents)
        else:  # unknown criterion: overlap with words of its description
            words = {w for w in re.findall(r"[a-z]{4,}", c["description"].lower())}
            hits = [w for w in sorted(words) if w in text.lower()]
            points, pos, neg = 1.0 + 1.5 * len(hits), hits, []
            evidence = next((s for s in sents if hits and hits[0] in s.lower()), NOT_FOUND)
        score = round(min(max(points, 0.0), 10.0) * float(c["max_score"]) / 10.0 * 2) / 2  # half-point steps
        if pos:
            strengths.append(f"{c['name']}: {', '.join(pos[:3])}")
        if neg:
            weaknesses.append(f"{c['name']}: vague or missing detail ({', '.join(neg[:3])})")
        justification = (
            f"Document indicators found: {', '.join(pos[:5]) or 'none'}."
            + (f" Gaps or vague wording: {', '.join(neg[:4])}." if neg else "")
        )
        results.append(
            {
                "criterion_id": c["criterion_id"],
                "score": score,
                "max_score": c["max_score"],
                "justification": justification,
                "evidence": evidence,
            }
        )
    return {
        "supplier_name": supplier_name,
        "criteria": results,
        "risks": weaknesses or ["No notable risks detected by the offline heuristic."],
        "overall_summary": "Offline heuristic summary. Strengths - " + ("; ".join(strengths[:3]) or "none detected") + ".",
    }
