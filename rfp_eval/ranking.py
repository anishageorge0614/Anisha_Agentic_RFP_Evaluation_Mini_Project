"""Ranking Tool: deterministic scoring, peer benchmarking, PPI and tie-breaks.

No LLM is involved here. The same validated inputs always give the same output.
"""
from __future__ import annotations

from datetime import date
from typing import Any

PPI_DECIMALS = 4
SCORE_DECIMALS = 4

TIE_BREAK_RULES = [
    "1. Higher Peer Performance Index (PPI)",
    "2. Earlier submission date",
    "3. Higher historical experience rating",
    "4. Supplier name ascending (case-insensitive)",
]


def normalised_weights(criteria: list[dict[str, Any]]) -> tuple[dict[int, float], float]:
    """Return ({criterion_id: weight rescaled to total 100}, original_total)."""
    total = sum(float(c["weight"]) for c in criteria)
    if total <= 0:
        raise ValueError("Active criteria weights must sum to more than zero")
    return {c["criterion_id"]: float(c["weight"]) / total * 100.0 for c in criteria}, total


def compute_benchmarks(suppliers: list[dict[str, Any]], criteria: list[dict[str, Any]]) -> dict[int, float]:
    """Highest VALID score per criterion across all suppliers (invalid/missing scores are ignored)."""
    bench: dict[int, float] = {}
    for c in criteria:
        valid = [
            e["score"]
            for s in suppliers
            for e in s["evaluation"]["criteria"]
            if e["criterion_id"] == c["criterion_id"] and e["valid"]
        ]
        bench[c["criterion_id"]] = max(valid) if valid else 0.0
    return bench


def relative_percent(score: float, benchmark: float) -> float:
    """(score / benchmark) x 100. Safe rule: benchmark == 0 means nobody scored, so everyone gets 100."""
    if benchmark <= 0:
        return 100.0
    return score / benchmark * 100.0


def _sort_key(s: dict[str, Any]) -> tuple:
    return (-s["ppi"], s["submission_date"], -s["experience_rating"], s["supplier_name"].lower(), s["supplier_name"])


def _deciding_rule(prev: dict[str, Any], cur: dict[str, Any]) -> str:
    if prev["ppi"] != cur["ppi"]:
        return f"Higher PPI ({prev['ppi']:.{PPI_DECIMALS}f} vs {cur['ppi']:.{PPI_DECIMALS}f})"
    if prev["submission_date"] != cur["submission_date"]:
        return f"PPI tied; earlier submission date ({prev['submission_date']} vs {cur['submission_date']})"
    if prev["experience_rating"] != cur["experience_rating"]:
        return (
            f"PPI and submission date tied; higher experience rating "
            f"({prev['experience_rating']:g} vs {cur['experience_rating']:g})"
        )
    return "PPI, submission date and experience tied; supplier name ascending"


def rank_suppliers(
    suppliers: list[dict[str, Any]], criteria: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score, benchmark and rank.

    Each supplier dict needs: supplier_name, submission_date (ISO str), experience_rating (float),
    evaluation (output of validation.validate_evaluation). Returns the full result document.
    """
    if not suppliers:
        raise ValueError("At least one supplier is required")
    weights, original_total = normalised_weights(criteria)
    benchmarks = compute_benchmarks(suppliers, criteria)

    ranked: list[dict[str, Any]] = []
    for s in suppliers:
        absolute = 0.0
        ppi_acc = 0.0
        rows = []
        for e in s["evaluation"]["criteria"]:
            cid = e["criterion_id"]
            w = weights[cid]
            score = e["score"] if e["valid"] else 0.0
            bench = benchmarks[cid]
            weighted = score / e["max_score"] * w
            rel = relative_percent(score, bench)
            absolute += weighted
            ppi_acc += rel * w / 100.0
            rows.append(
                {
                    **e,
                    "effective_weight": round(w, SCORE_DECIMALS),
                    "score": score,
                    "score_missing": not e["valid"],
                    "benchmark": bench,
                    "gap": round(score - bench, SCORE_DECIMALS),
                    "relative_pct": round(rel, SCORE_DECIMALS),
                    "weighted_score": round(weighted, SCORE_DECIMALS),
                }
            )
        ranked.append(
            {
                "supplier_name": s["supplier_name"],
                "submission_date": s["submission_date"],
                "experience_rating": float(s["experience_rating"]),
                "source_file": s.get("source_file"),
                "page_count": s.get("page_count"),
                "absolute_score": round(absolute, SCORE_DECIMALS),
                "ppi": round(ppi_acc, PPI_DECIMALS),
                "criteria": rows,
                "risks": s["evaluation"]["risks"],
                "overall_summary": s["evaluation"]["overall_summary"],
                "warnings": s.get("warnings", []),
            }
        )

    ranked.sort(key=_sort_key)
    explanations = []
    for i, s in enumerate(ranked, 1):
        s["final_rank"] = i
        if i == 1:
            s["tie_break_note"] = "Rank 1: highest in the ordering."
        else:
            s["tie_break_note"] = f"Ranked below {ranked[i - 2]['supplier_name']}: " + _deciding_rule(ranked[i - 2], s)
        explanations.append({"rank": i, "supplier_name": s["supplier_name"], "reason": s["tie_break_note"]})

    return {
        "formulas": {
            "absolute_score": "sum((score / max_score) x effective_weight)",
            "benchmark": "highest valid score for the criterion across suppliers",
            "gap": "supplier score - benchmark",
            "relative_pct": "(score / benchmark) x 100; benchmark 0 -> 100",
            "ppi": "sum(relative_pct x effective_weight) / 100",
            "tie_break_order": TIE_BREAK_RULES,
        },
        "criteria_used": [
            {**c, "effective_weight": round(weights[c["criterion_id"]], SCORE_DECIMALS)} for c in criteria
        ],
        "original_weight_total": original_total,
        "benchmarks": {str(k): v for k, v in benchmarks.items()},
        "tie_break_explanation": explanations,
        "suppliers": ranked,
    }


def parse_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)
