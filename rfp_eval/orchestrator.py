"""Orchestrator Agent: runs the workflow and calls the tools in the required order.

Setup -> Input validation -> Batch -> Evaluate (extract, prompt, LLM) -> Validate -> Score ->
Benchmark -> Rank -> Persist -> Present.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable

from . import db
from .document_tool import DocumentError, extract_pdf_text
from .llm import LLMConfig, LLMError, LLMFormatError, evaluate_supplier
from .ranking import rank_suppliers
from .validation import validate_evaluation

RATING_MIN, RATING_MAX = 0.0, 10.0
WEIGHT_TOLERANCE = 0.01


@dataclass
class SupplierInput:
    supplier_name: str
    submission_date: date | str | None
    experience_rating: float | None
    pdf_bytes: bytes | None
    filename: str = ""


class InputValidationError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def validate_inputs(suppliers: list[SupplierInput]) -> list[str]:
    errors: list[str] = []
    if not suppliers:
        return ["Upload at least one supplier PDF."]
    seen: dict[str, int] = {}
    for i, s in enumerate(suppliers, 1):
        label = f"Supplier {i} ({s.filename or 'no file'})"
        name = (s.supplier_name or "").strip()
        if not name:
            errors.append(f"{label}: supplier name is required.")
        elif name.lower() in seen:
            errors.append(f"{label}: duplicate supplier name '{name}'.")
        else:
            seen[name.lower()] = i
        if not s.pdf_bytes:
            errors.append(f"{label}: PDF file is missing or empty.")
        if s.submission_date is None or s.submission_date == "":
            errors.append(f"{label}: submission date is required.")
        else:
            try:
                d = s.submission_date if isinstance(s.submission_date, date) else date.fromisoformat(str(s.submission_date))
                if d > datetime.now(timezone.utc).date():
                    errors.append(f"{label}: submission date {d} is in the future.")
            except ValueError:
                errors.append(f"{label}: submission date must be YYYY-MM-DD.")
        r = s.experience_rating
        if r is None or not (RATING_MIN <= float(r) <= RATING_MAX):
            errors.append(f"{label}: experience rating must be between {RATING_MIN:g} and {RATING_MAX:g}.")
    return errors


def run_evaluation(
    conn: sqlite3.Connection,
    llm_cfg: LLMConfig,
    suppliers: list[SupplierInput],
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Execute one full RFP run and return (and persist) the complete result document."""
    note = progress or (lambda msg, frac: None)

    errors = validate_inputs(suppliers)
    if errors:
        raise InputValidationError(errors)

    # Reload criteria for every run so edits in the DB apply without any code change.
    criteria = db.get_active_criteria(conn)
    if not criteria:
        raise InputValidationError(["No active evaluation criteria in the database."])
    run_warnings: list[str] = []
    total_weight = sum(c["weight"] for c in criteria)
    if total_weight <= 0:
        raise InputValidationError(["Active criteria weights must total more than 0."])
    if abs(total_weight - 100) > WEIGHT_TOLERANCE:
        run_warnings.append(
            f"Active criteria weights total {total_weight:g}% (expected 100%). Weights were rescaled proportionally."
        )

    run_id = db.create_run(conn)
    prepared: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    n = len(suppliers)

    for idx, s in enumerate(suppliers):
        name = s.supplier_name.strip()
        note(f"Evaluating {name} ({idx + 1}/{n})", idx / n)
        try:
            text, pages = extract_pdf_text(s.pdf_bytes)
            try:
                raw = evaluate_supplier(llm_cfg, criteria, name, text)
            except LLMFormatError:
                raw = evaluate_supplier(llm_cfg, criteria, name, text)  # one retry on unusable JSON
        except (DocumentError, LLMError) as exc:
            failed.append({"supplier_name": name, "error": str(exc)})
            run_warnings.append(f"{name}: excluded from ranking - {exc}")
            continue
        evaluation, warnings = validate_evaluation(raw, criteria, name)
        prepared.append(
            {
                "supplier_name": name,
                "submission_date": s.submission_date.isoformat()
                if isinstance(s.submission_date, date)
                else str(s.submission_date),
                "experience_rating": float(s.experience_rating),
                "source_file": s.filename,
                "page_count": pages,
                "evaluation": evaluation,
                "warnings": warnings,
            }
        )

    if not prepared:
        db.set_run_status(conn, run_id, "FAILED")
        raise InputValidationError(
            ["No supplier could be evaluated."] + [f"{f['supplier_name']}: {f['error']}" for f in failed]
        )

    note("Scoring, benchmarking and ranking", 0.9)
    result = rank_suppliers(prepared, criteria)
    result.update(
        {
            "rfp_run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "COMPLETED" if not failed else "COMPLETED_WITH_ERRORS",
            "llm_provider": llm_cfg.provider,
            "llm_model": "offline-heuristic" if llm_cfg.is_mock else llm_cfg.model,
            "run_warnings": run_warnings,
            "failed_suppliers": failed,
        }
    )
    db.save_run_results(conn, run_id, result, status=result["status"])
    note("Done", 1.0)
    return result
