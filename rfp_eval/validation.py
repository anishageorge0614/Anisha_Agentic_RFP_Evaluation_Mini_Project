"""Validation Tool: schema check, normalisation of LLM output, warnings."""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

NOT_FOUND = "Not found in document"


class RawCriterion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    criterion_id: Any = None
    score: Any = None
    max_score: Any = None
    justification: Any = None
    evidence: Any = None


class RawEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    supplier_name: Any = None
    criteria: list[Any] = []
    risks: Any = []
    overall_summary: Any = ""

    @field_validator("criteria", mode="before")
    @classmethod
    def _criteria_list(cls, v: Any) -> Any:
        return v if isinstance(v, list) else []


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().split("/")[0].strip()  # tolerate "8/10"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_int(value: Any) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None and f == int(f) else None


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s or default


def validate_evaluation(
    raw: Any, criteria: list[dict[str, Any]], supplier_name: str
) -> tuple[dict[str, Any], list[str]]:
    """Return (normalised evaluation, warnings).

    Every active criterion gets exactly one entry. Missing/invalid scores become
    score=None, valid=False (excluded from benchmarks, counted as 0 in totals).
    Out-of-range scores are clipped to [0, max_score] and flagged.
    """
    warnings: list[str] = []
    try:
        parsed = RawEvaluation.model_validate(raw if isinstance(raw, dict) else {})
    except ValidationError as exc:
        warnings.append(f"Schema validation failed ({exc.error_count()} errors); all criteria treated as missing.")
        parsed = RawEvaluation()
    if not isinstance(raw, dict):
        warnings.append("LLM output was not a JSON object; all criteria treated as missing.")

    returned_name = _text(parsed.supplier_name)
    if returned_name and returned_name.lower() != supplier_name.lower():
        warnings.append(f"LLM reported supplier '{returned_name}' but the document was uploaded as '{supplier_name}'.")

    by_id: dict[int, RawCriterion] = {}
    for item in parsed.criteria:
        try:
            rc = RawCriterion.model_validate(item)
        except ValidationError:
            warnings.append("Ignored a malformed criterion entry.")
            continue
        cid = _to_int(rc.criterion_id)
        if cid is None:
            warnings.append("Ignored a criterion entry with an invalid criterion_id.")
            continue
        if cid in by_id:
            warnings.append(f"Duplicate result for criterion_id {cid}; the first one was kept.")
            continue
        by_id[cid] = rc

    active_ids = {c["criterion_id"] for c in criteria}
    for cid in sorted(set(by_id) - active_ids):
        warnings.append(f"Ignored result for unknown/inactive criterion_id {cid}.")

    normalised: list[dict[str, Any]] = []
    for c in criteria:
        cid, name, max_score = c["criterion_id"], c["name"], float(c["max_score"])
        rc = by_id.get(cid)
        entry: dict[str, Any] = {
            "criterion_id": cid,
            "criterion_name": name,
            "weight": float(c["weight"]),
            "max_score": max_score,
            "score": None,
            "valid": False,
            "justification": "",
            "evidence": NOT_FOUND,
        }
        if rc is None:
            warnings.append(f"Criterion '{name}' missing from LLM output; scored as 0 and excluded from benchmark.")
            entry["justification"] = "No result returned by the LLM for this criterion."
        else:
            entry["justification"] = _text(rc.justification, "No justification provided.")
            entry["evidence"] = _text(rc.evidence, NOT_FOUND)
            score = _to_float(rc.score)
            reported_max = _to_float(rc.max_score)
            if reported_max is not None and reported_max != max_score:
                warnings.append(
                    f"Criterion '{name}': LLM max_score {reported_max:g} differs from configured {max_score:g}; "
                    "configured value used."
                )
            if score is None:
                warnings.append(f"Criterion '{name}': score missing or not numeric; scored as 0 and excluded from benchmark.")
            else:
                clipped = min(max(score, 0.0), max_score)
                if clipped != score:
                    warnings.append(f"Criterion '{name}': score {score:g} outside 0-{max_score:g}; clipped to {clipped:g}.")
                entry["score"] = clipped
                entry["valid"] = True
        normalised.append(entry)

    risks = parsed.risks if isinstance(parsed.risks, list) else [parsed.risks]
    evaluation = {
        "criteria": normalised,
        "risks": [_text(r) for r in risks if _text(r)],
        "overall_summary": _text(parsed.overall_summary),
    }
    return evaluation, warnings
