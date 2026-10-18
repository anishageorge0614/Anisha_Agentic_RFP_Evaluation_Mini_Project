import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rfp_eval import db  # noqa: E402
from rfp_eval.validation import validate_evaluation  # noqa: E402

CRITERIA = [
    {"criterion_id": 1, "name": "Tech", "description": "d", "weight": 60.0, "max_score": 10.0, "is_active": 1},
    {"criterion_id": 2, "name": "Price", "description": "d", "weight": 40.0, "max_score": 10.0, "is_active": 1},
]


def make_supplier(name, scores, date="2026-09-01", rating=5.0):
    """Build a ranking-tool input from {criterion_id: score or None}."""
    raw = {
        "supplier_name": name,
        "criteria": [
            {"criterion_id": cid, "score": sc, "max_score": 10, "justification": "j", "evidence": "e"}
            for cid, sc in scores.items()
            if sc is not None
        ],
    }
    evaluation, warnings = validate_evaluation(raw, CRITERIA, name)
    return {
        "supplier_name": name, "submission_date": date, "experience_rating": rating,
        "evaluation": evaluation, "warnings": warnings,
    }


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    return c


@pytest.fixture
def sample_pdfs():
    from sample_data.sample_inputs import SAMPLE_SUPPLIERS

    return [(s, (ROOT / "sample_data" / "rfps" / s["file"]).read_bytes()) for s in SAMPLE_SUPPLIERS]
