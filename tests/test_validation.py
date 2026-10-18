import json
from pathlib import Path

import pytest

from rfp_eval.llm import LLMError, build_prompt, extract_json_object
from rfp_eval.validation import validate_evaluation

from .conftest import CRITERIA, ROOT


def test_malformed_sample_is_normalised():
    raw = json.loads((ROOT / "sample_data" / "invalid_samples" / "malformed_llm_output.json").read_text())
    ev, warnings = validate_evaluation(raw, CRITERIA, "Apex Systems")
    c1, c2 = ev["criteria"]
    assert c1["score"] == 10.0 and c1["valid"]  # 14 clipped to max
    assert c2["score"] is None and not c2["valid"]  # "eight" is not numeric
    text = " ".join(warnings)
    assert "clipped" in text and "not numeric" in text and "unknown/inactive criterion_id 99" in text
    assert ev["risks"] == ["single string instead of list"]


def test_missing_criteria_filled_in():
    ev, warnings = validate_evaluation({"supplier_name": "X", "criteria": []}, CRITERIA, "X")
    assert len(ev["criteria"]) == len(CRITERIA)
    assert all(not c["valid"] and c["score"] is None for c in ev["criteria"])
    assert sum("missing from LLM output" in w for w in warnings) == 2


@pytest.mark.parametrize("bad", [None, [], "text", 42])
def test_non_object_output(bad):
    ev, warnings = validate_evaluation(bad, CRITERIA, "X")
    assert len(ev["criteria"]) == 2 and warnings


def test_negative_nan_and_string_scores():
    raw = {"criteria": [{"criterion_id": 1, "score": -3}, {"criterion_id": "2", "score": "7/10"}]}
    ev, _ = validate_evaluation(raw, CRITERIA, "X")
    assert ev["criteria"][0]["score"] == 0.0 and ev["criteria"][0]["valid"]
    assert ev["criteria"][1]["score"] == 7.0
    ev, _ = validate_evaluation({"criteria": [{"criterion_id": 1, "score": float("nan")}]}, CRITERIA, "X")
    assert not ev["criteria"][0]["valid"]


def test_supplier_name_mismatch_warns():
    _, warnings = validate_evaluation({"supplier_name": "Other", "criteria": []}, CRITERIA, "X")
    assert any("Other" in w for w in warnings)


def test_configured_max_score_wins():
    raw = {"criteria": [{"criterion_id": 1, "score": 9, "max_score": 100}]}
    ev, warnings = validate_evaluation(raw, CRITERIA, "X")
    assert ev["criteria"][0]["max_score"] == 10.0 and any("differs" in w for w in warnings)


@pytest.mark.parametrize(
    "raw",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go:\n{"a": 1}\nHope this helps.'],
)
def test_extract_json_tolerates_fences_and_prose(raw):
    assert extract_json_object(raw) == {"a": 1}


@pytest.mark.parametrize("raw", ["", "no json here", "{broken", "[1, 2]"])
def test_extract_json_rejects_garbage(raw):
    with pytest.raises(LLMError):
        extract_json_object(raw)


def test_prompt_is_built_from_active_criteria_and_rules():
    p = build_prompt(CRITERIA, "Acme", "DOC BODY")
    assert 'criterion_id 1: "Tech"' in p and 'criterion_id 2: "Price"' in p
    assert "ONLY evidence present in the supplier document" in p
    assert "EVERY active criterion" in p and "Output JSON only" in p and "DOC BODY" in p
