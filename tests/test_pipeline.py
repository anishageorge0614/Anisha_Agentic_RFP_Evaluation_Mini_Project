import json
from datetime import date

import pytest
import requests

from rfp_eval import db, llm
from rfp_eval.document_tool import DocumentError, extract_pdf_text
from rfp_eval.llm import LLMConfig, LLMError
from rfp_eval.orchestrator import InputValidationError, SupplierInput, run_evaluation, validate_inputs

from .conftest import ROOT

MOCK = LLMConfig(provider="Mock (offline demo)")


def sample_inputs(sample_pdfs):
    return [SupplierInput(s["name"], s["date"], s["rating"], pdf, s["file"]) for s, pdf in sample_pdfs]


def test_pdf_extraction_and_page_counts(sample_pdfs):
    for s, pdf in sample_pdfs:
        text, pages = extract_pdf_text(pdf)
        assert 2 <= pages <= 4
        assert "Executive Summary" in text and "Total three-year cost" in text


def test_corrupt_pdf_rejected():
    with pytest.raises(DocumentError):
        extract_pdf_text((ROOT / "sample_data" / "invalid_samples" / "corrupt_proposal.pdf").read_bytes())


def test_full_run_is_persisted_and_reproducible(conn, sample_pdfs):
    r1 = run_evaluation(conn, MOCK, sample_inputs(sample_pdfs))
    r2 = run_evaluation(conn, MOCK, sample_inputs(sample_pdfs))
    assert r1["rfp_run_id"] != r2["rfp_run_id"]
    assert [s["supplier_name"] for s in r1["suppliers"]] == [s["supplier_name"] for s in r2["suppliers"]]
    assert [s["ppi"] for s in r1["suppliers"]] == [s["ppi"] for s in r2["suppliers"]]
    assert len(r1["suppliers"]) == 4 and r1["status"] == "COMPLETED"

    rows = conn.execute("SELECT * FROM supplier_results WHERE rfp_run_id=? ORDER BY final_rank", (r1["rfp_run_id"],)).fetchall()
    assert [r["final_rank"] for r in rows] == [1, 2, 3, 4]
    assert json.loads(rows[0]["result_json"])["supplier_name"] == rows[0]["supplier_name"]
    assert conn.execute("SELECT status FROM rfp_runs WHERE rfp_run_id=?", (r1["rfp_run_id"],)).fetchone()[0] == "COMPLETED"
    loaded = db.load_run(conn, r1["rfp_run_id"])
    assert [s["final_rank"] for s in loaded["suppliers"]] == [1, 2, 3, 4]


def test_expected_sample_profile(conn, sample_pdfs):
    r = run_evaluation(conn, MOCK, sample_inputs(sample_pdfs))
    s = {x["supplier_name"]: {c["criterion_name"]: c["score"] for c in x["criteria"]} for x in r["suppliers"]}
    assert s["Apex Systems"]["Security & Compliance"] == max(v["Security & Compliance"] for v in s.values())
    assert s["BrightPath Tech"]["Commercial Value"] > s["Apex Systems"]["Commercial Value"]
    assert s["NexaWorks"]["Implementation Plan"] == max(v["Implementation Plan"] for v in s.values())
    assert s["BrightPath Tech"]["Security & Compliance"] == min(v["Security & Compliance"] for v in s.values())
    assert r["suppliers"][-1]["supplier_name"] == "BrightPath Tech"


def test_criteria_changes_apply_without_code_change(conn, sample_pdfs):
    db.update_criterion(conn, 5, is_active=False)
    db.update_criterion(conn, 1, weight=50)
    r = run_evaluation(conn, MOCK, sample_inputs(sample_pdfs))
    assert len(r["suppliers"][0]["criteria"]) == 4
    assert any("total 110%" in w and "rescaled" in w for w in r["run_warnings"])


def test_corrupt_pdf_excluded_but_run_continues(conn, sample_pdfs):
    inputs = sample_inputs(sample_pdfs)
    inputs.append(SupplierInput("Broken Co", date(2026, 9, 1), 5, (ROOT / "sample_data" / "invalid_samples" / "corrupt_proposal.pdf").read_bytes(), "x.pdf"))
    r = run_evaluation(conn, MOCK, inputs)
    assert len(r["suppliers"]) == 4 and r["status"] == "COMPLETED_WITH_ERRORS"
    assert r["failed_suppliers"][0]["supplier_name"] == "Broken Co"


def test_all_failures_mark_run_failed(conn):
    bad = SupplierInput("Broken Co", date(2026, 9, 1), 5, b"not a pdf", "x.pdf")
    with pytest.raises(InputValidationError):
        run_evaluation(conn, MOCK, [bad])
    assert db.list_runs(conn)[0]["status"] == "FAILED"


def test_input_validation_messages():
    ok = b"%PDF"
    errs = validate_inputs([
        SupplierInput("", date(2026, 1, 1), 5, ok, "a.pdf"),
        SupplierInput("Dup", "not-a-date", 5, ok, "b.pdf"),
        SupplierInput("dup", date(2999, 1, 1), 11, ok, "c.pdf"),
        SupplierInput("NoFile", date(2026, 1, 1), None, None, "d.pdf"),
    ])
    text = " ".join(errs)
    for expected in ["name is required", "YYYY-MM-DD", "duplicate supplier name", "in the future", "experience rating", "PDF file is missing"]:
        assert expected in text
    assert validate_inputs([]) == ["Upload at least one supplier PDF."]


def test_no_active_criteria(conn, sample_pdfs):
    for c in db.get_all_criteria(conn):
        db.update_criterion(conn, c["criterion_id"], is_active=False)
    with pytest.raises(InputValidationError):
        run_evaluation(conn, MOCK, sample_inputs(sample_pdfs))


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._payload


CRIT = [{"criterion_id": 1, "name": "T", "description": "d", "weight": 100, "max_score": 10}]
GOOD = {"supplier_name": "S", "criteria": [{"criterion_id": 1, "score": 5, "max_score": 10}]}


def test_openai_compatible_call_uses_supplied_key(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, auth=headers.get("Authorization"), model=json["model"])
        return FakeResponse(200, {"choices": [{"message": {"content": "```json\n" + __import__("json").dumps(GOOD) + "\n```"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    cfg = LLMConfig(provider="Groq", api_key="my-secret", model="m1", base_url="https://api.groq.com/openai/v1")
    assert llm.evaluate_supplier(cfg, CRIT, "S", "doc") == GOOD
    assert seen == {"url": "https://api.groq.com/openai/v1/chat/completions", "auth": "Bearer my-secret", "model": "m1"}


def test_anthropic_call(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert headers["x-api-key"] == "k" and url.endswith("/v1/messages")
        return FakeResponse(200, {"content": [{"type": "text", "text": __import__("json").dumps(GOOD)}]})

    monkeypatch.setattr(requests, "post", fake_post)
    assert llm.evaluate_supplier(LLMConfig(provider="Anthropic", api_key="k"), CRIT, "S", "doc") == GOOD


def test_missing_key_and_api_errors(monkeypatch):
    with pytest.raises(LLMError, match="API key"):
        llm.evaluate_supplier(LLMConfig(provider="OpenAI", model="m", base_url="https://x"), CRIT, "S", "doc")
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(401, {"error": "bad key"}))
    with pytest.raises(LLMError, match="401"):
        llm.evaluate_supplier(LLMConfig(provider="OpenAI", api_key="k", model="m", base_url="https://x"), CRIT, "S", "doc")


def test_malformed_llm_json_retried_once(conn, sample_pdfs, monkeypatch):
    calls = {"n": 0}

    def flaky(cfg, criteria, name, text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise llm.LLMFormatError("bad json")
        return GOOD | {"criteria": [{"criterion_id": c["criterion_id"], "score": 5} for c in criteria]}

    monkeypatch.setattr("rfp_eval.orchestrator.evaluate_supplier", flaky)
    s, pdf = sample_pdfs[0]
    r = run_evaluation(conn, MOCK, [SupplierInput(s["name"], s["date"], s["rating"], pdf, s["file"])])
    assert calls["n"] == 2 and len(r["suppliers"]) == 1
