# Agentic RFP Evaluation & Supplier Ranking

An AI-assisted Streamlit app that reads supplier RFP PDFs, has an LLM score each one against configurable
criteria, then uses **deterministic Python** for benchmarking, the Peer Performance Index (PPI), tie-breaks and ranking.
Results are stored in SQLite and can be downloaded as JSON.

**You supply the API key at runtime** (sidebar). Any JSON-capable chat model works. There is also an **offline mock
provider** that needs no key, so you can try everything immediately.

## Quick start

```powershell
pip install -r requirements.txt
python scripts/init_db.py          # optional: the app also creates/seeds data/rfp.db on first start
streamlit run app.py
```

1. Pick a provider in the sidebar and paste your API key (or leave **Mock (offline demo)** selected).
2. Open **Supplier input**, switch on *Use the bundled sample proposals* (or upload your own PDFs), adjust metadata, click **Evaluate**.
3. Browse **Leaderboard & scorecards** and **Run details** (warnings, tie-break explanation, JSON download).

### Supplying an API key

| Provider | Notes |
|---|---|
| OpenAI, Google Gemini, Groq, OpenRouter, Ollama, *Custom OpenAI-compatible* | Uses the `/chat/completions` protocol. Base URL and model are editable. |
| Anthropic | Messages API. |
| Mock (offline demo) | Keyword heuristic, no key, no network. Only for exercising the pipeline. |

The key is typed into a password field, held in memory for the request, and never written to the database, the JSON
export or logs. For a hosted deployment you can instead preconfigure a key with the `LLM_API_KEY` environment variable or
Streamlit secret (see `.streamlit/secrets.toml.example`); a key typed in the sidebar overrides it.
Any other OpenAI-compatible service works via *Custom OpenAI-compatible* + its base URL.

CLI equivalent: `python scripts/run_sample.py --provider OpenAI --model gpt-4o-mini --api-key sk-...`

## Architecture

```
app.py (Streamlit UI)
  └─ rfp_eval/orchestrator.py   Orchestrator Agent: runs the workflow in order
        ├─ document_tool.py     Document Tool: PDF -> clean text (pypdf)
        ├─ llm.py (+mock_llm)   Evaluation Agent: dynamic prompt, provider call, JSON parsing
        ├─ validation.py        Validation Tool: schema check, fill missing, clip scores, warnings (Pydantic)
        ├─ ranking.py           Ranking Tool: formulas, benchmarks, PPI, tie-breaks (pure Python, no LLM)
        └─ db.py                SQLite: criteria, runs, results
```

Data flow: Setup (load criteria) → Input → Batch (`RFP_RUN_ID`) → Evaluate (reload criteria, extract, prompt, LLM) →
Validate → Score → Benchmark → Rank → Persist → Present.

The LLM only produces per-criterion scores, justifications and evidence. It is never asked for totals, benchmarks or ranks.

## Formulas

Let *w* be a criterion's effective weight (active weights rescaled to total 100).

| Metric | Formula |
|---|---|
| Absolute weighted score | Σ (score / max_score) × w |
| Criterion benchmark | highest **valid** score for the criterion across all suppliers in the run |
| Criterion gap | score − benchmark (0 for the leader, otherwise negative) |
| Relative % | (score / benchmark) × 100; **if benchmark = 0 → 100** (nobody outperformed anybody) |
| PPI | Σ (relative % × w) / 100 |

**Tie-break order:** higher PPI → earlier submission date → higher experience rating → supplier name ascending
(case-insensitive). Ranks 1, 2, 3… are assigned after this sort. PPI is rounded to 4 decimals before comparison.
Each supplier's scorecard states which rule placed it below the previous supplier.

## Validation rules

* Output not a JSON object / unparsable → one automatic retry, then the supplier is excluded with an error.
* Every active criterion gets exactly one entry; missing, non-numeric (`"eight"`) or `NaN` scores are marked invalid:
  **counted as 0 in totals, excluded from benchmarks**, with a warning. `"7/10"` is accepted as 7.
* Scores outside `0..max_score` are clipped and flagged. Configured `max_score` always wins over the model's.
* Duplicate, unknown/inactive criterion ids and malformed entries are ignored with a warning.
* Input checks (shown before **Evaluate** is enabled): name required and unique, date valid and not in the future,
  rating 0–10, PDF present and readable, API key present for hosted providers.

## SQLite schema (`data/rfp.db`)

* `evaluation_criteria(criterion_id, name, description, weight, max_score, is_active)` – seeded with the 5 example criteria (30/20/20/20/10).
  Edit them in the app (**Criteria → Edit criteria**) or in SQL; prompts are rebuilt from the table on every run.
* `rfp_runs(rfp_run_id, created_at, status, summary_json)`
* `supplier_results(rfp_run_id, supplier_name, submission_date, experience_rating, absolute_score, ppi, final_rank, result_json)`

## Sample / synthetic data

* `sample_data/rfps/` – four fictional 2-page proposals (Apex Systems, BrightPath Tech, NexaWorks, Orbit Digital) for the same
  request, each with the strengths/weaknesses from the brief. Regenerate with `python scripts/generate_sample_pdfs.py`.
* `sample_data/sample_inputs.py` – submission dates and experience ratings for them.
* `sample_data/sample_export.json` – exported result of one completed run (offline mock provider).
* `sample_data/invalid_samples/` – `corrupt_proposal.pdf` (unreadable PDF) and `malformed_llm_output.json`
  (out-of-range, non-numeric, duplicate and unknown criteria) for error-case demos.

### Demo script (success + error case)

1. Success: sample proposals toggle → **Evaluate** → leaderboard, scorecards, JSON download.
2. Validation case: clear a supplier name, or choose OpenAI with no key → warning shown and **Evaluate** disabled.
3. Error case: upload `corrupt_proposal.pdf` alongside valid PDFs → it is excluded, the run completes as `COMPLETED_WITH_ERRORS`, warning shown in **Run details**.

## Tests

```powershell
python -m pytest
```

42 tests cover formulas (hand-calculated), each tie-break level, zero benchmarks, determinism, validation edge cases,
JSON extraction, PDF extraction, persistence, input validation, provider requests (mocked HTTP) and the retry path.

## Deploy to Streamlit Community Cloud

Push this folder to GitHub, create an app pointing at `app.py`, and (optionally) add `LLM_API_KEY` under *Secrets*.
Community Cloud storage is ephemeral, so `data/rfp.db` is recreated and re-seeded after restarts.

## Assumptions

* Experience rating scale is 0–10. Cost is part of the *Commercial Value* criterion, judged by the LLM; price is not computed separately.
* Documents are truncated to 60,000 characters for the prompt. Scanned (image-only) PDFs are not OCR'd and are rejected.
* Weights not totalling 100 are rescaled proportionally with a warning rather than blocking the run.
* Screenshots: take them from your own deployed app and add them to the `docs/` folder before submitting.
