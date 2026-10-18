"""Streamlit UI for the Agentic RFP Evaluation & Supplier Ranking mini project."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from rfp_eval import db
from rfp_eval.llm import OPENAI_COMPATIBLE_PRESETS, LLMConfig, LLMError
from rfp_eval.orchestrator import RATING_MAX, RATING_MIN, InputValidationError, SupplierInput, run_evaluation, validate_inputs
from rfp_eval.ranking import TIE_BREAK_RULES
from sample_data.sample_inputs import SAMPLE_SUPPLIERS

ROOT = Path(__file__).resolve().parent
MOCK = "Mock (offline demo)"
PROVIDERS = [MOCK, *OPENAI_COMPATIBLE_PRESETS.keys(), "Anthropic"]

st.set_page_config(page_title="Agentic RFP Evaluation", page_icon="📑", layout="wide")


@st.cache_resource
def get_conn():
    conn = db.connect(check_same_thread=False)
    db.init_db(conn)
    return conn


def secret_key() -> str:
    """API key preconfigured by the host (env var or Streamlit secrets); the UI field overrides it."""
    for name in ("LLM_API_KEY",):
        if os.environ.get(name):
            return os.environ[name]
    try:
        return str(st.secrets.get("LLM_API_KEY", ""))
    except Exception:
        return ""


conn = get_conn()
st.session_state.setdefault("result", None)


# ---------------------------------------------------------------- sidebar: LLM settings
def llm_sidebar() -> LLMConfig:
    st.sidebar.header("🔑 LLM settings")
    provider = st.sidebar.selectbox("Provider", PROVIDERS, help="Any JSON-capable chat model works.")
    if provider == MOCK:
        st.sidebar.info("Offline keyword heuristic - no API key needed. Switch provider to use a real model.")
        return LLMConfig(provider=provider)
    preset = OPENAI_COMPATIBLE_PRESETS.get(provider, {"base_url": "https://api.anthropic.com", "model": "claude-3-5-haiku-latest"})
    api_key = st.sidebar.text_input(
        "API key", type="password", help="Used only for this session's requests. Never stored in the database or logs."
    )
    if not api_key and (preset_key := secret_key()):
        api_key = preset_key
        st.sidebar.caption("Using the key configured in the server environment/secrets.")
    model = st.sidebar.text_input("Model", value=preset["model"], key=f"model-{provider}")
    base_url = st.sidebar.text_input("Base URL", value=preset["base_url"], key=f"url-{provider}")
    return LLMConfig(provider=provider, api_key=api_key, model=model.strip(), base_url=base_url.strip())


cfg = llm_sidebar()

st.title("📑 Agentic RFP Evaluation & Supplier Ranking")
st.caption("The LLM judges proposal content. Deterministic Python does all arithmetic, benchmarking, tie-breaks and ranking.")

tab_criteria, tab_input, tab_results, tab_run, tab_history = st.tabs(
    ["1 · Criteria", "2 · Supplier input", "3 · Leaderboard & scorecards", "4 · Run details", "History"]
)


# ---------------------------------------------------------------- criteria
with tab_criteria:
    active = db.get_active_criteria(conn)
    total_w = sum(c["weight"] for c in active)
    st.subheader("Active evaluation criteria (loaded from SQLite)")
    if active:
        st.dataframe(
            pd.DataFrame(active)[["criterion_id", "name", "description", "weight", "max_score"]].rename(
                columns={"weight": "weight (%)"}
            ),
            hide_index=True,
            width="stretch",
        )
    if not active:
        st.error("No active criteria. Activate at least one below.")
    elif abs(total_w - 100) > 0.01:
        st.warning(f"Active weights total {total_w:g}%, not 100%. They will be rescaled proportionally at run time.")
    else:
        st.success("Active weights total 100%.")

    with st.expander("Edit criteria (activate / deactivate / change weight)"):
        edited = st.data_editor(
            pd.DataFrame(db.get_all_criteria(conn)),
            hide_index=True,
            width="stretch",
            disabled=["criterion_id", "name", "description"],
            column_config={
                "weight": st.column_config.NumberColumn("weight (%)", min_value=0.0, max_value=100.0, step=1.0),
                "max_score": st.column_config.NumberColumn("max_score", min_value=1.0, step=1.0),
                "is_active": st.column_config.CheckboxColumn("active"),
            },
            key="criteria_editor",
        )
        if st.button("Save criteria"):
            for _, r in edited.iterrows():
                db.update_criterion(conn, int(r["criterion_id"]), weight=float(r["weight"]), max_score=float(r["max_score"]), is_active=bool(r["is_active"]))
            st.rerun()


# ---------------------------------------------------------------- input
with tab_input:
    st.subheader("Upload supplier proposals")
    use_samples = st.toggle("Use the bundled sample proposals instead of uploading", value=False)
    files: list[tuple[str, bytes]] = []
    defaults: dict[str, dict] = {}
    if use_samples:
        for s in SAMPLE_SUPPLIERS:
            files.append((s["file"], (ROOT / "sample_data" / "rfps" / s["file"]).read_bytes()))
            defaults[s["file"]] = {"name": s["name"], "date": date.fromisoformat(s["date"]), "rating": s["rating"]}
    else:
        uploads = st.file_uploader("Supplier RFP PDFs", type=["pdf"], accept_multiple_files=True)
        files = [(u.name, u.getvalue()) for u in uploads or []]

    inputs: list[SupplierInput] = []
    for i, (fname, data) in enumerate(files):
        d = defaults.get(fname, {})
        with st.container(border=True):
            st.markdown(f"**{fname}**  ·  {len(data) / 1024:.0f} KB")
            c1, c2, c3 = st.columns([3, 2, 2])
            name = c1.text_input("Supplier name", value=d.get("name", Path(fname).stem.replace("_", " ")), key=f"n-{use_samples}-{i}-{fname}")
            sub = c2.date_input("Submission date", value=d.get("date", date.today()), key=f"d-{use_samples}-{i}-{fname}")
            rating = c3.number_input(
                f"Experience rating ({RATING_MIN:g}-{RATING_MAX:g})", min_value=RATING_MIN, max_value=RATING_MAX,
                value=float(d.get("rating", 5.0)), step=0.5, key=f"r-{use_samples}-{i}-{fname}",
            )
            inputs.append(SupplierInput(name, sub, rating, data, fname))

    errors = validate_inputs(inputs) if inputs else ["Upload at least one supplier PDF (or enable the sample proposals)."]
    needs_key = not cfg.is_mock and cfg.provider != "Ollama (local)" and not cfg.api_key
    if needs_key:
        errors.append(f"Provide an API key for {cfg.provider} in the sidebar (or choose the offline mock provider).")
    for e in errors:
        st.warning(e)

    if st.button("▶ Evaluate", type="primary", disabled=bool(errors)):
        bar = st.progress(0.0, text="Starting")
        try:
            st.session_state["result"] = run_evaluation(conn, cfg, inputs, progress=lambda m, f: bar.progress(min(f, 1.0), text=m))
            bar.empty()
            st.success(f"Run {st.session_state['result']['rfp_run_id']} complete - see the next tabs.")
        except InputValidationError as exc:
            bar.empty()
            for e in exc.errors:
                st.error(e)
        except LLMError as exc:
            bar.empty()
            st.error(f"LLM error: {exc}")


# ---------------------------------------------------------------- results
def render_results(result: dict) -> None:
    sup = result["suppliers"]
    if not sup:
        st.info("This run has no ranked suppliers (it failed).")
        return
    st.subheader("Leaderboard")
    board = pd.DataFrame(
        [
            {
                "Rank": s["final_rank"], "Supplier": s["supplier_name"], "Absolute score (/100)": s["absolute_score"],
                "PPI": s["ppi"], "Submission date": s["submission_date"], "Experience rating": s["experience_rating"],
            }
            for s in sup
        ]
    )
    st.dataframe(board, hide_index=True, width="stretch")
    st.bar_chart(board.set_index("Supplier")[["Absolute score (/100)", "PPI"]])

    st.subheader("Detailed scorecard")
    names = [f"#{s['final_rank']} {s['supplier_name']}" for s in sup]
    chosen = sup[names.index(st.selectbox("Supplier", names))]
    m1, m2, m3 = st.columns(3)
    m1.metric("Rank", chosen["final_rank"])
    m2.metric("Absolute score", f"{chosen['absolute_score']:.2f}")
    m3.metric("PPI", f"{chosen['ppi']:.2f}")
    st.caption(chosen["tie_break_note"])
    table = pd.DataFrame(
        [
            {
                "Criterion": c["criterion_name"], "Score": c["score"], "Max": c["max_score"], "Benchmark": c["benchmark"],
                "Gap": c["gap"], "Relative %": c["relative_pct"], "Weight %": c["effective_weight"],
                "Weighted pts": c["weighted_score"],
            }
            for c in chosen["criteria"]
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    for c in chosen["criteria"]:
        flag = " ⚠️ no valid score" if c["score_missing"] else ""
        with st.expander(f"{c['criterion_name']} - {c['score']:g}/{c['max_score']:g}{flag}"):
            st.markdown(f"**Justification:** {c['justification']}")
            st.markdown(f"**Evidence:** {c['evidence']}")
    st.markdown("**Summary:** " + (chosen["overall_summary"] or "-"))
    if chosen["risks"]:
        st.markdown("**Risks:**\n" + "\n".join(f"- {r}" for r in chosen["risks"]))


def render_run_details(result: dict) -> None:
    if not result["suppliers"]:
        st.info("This run has no ranked suppliers (it failed).")
        return
    st.subheader("Run details")
    c1, c2, c3 = st.columns(3)
    c1.metric("RFP_RUN_ID", result["rfp_run_id"])
    c2.metric("Status", result["status"])
    c3.metric("Model", f"{result.get('llm_provider', '-')} / {result.get('llm_model', '-')}")
    st.markdown("**Warnings**")
    all_warnings = [f"[run] {w}" for w in result.get("run_warnings", [])] + [
        f"[{s['supplier_name']}] {w}" for s in result["suppliers"] for w in s.get("warnings", [])
    ]
    if all_warnings:
        for w in all_warnings:
            st.warning(w)
    else:
        st.success("No warnings: every criterion was returned and valid.")
    st.markdown("**Tie-break explanation**  \nOrder applied: " + " → ".join(TIE_BREAK_RULES))
    st.table(pd.DataFrame(result["tie_break_explanation"]).rename(columns={"rank": "Rank", "supplier_name": "Supplier", "reason": "Reason"}))
    with st.expander("Formulas used"):
        st.json(result["formulas"])
    st.download_button(
        "⬇ Download complete result (JSON)", data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name=f"{result['rfp_run_id']}.json", mime="application/json",
    )


with tab_results:
    if st.session_state["result"]:
        render_results(st.session_state["result"])
    else:
        st.info("Run an evaluation (or open one from History) to see results.")

with tab_run:
    if st.session_state["result"]:
        render_run_details(st.session_state["result"])
    else:
        st.info("No run selected yet.")

with tab_history:
    runs = db.list_runs(conn)
    if not runs:
        st.info("No stored runs yet.")
    else:
        st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
        pick = st.selectbox("Open a stored run", [r["rfp_run_id"] for r in runs])
        if st.button("Load run"):
            st.session_state["result"] = db.load_run(conn, pick)
            st.rerun()
