"""Command-line run of the full pipeline on the sample PDFs (offline mock by default).

  python scripts/run_sample.py                       # mock LLM, writes sample_data/sample_export.json
  python scripts/run_sample.py --provider OpenAI --model gpt-4o-mini --api-key sk-...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rfp_eval import db  # noqa: E402
from rfp_eval.llm import OPENAI_COMPATIBLE_PRESETS, LLMConfig  # noqa: E402
from rfp_eval.orchestrator import SupplierInput, run_evaluation  # noqa: E402
from sample_data.sample_inputs import SAMPLE_SUPPLIERS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="Mock (offline demo)")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--db", default=":memory:", help="SQLite path (default: throwaway in-memory DB)")
    ap.add_argument("--out", default=str(ROOT / "sample_data" / "sample_export.json"))
    args = ap.parse_args()

    preset = OPENAI_COMPATIBLE_PRESETS.get(args.provider, {})
    cfg = LLMConfig(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model or preset.get("model", ""),
        base_url=args.base_url or preset.get("base_url", ""),
    )
    conn = db.connect(args.db)
    db.init_db(conn)
    suppliers = [
        SupplierInput(s["name"], s["date"], s["rating"], (ROOT / "sample_data" / "rfps" / s["file"]).read_bytes(), s["file"])
        for s in SAMPLE_SUPPLIERS
    ]
    result = run_evaluation(conn, cfg, suppliers, progress=lambda m, f: print(f"[{f:4.0%}] {m}"))
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRun {result['rfp_run_id']} -> {args.out}\n")
    for s in result["suppliers"]:
        print(f"#{s['final_rank']} {s['supplier_name']:<16} PPI={s['ppi']:>8.4f}  abs={s['absolute_score']:>7.2f}")
        print("    " + "  ".join(f"{c['criterion_name'].split()[0]}={c['score']:g}" for c in s["criteria"]))


if __name__ == "__main__":
    main()
