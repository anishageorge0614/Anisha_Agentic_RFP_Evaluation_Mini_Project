"""SQLite persistence: schema, seed data, criteria access and run storage."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rfp.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    description  TEXT    NOT NULL,
    weight       REAL    NOT NULL CHECK (weight >= 0),
    max_score    REAL    NOT NULL DEFAULT 10 CHECK (max_score > 0),
    is_active    INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS supplier_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id        TEXT NOT NULL REFERENCES rfp_runs(rfp_run_id),
    supplier_name     TEXT NOT NULL,
    submission_date   TEXT NOT NULL,
    experience_rating REAL NOT NULL,
    absolute_score    REAL,
    ppi               REAL,
    final_rank        INTEGER,
    result_json       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_run ON supplier_results(rfp_run_id);
"""

SEED_CRITERIA = [
    ("Technical Capability", "Architecture, integrations, scalability, technical fit", 30, 10),
    ("Implementation Plan", "Timeline, milestones, staffing, risk plan", 20, 10),
    ("Commercial Value", "Pricing clarity, total cost, assumptions", 20, 10),
    ("Security & Compliance", "Controls, certifications, privacy, auditability", 20, 10),
    ("Support & Experience", "Support model, similar projects, references", 10, 10),
]


def connect(db_path: str | Path | None = None, check_same_thread: bool = True) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection, seed: bool = True) -> None:
    conn.executescript(SCHEMA)
    if seed:
        count = conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0]
        if count == 0:
            conn.executemany(
                "INSERT INTO evaluation_criteria (name, description, weight, max_score, is_active) "
                "VALUES (?, ?, ?, ?, 1)",
                SEED_CRITERIA,
            )
    conn.commit()


def get_active_criteria(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT criterion_id, name, description, weight, max_score, is_active "
        "FROM evaluation_criteria WHERE is_active = 1 ORDER BY criterion_id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_criteria(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT criterion_id, name, description, weight, max_score, is_active "
        "FROM evaluation_criteria ORDER BY criterion_id"
    ).fetchall()
    return [dict(r) for r in rows]


def update_criterion(
    conn: sqlite3.Connection,
    criterion_id: int,
    *,
    weight: float | None = None,
    max_score: float | None = None,
    is_active: bool | None = None,
) -> None:
    if weight is not None:
        conn.execute("UPDATE evaluation_criteria SET weight=? WHERE criterion_id=?", (weight, criterion_id))
    if max_score is not None:
        conn.execute("UPDATE evaluation_criteria SET max_score=? WHERE criterion_id=?", (max_score, criterion_id))
    if is_active is not None:
        conn.execute(
            "UPDATE evaluation_criteria SET is_active=? WHERE criterion_id=?", (int(is_active), criterion_id)
        )
    conn.commit()


def create_run(conn: sqlite3.Connection, status: str = "RUNNING") -> str:
    now = datetime.now(timezone.utc)
    run_id = "RFP-" + now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6].upper()
    conn.execute(
        "INSERT INTO rfp_runs (rfp_run_id, created_at, status) VALUES (?, ?, ?)",
        (run_id, now.isoformat(timespec="seconds"), status),
    )
    conn.commit()
    return run_id


def save_run_results(conn: sqlite3.Connection, run_id: str, result: dict[str, Any], status: str = "COMPLETED") -> None:
    """Persist all supplier rows and the run summary in one transaction."""
    with conn:
        conn.execute("DELETE FROM supplier_results WHERE rfp_run_id = ?", (run_id,))
        for s in result["suppliers"]:
            conn.execute(
                "INSERT INTO supplier_results (rfp_run_id, supplier_name, submission_date, experience_rating, "
                "absolute_score, ppi, final_rank, result_json) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    s["supplier_name"],
                    s["submission_date"],
                    s["experience_rating"],
                    s["absolute_score"],
                    s["ppi"],
                    s["final_rank"],
                    json.dumps(s, ensure_ascii=False),
                ),
            )
        summary = {k: v for k, v in result.items() if k != "suppliers"}
        conn.execute(
            "UPDATE rfp_runs SET status=?, summary_json=? WHERE rfp_run_id=?",
            (status, json.dumps(summary, ensure_ascii=False), run_id),
        )


def set_run_status(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    conn.execute("UPDATE rfp_runs SET status=? WHERE rfp_run_id=?", (status, run_id))
    conn.commit()


def list_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT rfp_run_id, created_at, status FROM rfp_runs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def load_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    run = conn.execute("SELECT * FROM rfp_runs WHERE rfp_run_id=?", (run_id,)).fetchone()
    if run is None:
        return None
    rows = conn.execute(
        "SELECT result_json FROM supplier_results WHERE rfp_run_id=? ORDER BY final_rank", (run_id,)
    ).fetchall()
    result = json.loads(run["summary_json"]) if run["summary_json"] else {}
    result.update(
        {
            "rfp_run_id": run["rfp_run_id"],
            "created_at": run["created_at"],
            "status": run["status"],
            "suppliers": [json.loads(r["result_json"]) for r in rows],
        }
    )
    return result
