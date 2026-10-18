"""Create and seed the SQLite database.   python scripts/init_db.py [--db data/rfp.db] [--reset]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfp_eval import db  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    ap.add_argument("--reset", action="store_true", help="delete the existing database file first")
    args = ap.parse_args()
    path = Path(args.db)
    if args.reset and path.exists():
        path.unlink()
    conn = db.connect(path)
    db.init_db(conn)
    print(f"Database ready at {path}")
    for c in db.get_active_criteria(conn):
        print(f"  [{c['criterion_id']}] {c['name']:<24} weight={c['weight']:g}% max={c['max_score']:g}")


if __name__ == "__main__":
    main()
