"""Inspect a baseline artifact directory without making external API calls."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir

    required = ("input.json", "events.jsonl", "runtime.db", "report.md", "summary.json")
    missing = [name for name in required if not (run_dir / name).exists()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    task_count = successful_tasks = result_count = evidence_count = claim_count = 0
    with sqlite3.connect(run_dir / "runtime.db") as connection:
        if table_exists(connection, "tasks"):
            task_count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            successful_tasks = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE status='SUCCEEDED'"
            ).fetchone()[0]
        if table_exists(connection, "task_results"):
            result_count = connection.execute("SELECT COUNT(*) FROM task_results").fetchone()[0]
        if table_exists(connection, "evidence"):
            evidence_count = connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        if table_exists(connection, "claims"):
            claim_count = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

    checks = {
        "artifacts_complete": not missing,
        "run_succeeded": summary.get("status") == "succeeded",
        "report_generated": len(report) > 0,
        "run_finished_event": any(event.get("event_type") == "run_finished" for event in events),
        "research_executed": task_count > 0,
        "research_results_saved": successful_tasks > 0 and result_count > 0,
        "evidence_persisted": evidence_count > 0,
    }
    print(json.dumps({
        "run_id": summary.get("run_id"),
        "checks": checks,
        "metrics": {
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "report_characters": len(report),
            "events": len(events),
            "research_tasks": task_count,
            "successful_research_tasks": successful_tasks,
            "saved_results": result_count,
            "evidence": evidence_count,
            "claims": claim_count,
        },
        "missing": missing,
        "full_acceptance": all(checks.values()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
