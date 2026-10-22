"""Diagnose coverage_v1 against immutable saved evidence, without new retrieval.

This is a report-stage replay, not an end-to-end benchmark. Only question text
reaches the generator; reference points are reserved for separate evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from open_deep_research.configuration import Configuration
from open_deep_research.coverage_reporting import (
    decompose_questions,
    generate_coverage_report,
)
from open_deep_research.run_budget import RunBudget


def load_evidence_readonly(database: Path) -> list[dict]:
    """Read a saved DB through SQLite's immutable, read-only connection."""
    import sqlite3

    # RuntimeStore initialization may run migrations. Snapshot into memory
    # through SQLite backup instead, leaving the original database untouched.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(":memory:") as target:
            source.backup(target)
            target.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in target.execute(
                    "SELECT e.*, d.url AS source_url, d.title AS source_title "
                    "FROM evidence e JOIN documents d ON e.document_id = d.document_id "
                    "ORDER BY e.evidence_id"
                )
            ]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="EXPERIMENT_API_KEY")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen diagnostic file changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    model = ChatOpenAI(
        model=manifest["model"],
        base_url=manifest["base_url"],
        api_key=os.environ[args.api_key_env],
        reasoning_effort="low",
        streaming=True,
        stream_usage=True,
        max_tokens=6000,
        timeout=180,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(manifest["workers"])

    async def run(row):
        async with semaphore:
            case_dir = args.output / row["case_id"]
            case_dir.mkdir()
            evidence = load_evidence_readonly(Path(row["database"]))
            budget = RunBudget(
                max_llm_calls=8,
                max_prompt_tokens=250000,
                max_completion_tokens=40000,
                max_total_seconds=600,
            )
            config = {
                "configurable": {
                    "observability_enabled": True,
                    "event_log_path": str(case_dir / "events.jsonl"),
                    "run_id": row["case_id"],
                    "run_budget": budget,
                }
            }
            options = Configuration(
                native_structured_output=False,
                max_structured_output_retries=1,
                enable_gap_search=False,
            )
            started = time.monotonic()
            result = {}
            error = None
            try:
                questions = await decompose_questions(
                    model, row["question"], native=False, retries=1, config=config
                )
                (case_dir / "requirements.json").write_text(
                    json.dumps(
                        [q.model_dump() for q in questions],
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )

                async def no_search(_):
                    raise AssertionError("Fixed-evidence replay must never retrieve")

                result = await generate_coverage_report(
                    model,
                    questions=questions,
                    evidence=evidence,
                    config=config,
                    configurable=options,
                    gap_search=no_search,
                    reload_evidence=lambda: evidence,
                )
            except Exception as exception:
                # Keep errors local and redact the active credential before any output.
                message = str(exception).replace(
                    os.environ[args.api_key_env], "[REDACTED]"
                )
                error = {"type": type(exception).__name__, "message": message}
            status = (
                "failed"
                if error
                else "degraded"
                if result["report_quality_status"] == "degraded_needs_review"
                else "succeeded"
            )
            summary = {
                "case_id": row["case_id"],
                "profile": "coverage_v1",
                "mode": "fixed_evidence_report_replay",
                "status": status,
                "error": error,
                "report_quality_status": result.get("report_quality_status"),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "budget": budget.snapshot().model_dump(),
                "evidence_cards": len(evidence),
                "source_report": row["source_report"],
            }
            (case_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (case_dir / "report.md").write_text(result.get("final_report", ""))
            print(row["case_id"], status, flush=True)
            return summary

    summaries = await asyncio.gather(*(run(row) for row in manifest["cases"]))
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
