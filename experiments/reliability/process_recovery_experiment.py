"""Measure task-level recovery across an actual process exit and restart."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from open_deep_research.evidence import ClaimStatus, Document, Evidence, ResearchClaim
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.task_runtime import DagTaskScheduler, ResearchTask


SCENARIOS = ("after_completed_task", "during_evidence_write")
CRASH_EXIT_CODE = 71


def record_call(root: Path, task_id: str, phase: str) -> None:
    """Flush call records before an intentional hard exit."""
    payload = json.dumps({"task_id": task_id, "phase": phase}) + "\n"
    fd = os.open(root / "calls.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def persist_root_evidence(store: RuntimeStore, phase: str) -> None:
    document_id = f"doc:{phase}"
    evidence_id = f"ev:{phase}"
    store.save_document(Document(
        document_id=document_id,
        url=f"https://example.test/{phase}",
        title=phase,
        content_hash=f"hash:{phase}",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    ))
    store.save_evidence(Evidence(
        evidence_id=evidence_id,
        task_id="root",
        document_id=document_id,
        excerpt=f"Evidence from {phase}",
        claim_text=f"Claim from {phase}",
    ))
    store.save_claim(ResearchClaim(
        claim_id=f"claim:{phase}",
        statement=f"Claim from {phase}",
        evidence_ids=[evidence_id],
        status=ClaimStatus.SUPPORTED,
    ))


async def run_worker(root: Path, mode: str, scenario: str, phase: str) -> None:
    store = RuntimeStore(root / "runtime.db") if mode == "reliable" else None

    async def work(task: ResearchTask) -> str:
        record_call(root, task.task_id, phase)
        if task.task_id == "root":
            if store is not None:
                persist_root_evidence(store, phase)
            if phase == "initial" and scenario == "during_evidence_write":
                os._exit(CRASH_EXIT_CODE)
        elif phase == "initial" and scenario == "after_completed_task":
            os._exit(CRASH_EXIT_CODE)
        return task.task_id

    if mode == "baseline":
        # The control repeats both units after restart, as the original gather
        # path has no task-result store. An event makes the ordering reproducible.
        root_done = asyncio.Event()

        async def baseline_root() -> str:
            result = await work(ResearchTask(task_id="root", description="root"))
            root_done.set()
            return result

        async def baseline_child() -> str:
            await root_done.wait()
            return await work(ResearchTask(task_id="child", description="child"))

        await asyncio.gather(baseline_root(), baseline_child())
    else:
        tasks = [
            ResearchTask(task_id="root", description="root", retry_limit=1),
            ResearchTask(task_id="child", description="child", dependencies=["root"], retry_limit=1),
        ]
        result = await DagTaskScheduler(max_concurrency=2).run(
            tasks,
            work,
            {"configurable": {"runtime_db_path": str(root / "runtime.db"), "observability_enabled": False}},
        )
        if not result.succeeded:
            raise RuntimeError("Resumed scheduler did not finish both tasks")


def run_trial(root: Path, mode: str, scenario: str) -> dict:
    root.mkdir(parents=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--root", str(root),
        "--mode", mode,
        "--scenario", scenario,
    ]
    first = subprocess.run(
        [*command, "--phase", "initial"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if first.returncode != CRASH_EXIT_CODE:
        raise RuntimeError(f"Expected hard exit {CRASH_EXIT_CODE}, got {first.returncode}: {first.stderr[-500:]}")
    started = time.perf_counter()
    resumed = subprocess.run(
        [*command, "--phase", "resumed"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    elapsed = round(time.perf_counter() - started, 3)
    if resumed.returncode:
        raise RuntimeError(f"Resume failed: {resumed.stderr[-1000:]}")

    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    root_calls = sum(item["task_id"] == "root" for item in calls)
    evidence_ids: list[str] | None = None
    claim_ids: list[str] | None = None
    if mode == "reliable":
        store = RuntimeStore(root / "runtime.db")
        if store.task_counts() != (2, 2):
            raise RuntimeError("Committed task count was not restored")
        evidence_ids = [item["evidence_id"] for item in store.list_evidence()]
        claim_ids = [item["claim_id"] for item in store.list_claims()]
    return {
        "resumed_successfully": True,
        "root_worker_calls": root_calls,
        "duplicated_completed_root": max(0, root_calls - 1) if scenario == "after_completed_task" else None,
        "resume_seconds": elapsed,
        "evidence_ids": evidence_ids,
        "claim_ids": claim_ids,
        "stale_evidence": int("ev:initial" in evidence_ids) if scenario == "during_evidence_write" and evidence_ids is not None else None,
        "stale_claim": int("claim:initial" in claim_ids) if scenario == "during_evidence_write" and claim_ids is not None else None,
    }


def run_experiment(trials: int, output: Path) -> dict:
    if trials < 1:
        raise ValueError("trials must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for scenario in SCENARIOS:
        for mode in ("baseline", "reliable"):
            for trial in range(trials):
                run_dir = output / scenario / mode / f"trial-{trial:03d}"
                result = run_trial(run_dir, mode, scenario)
                results.append({"scenario": scenario, "mode": mode, "trial": trial, **result})
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for mode in ("baseline", "reliable"):
            group = [item for item in results if item["scenario"] == scenario and item["mode"] == mode]
            summary[scenario][mode] = {
                "trials": len(group),
                "recovered": sum(item["resumed_successfully"] for item in group),
                "mean_root_worker_calls": statistics.mean(item["root_worker_calls"] for item in group),
                "mean_resume_seconds": round(statistics.mean(item["resume_seconds"] for item in group), 3),
                "duplicate_completed_root_calls": (
                    sum(item["duplicated_completed_root"] or 0 for item in group)
                    if scenario == "after_completed_task" else None
                ),
                "stale_evidence_records": (
                    sum(item["stale_evidence"] or 0 for item in group)
                    if mode == "reliable" and scenario == "during_evidence_write" else None
                ),
                "stale_claim_records": (
                    sum(item["stale_claim"] or 0 for item in group)
                    if mode == "reliable" and scenario == "during_evidence_write" else None
                ),
            }
    payload = {
        "scope": "Task-level SQLite recovery after os._exit; not LangGraph graph-checkpoint recovery or exactly-once external API execution.",
        "trials_per_scenario_per_mode": trials,
        "summary": summary,
        "runs": results,
    }
    (output / "evaluation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/process_recovery"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("baseline", "reliable"), help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("initial", "resumed"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not all((args.root, args.mode, args.scenario, args.phase)):
            parser.error("Worker invocation is incomplete")
        asyncio.run(run_worker(args.root, args.mode, args.scenario, args.phase))
        return
    payload = run_experiment(args.trials, args.output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
