"""Compare all-or-nothing baseline execution with the reliable scheduler."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from open_deep_research.task_runtime import DagTaskScheduler, ResearchTask


async def run_baseline_trial() -> bool:
    """Simulate upstream gather: one transient error fails the whole batch."""
    attempts = {"search": 0, "verify": 0}

    async def work(task_id: str) -> str:
        attempts[task_id] += 1
        if task_id == "search" and attempts[task_id] == 1:
            raise RuntimeError("injected transient failure")
        return task_id

    try:
        await asyncio.gather(work("search"), work("verify"))
    except Exception:
        return False
    return True


async def run_reliable_trial(trial: int, output: Path) -> tuple[bool, bool]:
    """Run the same transient fault with retry and durable result storage."""
    config = {
        "configurable": {
            "run_id": f"fault-trial-{trial}",
            "runtime_db_path": str(output / f"trial-{trial}.db"),
            "event_log_path": str(output / f"trial-{trial}.jsonl"),
            "fault_injection_rules": [
                {"task_id": "search", "attempt": 1, "kind": "error"}
            ],
        }
    }
    calls = []

    async def worker(task: ResearchTask) -> str:
        calls.append(f"{task.task_id}:{task.attempt_count}")
        return task.task_id

    tasks = [
        ResearchTask(task_id="search", description="search", retry_limit=1),
        ResearchTask(task_id="verify", description="verify", dependencies=["search"]),
    ]
    result = await DagTaskScheduler(max_concurrency=2).run(tasks, worker, config)

    calls_before_resume = list(calls)
    resumed = await DagTaskScheduler(max_concurrency=2).run(tasks, worker, config)
    avoided_duplicate_work = calls == calls_before_resume
    return result.succeeded and resumed.succeeded, avoided_duplicate_work


async def run_reliable_fault_trial(
    trial: int, output: Path, kind: str, persistent: bool = False
) -> bool:
    """Run one named fault model through the durable scheduler.

    ``persistent`` models an error that survives the retry budget. It should
    remain failed; treating that as a recovery failure would hide the useful
    distinction between transient recovery and correct failure propagation.
    """
    rules = [{"task_id": "search", "attempt": 1, "kind": kind}]
    if persistent:
        rules.append({"task_id": "search", "attempt": 2, "kind": kind})
    config = {
        "configurable": {
            "run_id": f"fault-matrix-{kind}-{trial}",
            "runtime_db_path": str(output / f"{kind}-{persistent}-{trial}.db"),
            "event_log_path": str(output / f"{kind}-{persistent}-{trial}.jsonl"),
            "fault_injection_rules": rules,
        }
    }

    async def worker(task: ResearchTask) -> str:
        return task.task_id

    tasks = [
        ResearchTask(task_id="search", description="search", retry_limit=1),
        ResearchTask(task_id="verify", description="verify", dependencies=["search"]),
    ]
    result = await DagTaskScheduler(max_concurrency=2).run(tasks, worker, config)
    return result.succeeded


async def run_fault_matrix(trials: int, output: Path) -> dict[str, dict[str, int]]:
    """Measure recovery and expected failure propagation by fault type."""
    scenarios = {
        "transient_error": ("error", False, True),
        "transient_timeout": ("timeout", False, True),
        "persistent_error": ("error", True, False),
        "persistent_timeout": ("timeout", True, False),
    }
    matrix: dict[str, dict[str, int]] = {}
    for name, (kind, persistent, expected_success) in scenarios.items():
        recovered = 0
        expected = 0
        for trial in range(trials):
            actual = await run_reliable_fault_trial(trial, output, kind, persistent)
            recovered += int(actual)
            expected += int(actual == expected_success)
        matrix[name] = {
            "trials": trials,
            "succeeded": recovered,
            "expected_outcome_matches": expected,
        }
    return matrix


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--matrix", action="store_true", help="Run multiple transient/permanent fault models")
    parser.add_argument("--output", type=Path, default=Path("artifacts/formal_experiment/faults"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.matrix:
        matrix = await run_fault_matrix(args.trials, args.output)
        (args.output / "fault_matrix.json").write_text(
            json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(matrix, ensure_ascii=False, indent=2))
        return

    baseline = []
    reliable = []
    deduplicated = []
    for trial in range(args.trials):
        baseline.append(await run_baseline_trial())
        success, avoided_duplicate_work = await run_reliable_trial(trial, args.output)
        reliable.append(success)
        deduplicated.append(avoided_duplicate_work)

    result = {
        "trials": args.trials,
        "fault_model": "search task fails transiently on its first attempt",
        "baseline_recovery_rate": sum(baseline) / args.trials,
        "reliable_recovery_rate": sum(reliable) / args.trials,
        "resume_without_duplicate_work_rate": sum(deduplicated) / args.trials,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
