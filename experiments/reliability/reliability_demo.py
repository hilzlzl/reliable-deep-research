"""Run a deterministic, zero-API reliability demonstration."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.task_runtime import DagTaskScheduler, ResearchTask


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reliability_demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    db_path = args.output / "runtime.db"
    event_path = args.output / "events.jsonl"
    for path in (db_path, event_path):
        if path.exists():
            path.unlink()

    tasks = [
        ResearchTask(task_id="search", description="Collect sources", retry_limit=1),
        ResearchTask(task_id="verify", description="Verify claims", dependencies=["search"]),
        ResearchTask(task_id="report", description="Write report", dependencies=["verify"]),
    ]
    executions: list[str] = []

    async def worker(task: ResearchTask) -> dict[str, str]:
        executions.append(f"{task.task_id}:{task.attempt_count}")
        return {"output": task.description}

    base = {
        "runtime_db_path": str(db_path),
        "event_log_path": str(event_path),
        "run_id": "reliability-demo",
    }
    injected_config = {
        "configurable": {
            **base,
            "fault_injection_rules": [
                {"task_id": "search", "attempt": 1, "kind": "error"}
            ],
        }
    }
    first = await DagTaskScheduler(max_concurrency=2).run(tasks, worker, injected_config)

    before_replay = list(executions)
    replay_config = {
        "configurable": {**base, "replay_task_ids": ["verify"]}
    }
    second = await DagTaskScheduler(max_concurrency=2).run(tasks, worker, replay_config)
    store = RuntimeStore(db_path)
    report = {
        "first_run_succeeded": first.succeeded,
        "first_run_executions": before_replay,
        "replay_requested": "verify",
        "replay_executions": executions[len(before_replay) :],
        "second_run_succeeded": second.succeeded,
        "task_counts": store.task_counts(),
        "event_count": store.event_count(),
    }
    (args.output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
