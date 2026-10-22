"""Deterministic reliability and replay experiments."""

import asyncio

from open_deep_research.task_runtime import DagTaskScheduler, ResearchTask, TaskStatus


def test_injected_failure_recovers_on_retry():
    calls = []

    async def worker(task):
        calls.append((task.task_id, task.attempt_count))
        return "recovered"

    task = ResearchTask(task_id="flaky", description="flaky", retry_limit=1)
    config = {
        "configurable": {
            "observability_enabled": False,
            "fault_injection_rules": [
                {"task_id": "flaky", "attempt": 1, "kind": "error"}
            ],
        }
    }
    result = asyncio.run(DagTaskScheduler().run([task], worker, config))

    assert result.tasks["flaky"].status == TaskStatus.SUCCEEDED
    assert result.results["flaky"].attempts == 2
    assert calls == [("flaky", 2)]


def test_replay_invalidates_transitive_dependents(tmp_path):
    db_path = tmp_path / "replay.db"
    calls = []
    tasks = [
        ResearchTask(task_id="root", description="root"),
        ResearchTask(task_id="child", description="child", dependencies=["root"]),
        ResearchTask(task_id="leaf", description="leaf", dependencies=["child"]),
    ]

    async def worker(task):
        calls.append(task.task_id)
        return f"result-{len(calls)}"

    base_config = {
        "configurable": {
            "runtime_db_path": str(db_path),
            "observability_enabled": False,
        }
    }
    asyncio.run(DagTaskScheduler(max_concurrency=1).run(tasks, worker, base_config))
    assert calls == ["root", "child", "leaf"]

    replay_config = {
        "configurable": {
            **base_config["configurable"],
            "replay_task_ids": ["root"],
        }
    }
    asyncio.run(DagTaskScheduler(max_concurrency=1).run(tasks, worker, replay_config))

    assert calls == ["root", "child", "leaf", "root", "child", "leaf"]
