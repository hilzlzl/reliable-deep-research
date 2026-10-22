"""Behavioral tests for the explicit DAG task runtime."""

import asyncio

import pytest

from open_deep_research.task_runtime import DagTaskScheduler, ResearchTask, TaskStatus
from open_deep_research.run_budget import RunBudget

NO_EVENTS = {"configurable": {"observability_enabled": False}}


def test_scheduler_respects_dependencies_and_priority():
    order = []

    async def worker(task):
        order.append(task.task_id)
        return task.task_id.upper()

    tasks = [
        ResearchTask(task_id="root", description="root"),
        ResearchTask(task_id="low", description="low", dependencies=["root"]),
        ResearchTask(task_id="high", description="high", dependencies=["root"], priority=10),
    ]
    result = asyncio.run(DagTaskScheduler(max_concurrency=1).run(tasks, worker, NO_EVENTS))

    assert order == ["root", "high", "low"]
    assert result.succeeded
    assert result.results["high"].content == "HIGH"


def test_scheduler_retries_then_succeeds():
    attempts = 0

    async def flaky_worker(task):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return "recovered"

    task = ResearchTask(task_id="flaky", description="flaky", retry_limit=1)
    result = asyncio.run(DagTaskScheduler().run([task], flaky_worker, NO_EVENTS))

    assert result.tasks["flaky"].status == TaskStatus.SUCCEEDED
    assert result.results["flaky"].attempts == 2


def test_failed_task_blocks_dependents():
    async def worker(task):
        if task.task_id == "fails":
            raise RuntimeError("boom")
        return "should not run"

    tasks = [
        ResearchTask(task_id="fails", description="fails", retry_limit=0),
        ResearchTask(task_id="dependent", description="dependent", dependencies=["fails"]),
    ]
    result = asyncio.run(DagTaskScheduler().run(tasks, worker, NO_EVENTS))

    assert result.tasks["fails"].status == TaskStatus.FAILED
    assert result.tasks["dependent"].status == TaskStatus.BLOCKED
    assert "dependent" not in result.results


def test_scheduler_enforces_timeout():
    async def slow_worker(task):
        await asyncio.sleep(0.03)

    task = ResearchTask(
        task_id="slow",
        description="slow",
        timeout_seconds=0.001,
        retry_limit=0,
    )
    result = asyncio.run(DagTaskScheduler().run([task], slow_worker, NO_EVENTS))

    assert result.tasks["slow"].status == TaskStatus.FAILED
    assert result.tasks["slow"].error.startswith("TimeoutError")


def test_scheduler_rejects_cycles_and_missing_dependencies():
    async def worker(task):
        return None

    cyclic = [
        ResearchTask(task_id="a", description="a", dependencies=["b"]),
        ResearchTask(task_id="b", description="b", dependencies=["a"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        asyncio.run(DagTaskScheduler().run(cyclic, worker, NO_EVENTS))

    missing = [ResearchTask(task_id="a", description="a", dependencies=["missing"])]
    with pytest.raises(ValueError, match="Unknown task dependencies"):
        asyncio.run(DagTaskScheduler().run(missing, worker, NO_EVENTS))


def test_scheduler_cancels_remaining_tasks_when_attempt_budget_is_exhausted():
    async def worker(task):
        return task.task_id

    tasks = [
        ResearchTask(task_id="first", description="first"),
        ResearchTask(task_id="second", description="second"),
    ]
    config = {
        "configurable": {
            "observability_enabled": False,
            "run_budget": RunBudget(max_task_attempts=1),
        }
    }

    result = asyncio.run(DagTaskScheduler(max_concurrency=1).run(tasks, worker, config))

    assert result.tasks["first"].status == TaskStatus.SUCCEEDED
    assert result.tasks["second"].status == TaskStatus.CANCELLED
    assert "task_attempts" in result.tasks["second"].error
