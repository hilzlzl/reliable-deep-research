"""Typed task model and bounded DAG executor for research work."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from open_deep_research.observability import record_event
from open_deep_research.fault_injection import inject_configured_fault
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.run_budget import BudgetExceeded, budget_from_config


class TaskStatus(str, Enum):
    """Lifecycle states for one explicit research task."""

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class ResearchTask(BaseModel):
    """A schedulable unit of research with explicit runtime policy."""

    task_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    priority: int = 0
    timeout_seconds: float = Field(default=120.0, gt=0)
    retry_limit: int = Field(default=1, ge=0)
    status: TaskStatus = TaskStatus.PENDING
    attempt_count: int = Field(default=0, ge=0)
    result_id: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_dependencies(self) -> "ResearchTask":
        if self.task_id in self.dependencies:
            raise ValueError("A task cannot depend on itself")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("Task dependencies must be unique")
        return self


class TaskResult(BaseModel):
    """Successful output and attempt count for a research task."""

    task_id: str
    content: Any
    attempts: int


class SchedulerResult(BaseModel):
    """Complete scheduler snapshot returned even when some tasks fail."""

    tasks: dict[str, ResearchTask]
    results: dict[str, TaskResult]

    @property
    def succeeded(self) -> bool:
        return all(task.status == TaskStatus.SUCCEEDED for task in self.tasks.values())


TaskWorker = Callable[[ResearchTask], Awaitable[Any]]


class DagTaskScheduler:
    """Execute dependency-ready tasks with bounded concurrency and retries."""

    def __init__(self, max_concurrency: int = 3):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.max_concurrency = max_concurrency

    async def run(
        self,
        tasks: Iterable[ResearchTask],
        worker: TaskWorker,
        config: RunnableConfig | None = None,
    ) -> SchedulerResult:
        task_map = self._validate_graph(tasks)
        results: dict[str, TaskResult] = {}
        store = None
        db_path = (config or {}).get("configurable", {}).get("runtime_db_path")
        if db_path:
            store = RuntimeStore(db_path)
            replay_ids = set(
                (config or {}).get("configurable", {}).get("replay_task_ids", [])
            )
            replay_ids = self._dependent_closure(task_map, replay_ids)
            if replay_ids:
                invalidation = store.reset_tasks_for_replay(replay_ids)
                record_event(
                    config,
                    "task_replay_requested",
                    "task_scheduler",
                    data={"task_ids": sorted(replay_ids), **invalidation},
                )
            for task in task_map.values():
                snapshot = store.task_snapshot(task.task_id)
                saved_result = store.result_snapshot(task.task_id)
                if snapshot and snapshot["status"] == TaskStatus.SUCCEEDED.value and saved_result:
                    task.status = TaskStatus.SUCCEEDED
                    task.attempt_count = int(snapshot["attempt_count"])
                    task.result_id = snapshot["result_id"]
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        content=saved_result["content"],
                        attempts=int(saved_result["attempts"]),
                    )
                    record_event(
                        config,
                        "task_restored",
                        "task_scheduler",
                        data={"task_id": task.task_id, "result_id": task.result_id},
                    )
                else:
                    # RUNNING means the process stopped before committing a
                    # result. Its Evidence is likewise uncommitted and must be
                    # invalidated before requeueing to prevent mixed lineage.
                    if snapshot:
                        invalidated = store.invalidate_derived_artifacts({task.task_id})
                        if any(invalidated.values()):
                            record_event(
                                config,
                                "uncommitted_artifacts_invalidated",
                                "task_scheduler",
                                data={"task_id": task.task_id, **invalidated},
                            )
                        task.attempt_count = min(
                            int(snapshot["attempt_count"]), task.retry_limit
                        )
                    task.status = TaskStatus.PENDING
                    store.save_task(task)

        while self._has_unfinished_tasks(task_map):
            budget = budget_from_config(config)
            try:
                if budget:
                    budget.check_time()
            except BudgetExceeded as error:
                self._cancel_unfinished(task_map, config, store, str(error))
                break
            self._propagate_blocked(task_map, config, store)
            ready = self._ready_tasks(task_map)
            if not ready:
                break
            batch = ready[: self.max_concurrency]
            await asyncio.gather(
                *(self._execute_task(task, worker, results, config, store) for task in batch)
            )

        return SchedulerResult(tasks=task_map, results=results)

    async def _execute_task(
        self,
        task: ResearchTask,
        worker: TaskWorker,
        results: dict[str, TaskResult],
        config: RunnableConfig | None,
        store: RuntimeStore | None,
    ) -> None:
        for attempt in range(task.attempt_count + 1, task.retry_limit + 2):
            budget = budget_from_config(config)
            try:
                if budget:
                    budget.consume("task_attempts")
            except BudgetExceeded as error:
                task.status = TaskStatus.CANCELLED
                task.error = str(error)
                self._persist(store, task, "task_cancelled", {"reason": str(error)})
                record_event(config, "task_cancelled", "task_scheduler", data={"task_id": task.task_id, "reason": str(error)})
                return
            task.attempt_count = attempt
            task.status = TaskStatus.RUNNING
            self._persist(store, task, "task_started", {"attempt": attempt})
            record_event(
                config,
                "task_started",
                "task_scheduler",
                data={"task_id": task.task_id, "attempt": attempt},
            )
            try:
                await inject_configured_fault(task.task_id, attempt, config)
                content = await asyncio.wait_for(worker(task), timeout=task.timeout_seconds)
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                self._persist(store, task, "task_cancelled", {})
                record_event(config, "task_cancelled", "task_scheduler", data={"task_id": task.task_id})
                raise
            except Exception as error:
                task.error = f"{type(error).__name__}: {error}"
                if store:
                    # Search tools persist evidence before the task result commits.
                    # A failed attempt must not contribute evidence to a retry or report.
                    invalidated = store.invalidate_derived_artifacts({task.task_id})
                    if any(invalidated.values()):
                        record_event(
                            config,
                            "failed_attempt_artifacts_invalidated",
                            "task_scheduler",
                            data={"task_id": task.task_id, "attempt": attempt, **invalidated},
                        )
                if attempt <= task.retry_limit:
                    task.status = TaskStatus.RETRYING
                    self._persist(store, task, "task_retrying", {"attempt": attempt, "error": task.error})
                    record_event(
                        config,
                        "task_retrying",
                        "task_scheduler",
                        data={"task_id": task.task_id, "attempt": attempt, "error": task.error},
                    )
                    continue
                task.status = TaskStatus.FAILED
                self._persist(store, task, "task_failed", {"attempt": attempt, "error": task.error})
                record_event(
                    config,
                    "task_failed",
                    "task_scheduler",
                    data={"task_id": task.task_id, "attempt": attempt, "error": task.error},
                )
                return

            task.status = TaskStatus.SUCCEEDED
            task.error = None
            task.result_id = f"result:{task.task_id}"
            results[task.task_id] = TaskResult(
                task_id=task.task_id,
                content=content,
                attempts=attempt,
            )
            if store:
                # The task snapshot and result must become visible together;
                # otherwise a crash can restore SUCCEEDED without content.
                store.commit_task_success(task, results[task.task_id])
                store.append_event(task.task_id, "task_succeeded", {"attempt": attempt})
            record_event(
                config,
                "task_succeeded",
                "task_scheduler",
                data={"task_id": task.task_id, "attempt": attempt, "result_id": task.result_id},
            )
            return

    @staticmethod
    def _validate_graph(tasks: Iterable[ResearchTask]) -> dict[str, ResearchTask]:
        task_list = [task.model_copy(deep=True) for task in tasks]
        task_map = {task.task_id: task for task in task_list}
        if len(task_map) != len(task_list):
            raise ValueError("Task ids must be unique")
        missing = {
            dependency
            for task in task_list
            for dependency in task.dependencies
            if dependency not in task_map
        }
        if missing:
            raise ValueError(f"Unknown task dependencies: {', '.join(sorted(missing))}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("Task graph contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in task_map[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_map:
            visit(task_id)
        return task_map

    @staticmethod
    def _dependent_closure(
        tasks: dict[str, ResearchTask], requested: set[str]
    ) -> set[str]:
        unknown = requested - tasks.keys()
        if unknown:
            raise ValueError(f"Unknown replay task ids: {', '.join(sorted(unknown))}")
        invalidated = set(requested)
        changed = True
        while changed:
            changed = False
            for task in tasks.values():
                if task.task_id not in invalidated and invalidated.intersection(task.dependencies):
                    invalidated.add(task.task_id)
                    changed = True
        return invalidated

    @staticmethod
    def _has_unfinished_tasks(tasks: dict[str, ResearchTask]) -> bool:
        return any(
            task.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RETRYING}
            for task in tasks.values()
        )

    @staticmethod
    def _ready_tasks(tasks: dict[str, ResearchTask]) -> list[ResearchTask]:
        ready = []
        for task in tasks.values():
            if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                continue
            if all(tasks[dependency].status == TaskStatus.SUCCEEDED for dependency in task.dependencies):
                task.status = TaskStatus.READY
                ready.append(task)
        return sorted(ready, key=lambda task: (-task.priority, task.task_id))

    @staticmethod
    def _propagate_blocked(
        tasks: dict[str, ResearchTask], config: RunnableConfig | None, store: RuntimeStore | None
    ) -> None:
        terminal_failures = {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
        changed = True
        while changed:
            changed = False
            for task in tasks.values():
                if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                    continue
                failed_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if tasks[dependency].status in terminal_failures
                ]
                if failed_dependencies:
                    task.status = TaskStatus.BLOCKED
                    task.error = f"Blocked by failed dependencies: {', '.join(failed_dependencies)}"
                    if store:
                        store.save_task(task)
                        store.append_event(task.task_id, "task_blocked", {"dependencies": failed_dependencies})
                    record_event(
                        config,
                        "task_blocked",
                        "task_scheduler",
                        data={"task_id": task.task_id, "dependencies": failed_dependencies},
                    )
                    changed = True

    @staticmethod
    def _persist(
        store: RuntimeStore | None,
        task: ResearchTask,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if store:
            store.save_task(task)
            store.append_event(task.task_id, event_type, payload)

    @classmethod
    def _cancel_unfinished(
        cls,
        tasks: dict[str, ResearchTask],
        config: RunnableConfig | None,
        store: RuntimeStore | None,
        reason: str,
    ) -> None:
        for task in tasks.values():
            if task.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RETRYING}:
                task.status = TaskStatus.CANCELLED
                task.error = reason
                cls._persist(store, task, "task_cancelled", {"reason": reason})
                record_event(config, "task_cancelled", "task_scheduler", data={"task_id": task.task_id, "reason": reason})
