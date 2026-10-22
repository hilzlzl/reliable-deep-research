"""Shared, concurrency-safe budgets for one research run."""

from __future__ import annotations

import asyncio
import time
from threading import Lock
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from open_deep_research.observability import record_event


class BudgetExceeded(RuntimeError):
    """Raised before work starts when a configured run budget is exhausted."""

    def __init__(self, resource: str, limit: int | float):
        self.resource = resource
        self.limit = limit
        super().__init__(f"Run budget exhausted: {resource} limit is {limit}")


class BudgetSnapshot(BaseModel):
    llm_calls: int
    search_calls: int
    task_attempts: int
    estimated_prompt_tokens: int
    observed_completion_tokens: int
    elapsed_seconds: float
    exhausted_resource: str | None
    phase: str = "all"
    research_limit_fraction: float = 1.0


class RunBudget:
    """Atomic counters and a monotonic deadline shared by concurrent agents."""

    def __init__(
        self,
        *,
        max_llm_calls: int | None = None,
        max_search_calls: int | None = None,
        max_task_attempts: int | None = None,
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        max_total_seconds: float | None = None,
        research_limit_fraction: float = 1.0,
    ) -> None:
        self.limits = {
            "llm_calls": max_llm_calls,
            "search_calls": max_search_calls,
            "task_attempts": max_task_attempts,
            "estimated_prompt_tokens": max_prompt_tokens,
            "observed_completion_tokens": max_completion_tokens,
        }
        self.max_total_seconds = max_total_seconds
        self.phase = "all"
        self.reserve_search_calls = True
        self.research_limit_fraction = max(0.0, min(1.0, research_limit_fraction))
        self.started_at = time.monotonic()
        self._counts = {key: 0 for key in self.limits}
        self._exhausted_resource: str | None = None
        self._lock = Lock()

    def set_phase(self, phase: str) -> None:
        """Switch budget enforcement between research and finalization phases."""
        if phase not in {"all", "research", "final"}:
            raise ValueError(f"Invalid budget phase: {phase}")
        with self._lock:
            self.phase = phase

    def set_research_limit_fraction(
        self, fraction: float, *, reserve_search_calls: bool = True
    ) -> None:
        with self._lock:
            self.research_limit_fraction = max(0.0, min(1.0, fraction))
            self.reserve_search_calls = reserve_search_calls

    def _effective_limit(self, resource: str) -> int | float | None:
        limit = self.limits[resource]
        if (limit is None or self.phase != "research"
                or resource == "search_calls" and not self.reserve_search_calls):
            return limit
        return max(1, int(limit * self.research_limit_fraction))

    def consume(self, resource: str, amount: int = 1) -> None:
        if amount < 0 or resource not in self._counts:
            raise ValueError(f"Invalid budget consumption: {resource}={amount}")
        self.check_time()
        with self._lock:
            limit = self._effective_limit(resource)
            if limit is not None and self._counts[resource] + amount > limit:
                self._exhausted_resource = resource
                raise BudgetExceeded(resource, limit)
            self._counts[resource] += amount

    def reserve_llm_call(self, estimated_prompt_tokens: int) -> None:
        """Atomically reserve both counters before starting a model request."""
        self.check_time()
        requested = {
            "llm_calls": 1,
            "estimated_prompt_tokens": estimated_prompt_tokens,
        }
        with self._lock:
            for resource, amount in requested.items():
                limit = self._effective_limit(resource)
                if limit is not None and self._counts[resource] + amount > limit:
                    self._exhausted_resource = resource
                    raise BudgetExceeded(resource, limit)
            for resource, amount in requested.items():
                self._counts[resource] += amount

    def check_time(self) -> None:
        if self.max_total_seconds is None:
            return
        deadline = self.max_total_seconds
        if self.phase == "research":
            deadline *= self.research_limit_fraction
        if self.elapsed_seconds >= deadline:
            with self._lock:
                self._exhausted_resource = "total_seconds"
            raise BudgetExceeded("total_seconds", deadline)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float | None:
        if self.max_total_seconds is None:
            return None
        deadline = self.max_total_seconds
        if self.phase == "research":
            deadline *= self.research_limit_fraction
        return max(0.0, deadline - self.elapsed_seconds)

    def observe_completion(self, response: Any) -> None:
        metadata = getattr(response, "usage_metadata", None) or {}
        tokens = int(metadata.get("output_tokens", 0) or 0)
        if not tokens:
            response_metadata = getattr(response, "response_metadata", None) or {}
            usage = response_metadata.get("token_usage", {})
            tokens = int(usage.get("completion_tokens", 0) or 0)
        if tokens:
            self.consume("observed_completion_tokens", tokens)

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            values = dict(self._counts)
            exhausted = self._exhausted_resource
        return BudgetSnapshot(
            **values,
            elapsed_seconds=round(self.elapsed_seconds, 3),
            exhausted_resource=exhausted,
            phase=self.phase,
            research_limit_fraction=self.research_limit_fraction,
        )


def budget_from_config(config: RunnableConfig | None) -> RunBudget | None:
    return (config or {}).get("configurable", {}).get("run_budget")


async def budgeted_ainvoke(
    model: Any,
    messages: list[Any],
    config: RunnableConfig | None,
    *,
    node: str,
) -> Any:
    """Invoke a model within call, approximate token, and wall-clock budgets."""
    budget = budget_from_config(config)
    if budget is None:
        return await model.ainvoke(messages)

    prompt_characters = sum(len(str(getattr(message, "content", message))) for message in messages)
    budget.reserve_llm_call(max(1, prompt_characters // 4))
    remaining = budget.remaining_seconds
    if remaining is not None and remaining <= 0:
        budget.check_time()
    try:
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=remaining)
    except asyncio.TimeoutError as error:
        if budget.remaining_seconds is None or budget.remaining_seconds > 0.001:
            raise
        budget.check_time()
        raise BudgetExceeded("total_seconds", budget.max_total_seconds or 0) from error
    budget.observe_completion(response)
    record_event(config, "budget_updated", node, data=budget.snapshot().model_dump())
    return response
