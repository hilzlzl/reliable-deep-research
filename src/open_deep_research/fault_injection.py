"""Deterministic fault injection for scheduler reliability experiments."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig

from open_deep_research.observability import record_event


async def inject_configured_fault(
    task_id: str,
    attempt: int,
    config: RunnableConfig | None,
) -> None:
    """Inject a configured failure before task execution.

    Rules are dictionaries with `task_id`, `attempt`, and `kind`. Supported
    kinds are `error`, `timeout`, and `delay`; delay also accepts `seconds`.
    """
    rules = (config or {}).get("configurable", {}).get("fault_injection_rules", [])
    for rule in rules:
        if rule.get("task_id") != task_id or int(rule.get("attempt", 1)) != attempt:
            continue
        kind = str(rule.get("kind", "error"))
        record_event(
            config,
            "fault_injected",
            "task_scheduler",
            data={"task_id": task_id, "attempt": attempt, "kind": kind},
        )
        if kind == "error":
            raise RuntimeError("Injected task failure")
        if kind == "timeout":
            raise asyncio.TimeoutError("Injected task timeout")
        if kind == "delay":
            await asyncio.sleep(float(rule.get("seconds", 0.01)))
            return
        raise ValueError(f"Unknown fault injection kind: {kind}")
