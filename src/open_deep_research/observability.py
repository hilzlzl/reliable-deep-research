"""Lightweight, file-backed observability for research runs.

The observer is deliberately independent from LangSmith.  It gives every run a
small, replayable local trace that is useful when developing against a local
model, injecting failures, or comparing two runtime versions.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Callable, Mapping, TypeVar

from langchain_core.runnables import RunnableConfig

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
_WRITE_LOCK = Lock()


def _config_value(config: RunnableConfig | None, key: str, default: Any = None) -> Any:
    configurable = (config or {}).get("configurable", {})
    return configurable.get(key, default)


def _short(value: Any, limit: int = 500) -> Any:
    """Return JSON-friendly metadata without dumping prompts or tool results."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else value[:limit]
    if isinstance(value, Mapping):
        return {str(k): _short(v, limit) for k, v in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        return [_short(item, limit) for item in value[:30]]
    return str(value)[:limit]


def _event_path(config: RunnableConfig | None) -> Path | None:
    enabled = _config_value(config, "observability_enabled", True)
    if not enabled:
        return None
    path = _config_value(config, "event_log_path", os.environ.get("OPEN_DEEP_RESEARCH_EVENT_LOG"))
    if not path:
        path = "artifacts/research_events.jsonl"
    return Path(path)


def _run_id(config: RunnableConfig | None) -> str:
    configurable = (config or {}).get("configurable", {})
    metadata = (config or {}).get("metadata", {})
    return str(
        configurable.get("run_id")
        or configurable.get("thread_id")
        or metadata.get("run_id")
        or "unscoped"
    )


def record_event(
    config: RunnableConfig | None,
    event_type: str,
    node: str,
    *,
    data: Mapping[str, Any] | None = None,
) -> None:
    """Append one structured event; observability must never break research."""
    path = _event_path(config)
    if path is None:
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": _run_id(config),
        "event_type": event_type,
        "node": node,
        "data": _short(dict(data or {})),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # A broken log destination must not take down an otherwise valid run.
        return


def observed_node(name: str, function: F) -> F:
    """Wrap an async LangGraph node with start/success/failure events."""

    @wraps(function)
    async def wrapper(state: Any, config: RunnableConfig | None = None, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        record_event(config, "node_started", name, data={"state_keys": list(state.keys()) if isinstance(state, Mapping) else []})
        try:
            result = await function(state, config, *args, **kwargs)
        except Exception as error:
            record_event(
                config,
                "node_failed",
                name,
                data={
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(limit=3),
                },
            )
            raise
        result_keys = list(result.keys()) if isinstance(result, Mapping) else []
        record_event(
            config,
            "node_succeeded",
            name,
            data={
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "result_keys": result_keys,
            },
        )
        return result

    return wrapper  # type: ignore[return-value]
