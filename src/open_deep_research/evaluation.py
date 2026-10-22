"""Offline metrics and statistics for baseline comparison experiments."""

from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
from pathlib import Path
from typing import Callable


def bootstrap_confidence_interval(
    values: list[float],
    *,
    statistic: Callable[[list[float]], float] = statistics.mean,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Return a deterministic percentile bootstrap interval."""
    if not values:
        return None
    generator = random.Random(seed)
    estimates = sorted(
        statistic([generator.choice(values) for _ in values]) for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = estimates[max(0, int(tail * samples))]
    upper = estimates[min(samples - 1, int((1.0 - tail) * samples) - 1)]
    return lower, upper


def cohens_d(left: list[float], right: list[float]) -> float | None:
    """Compute standardized mean difference using pooled sample variance."""
    if len(left) < 2 or len(right) < 2:
        return None
    pooled_variance = (
        (len(left) - 1) * statistics.variance(left)
        + (len(right) - 1) * statistics.variance(right)
    ) / (len(left) + len(right) - 2)
    if pooled_variance == 0:
        return 0.0 if statistics.mean(left) == statistics.mean(right) else None
    return (statistics.mean(left) - statistics.mean(right)) / math.sqrt(pooled_variance)


def _table_count(connection: sqlite3.Connection, table: str, where: str = "") -> int:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return 0
    return int(connection.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0])


def load_run_metrics(run_dir: Path) -> dict:
    """Load comparable metrics from one baseline artifact directory."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    task_count = successful_tasks = recovered_tasks = evidence_count = claim_count = 0
    db_path = run_dir / "runtime.db"
    if db_path.exists():
        with sqlite3.connect(db_path) as connection:
            task_count = _table_count(connection, "tasks")
            successful_tasks = _table_count(connection, "tasks", "WHERE status='SUCCEEDED'")
            recovered_tasks = _table_count(
                connection, "tasks", "WHERE status='SUCCEEDED' AND attempt_count > 1"
            )
            evidence_count = _table_count(connection, "evidence")
            claim_count = _table_count(connection, "claims")
    return {
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0)),
        "report_characters": int(summary.get("report_characters", 0)),
        "http_urls": int(summary.get("http_urls", 0)),
        "task_count": task_count,
        "successful_tasks": successful_tasks,
        "recovered_tasks": recovered_tasks,
        "evidence_count": evidence_count,
        "claim_count": claim_count,
        "full_acceptance": (
            summary.get("status") == "succeeded"
            and task_count > 0
            and successful_tasks == task_count
            and evidence_count > 0
        ),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Aggregate reliability and efficiency metrics for one experiment group."""
    if not runs:
        raise ValueError("At least one run is required")
    latencies = [run["elapsed_seconds"] for run in runs]
    report_sizes = [float(run["report_characters"]) for run in runs]
    total_tasks = sum(run["task_count"] for run in runs)
    successful_tasks = sum(run["successful_tasks"] for run in runs)
    attempted_recoveries = sum(
        1 for run in runs if run["recovered_tasks"] > 0
    )
    return {
        "runs": len(runs),
        "run_success_rate": sum(run["status"] == "succeeded" for run in runs) / len(runs),
        "full_acceptance_rate": sum(run["full_acceptance"] for run in runs) / len(runs),
        "task_completion_rate": successful_tasks / total_tasks if total_tasks else 0.0,
        "runs_with_recovery": attempted_recoveries,
        "mean_elapsed_seconds": statistics.mean(latencies),
        "elapsed_seconds_ci95": bootstrap_confidence_interval(latencies),
        "mean_report_characters": statistics.mean(report_sizes),
        "report_characters_ci95": bootstrap_confidence_interval(report_sizes),
        "total_evidence": sum(run["evidence_count"] for run in runs),
        "total_claims": sum(run["claim_count"] for run in runs),
    }
