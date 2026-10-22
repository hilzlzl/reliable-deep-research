"""Tests for offline experiment statistics."""

from open_deep_research.evaluation import (
    aggregate_runs,
    bootstrap_confidence_interval,
    cohens_d,
)


def test_bootstrap_interval_is_deterministic_and_contains_mean():
    values = [1.0, 2.0, 3.0, 4.0]
    first = bootstrap_confidence_interval(values, samples=500, seed=7)
    second = bootstrap_confidence_interval(values, samples=500, seed=7)

    assert first == second
    assert first[0] <= 2.5 <= first[1]


def test_cohens_d_reports_direction_and_zero_difference():
    assert cohens_d([1, 2, 3], [3, 4, 5]) < 0
    assert cohens_d([1, 2, 3], [1, 2, 3]) == 0


def test_aggregate_runs_separates_report_success_from_full_acceptance():
    runs = [
        {"status": "succeeded", "full_acceptance": False, "elapsed_seconds": 10.0, "report_characters": 100, "task_count": 0, "successful_tasks": 0, "recovered_tasks": 0, "evidence_count": 0, "claim_count": 0},
        {"status": "succeeded", "full_acceptance": True, "elapsed_seconds": 20.0, "report_characters": 200, "task_count": 2, "successful_tasks": 2, "recovered_tasks": 1, "evidence_count": 3, "claim_count": 2},
    ]

    result = aggregate_runs(runs)

    assert result["run_success_rate"] == 1.0
    assert result["full_acceptance_rate"] == 0.5
    assert result["task_completion_rate"] == 1.0
    assert result["runs_with_recovery"] == 1
