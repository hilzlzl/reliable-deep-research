"""Tests for the reproducible fault comparison experiment."""

import asyncio

from experiments.reliability.fault_experiment import (
    run_baseline_trial,
    run_fault_matrix,
    run_reliable_trial,
)


def test_fault_experiment_distinguishes_recovery(tmp_path):
    baseline = asyncio.run(run_baseline_trial())
    reliable, no_duplicate_work = asyncio.run(run_reliable_trial(0, tmp_path))

    assert baseline is False
    assert reliable is True
    assert no_duplicate_work is True


def test_fault_matrix_distinguishes_transient_and_persistent_failures(tmp_path):
    matrix = asyncio.run(run_fault_matrix(2, tmp_path))

    assert matrix["transient_error"]["succeeded"] == 2
    assert matrix["transient_timeout"]["succeeded"] == 2
    assert matrix["persistent_error"]["succeeded"] == 0
    assert matrix["persistent_timeout"]["succeeded"] == 0
    assert all(row["expected_outcome_matches"] == 2 for row in matrix.values())
