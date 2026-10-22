"""Verify released evidence retains failed slots and detects metric drift."""

import json
from pathlib import Path

import pytest

from experiments.evaluation.verify_release_results import verify

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results/release_comparison.json"
QUESTIONS = ROOT / "benchmarks/expanded_20261021/questions.jsonl"


def test_published_results_reaggregate_all_attempts():
    metrics = verify(RESULTS, QUESTIONS)
    assert metrics["baseline"]["statuses"] == {"succeeded": 52, "budget_exhausted": 8}
    assert metrics["reliable_v2"]["statuses"] == {"succeeded": 51, "budget_exhausted": 9}
    assert metrics["coverage_replacement_view"]["statuses"] == {"succeeded": 59, "failed": 1}
    assert metrics["coverage_replacement_view"]["covered_points"] == 149
    assert metrics["coverage_replacement_view"]["supported_citations"] == 219


@pytest.mark.parametrize("change", ["remove_failure", "duplicate_slot", "change_score", "wrong_source", "hide_rerun_failure"])
def test_evidence_verification_rejects_incomplete_or_changed_records(tmp_path, change):
    data = json.loads(RESULTS.read_text())
    if change == "remove_failure":
        data["runs"] = [r for r in data["runs"] if r["status"] == "succeeded"]
    elif change == "duplicate_slot":
        data["runs"][1] = data["runs"][0]
    elif change == "change_score":
        data["aggregate"]["reliable_v2"]["supported_citations"] += 1
    elif change == "wrong_source":
        row = next(r for r in data["runs"] if r.get("source_profile") == "coverage_v2")
        row["source_profile"] = "coverage_v1"
    else:
        row = next(r for r in data["runs"] if r["profile"] == "coverage_replacement_view" and r["status"] == "failed")
        row["status"] = "succeeded"
    for spec in data["source_exports"].values():
        spec["path"] = str(RESULTS.parent / spec["path"])
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify(path, QUESTIONS)
