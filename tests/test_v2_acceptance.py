"""Tests for the formal Reliable V2 acceptance gate."""

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "evaluation" / "check_v2_acceptance.py"
SPEC = importlib.util.spec_from_file_location("check_v2_acceptance", MODULE_PATH)
acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(acceptance)


def evaluation(source_precision=1.0):
    return {
        "groups": {
            "reliable_v2": {
                "aggregate": {
                    "runs": 4,
                    "success_rate": 1.0,
                    "mean_claim_evidence_link_rate": 1.0,
                    "mean_report_evidence_id_coverage": 1.0,
                    "mean_report_source_precision": source_precision,
                    "v2_grounding_acceptance_rate": 1.0,
                    "total_llm_calls": 40,
                    "total_completion_tokens": 1000,
                }
            }
        }
    }


def test_acceptance_passes_only_when_all_hard_metrics_pass():
    result = acceptance.check_acceptance(evaluation(), expected_runs=4)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_acceptance_rejects_partial_source_precision():
    result = acceptance.check_acceptance(
        evaluation(source_precision=0.75), expected_runs=4
    )

    assert result["passed"] is False
    assert result["checks"]["source_precision"] is False
