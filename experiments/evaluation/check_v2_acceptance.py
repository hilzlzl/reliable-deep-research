"""Fail unless a formal Reliable V2 evaluation meets its hard requirements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_acceptance(
    evaluation: dict[str, Any], *, expected_runs: int, v2_group: str = "reliable_v2"
) -> dict[str, Any]:
    groups = evaluation.get("groups", {})
    aggregate = (groups.get(v2_group) or {}).get("aggregate")
    if aggregate is None:
        return {"passed": False, "checks": {"group_present": False}}

    expected = {
        "run_count": aggregate.get("runs") == expected_runs,
        "success_rate": aggregate.get("success_rate") == 1.0,
        "claim_evidence_link_rate": (
            aggregate.get("mean_claim_evidence_link_rate") == 1.0
        ),
        "citation_coverage": (
            aggregate.get("mean_report_evidence_id_coverage") == 1.0
        ),
        "source_precision": (
            aggregate.get("mean_report_source_precision") == 1.0
        ),
        "grounding_acceptance": (
            aggregate.get("v2_grounding_acceptance_rate") == 1.0
        ),
        "cost_recorded": aggregate.get("total_llm_calls") is not None
        and aggregate.get("total_completion_tokens") is not None,
    }
    return {"passed": all(expected.values()), "checks": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--expected-runs", type=int, default=4)
    parser.add_argument("--v2-group", default="reliable_v2")
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = check_acceptance(
        evaluation, expected_runs=args.expected_runs, v2_group=args.v2_group
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
