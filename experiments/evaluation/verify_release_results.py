"""Recalculate published mainline metrics from every attempt, without API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def verify_replacement_sources(data: dict, results: Path) -> None:
    """Require all 13 failed slots to use their reruns, including the new failure."""
    exports = {}
    for name, spec in data["source_exports"].items():
        path = results.parent / spec["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError(f"Source export hash mismatch: {name}")
        exports[name] = json.loads(raw)
    originals = exports["formal"]["runs"]
    old = {(r["repeat"], r["case_id"]): r for r in originals if r["profile"] == "coverage_v1"}
    reruns = {(r["repeat"], r["case_id"]): r for r in exports["retry"]["runs"]}
    failed = {slot for slot, run in old.items() if run["status"] != "succeeded"}
    if len(old) != 60 or len(failed) != 13 or set(reruns) != failed or len(exports["retry"]["runs"]) != 13:
        raise ValueError("Replacement selection must be exactly the original 13 unsuccessful slots")
    controls = {r["id"]: r for r in originals if r["profile"] in {"baseline", "reliable_v2"}}
    source_counts = Counter()
    for run in data["runs"]:
        if run["profile"] != "coverage_replacement_view":
            if run != controls.get(run["id"]):
                raise ValueError("Historical control was changed")
            continue
        slot = (run["repeat"], run["case_id"])
        source = reruns[slot] if slot in failed else old[slot]
        source_counts[source["profile"]] += 1
        if run.get("source_run_id") != source["id"] or run.get("source_profile") != source["profile"]:
            raise ValueError("Replacement source mapping mismatch")
        expected_experiment = "coverage_retry_20261021" if slot in failed else "coverage_formal_20261021"
        if run.get("source_experiment") != expected_experiment:
            raise ValueError("Replacement experiment mismatch")
        for key, value in source.items():
            if key not in {"id", "profile"} and run.get(key) != value:
                raise ValueError(f"Replacement changed its source record: {key}")
    if source_counts != {"coverage_v1": 47, "coverage_v2": 13}:
        raise ValueError("Expected 47 original accepted slots and all 13 reruns")


def verify(results: Path, questions: Path) -> dict:
    """Validate case coverage, denominators, statuses and published aggregates."""
    data = json.loads(results.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {row["case_id"]: row for row in rows}
    if len(rows) != 30 or len(cases) != 30:
        raise ValueError("Expected 30 distinct questions")
    profiles = {"baseline", "reliable_v2", "coverage_replacement_view"}
    expected = {(profile, repeat, case) for profile in profiles for repeat in (1, 2) for case in cases}
    runs = data["runs"]
    actual = [(r["profile"], r["repeat"], r["case_id"]) for r in runs]
    if len(actual) != 180 or len(set(actual)) != 180 or set(actual) != expected:
        raise ValueError("Expected exactly 180 distinct comparison slots")
    if len({r["id"] for r in runs}) != 180:
        raise ValueError("Duplicate run IDs")
    verify_replacement_sources(data, results)
    computed = {}
    for profile in sorted(profiles):
        group = [r for r in runs if r["profile"] == profile]
        accepted = [r for r in group if r["status"] == "succeeded"]
        for run in group:
            judgment = run.get("judgment")
            if run["status"] != "succeeded":
                if judgment is not None:
                    raise ValueError("Unaccepted attempt contributes a judgment")
                continue
            if judgment is None:
                raise ValueError("Accepted report lacks judgment")
            points = judgment["points"]
            count = len(cases[run["case_id"]]["reference_answer_points"])
            if sorted(p["point_id"] for p in points) != list(range(1, count + 1)):
                raise ValueError("Missing or duplicate reference point IDs")
            if any(type(p["covered"]) is not bool for p in points):
                raise ValueError("Reference-point flags must be booleans")
            citations = judgment["citations"]
            if len(citations) > 5 or len({c["claim_id"] for c in citations}) != len(citations):
                raise ValueError("Invalid citation sample")
        citations = [c for r in accepted for c in r["judgment"]["citations"]]
        metrics = {
            "attempts": len(group), "statuses": dict(Counter(r["status"] for r in group)),
            "judged_reports": len(accepted),
            "covered_points": sum(p["covered"] for r in accepted for p in r["judgment"]["points"]),
            "all_attempt_reference_points": sum(len(cases[r["case_id"]]["reference_answer_points"]) for r in group),
            "successful_report_points_total": sum(len(cases[r["case_id"]]["reference_answer_points"]) for r in accepted),
            "sampled_citations": len(citations),
            "supported_citations": sum(c["verdict"] == "SUPPORTED" for c in citations),
            "judged_reports_without_extractable_citations": sum(not r["judgment"]["citations"] for r in accepted),
            "successful_reports_not_judged": 0,
            "mean_seconds_all_attempts": sum(r["elapsed_seconds"] for r in group) / len(group),
            "mean_llm_calls_all_attempts": sum(r["llm_calls"] for r in group) / len(group),
            "mean_search_calls_all_attempts": sum(r["search_calls"] for r in group) / len(group),
            "total_completion_tokens": sum(r["completion_tokens"] for r in group),
        }
        metrics["observed_coverage_all_attempts"] = metrics["covered_points"] / metrics["all_attempt_reference_points"]
        metrics["successful_report_coverage"] = metrics["covered_points"] / metrics["successful_report_points_total"]
        metrics["citation_support_rate"] = metrics["supported_citations"] / metrics["sampled_citations"]
        for key, value in metrics.items():
            saved = data["aggregate"][profile][key]
            matches = math.isclose(value, saved, rel_tol=1e-9, abs_tol=1e-8) if isinstance(value, float) else value == saved
            if not matches:
                raise ValueError(f"{profile}.{key}: published aggregate does not match attempts")
        computed[profile] = metrics
    return computed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results/release_comparison.json"))
    parser.add_argument("--questions", type=Path, default=Path("benchmarks/expanded_20261021/questions.jsonl"))
    args = parser.parse_args()
    print(json.dumps({"verified": True, "profiles": verify(args.results, args.questions)}, indent=2))


if __name__ == "__main__":
    main()
