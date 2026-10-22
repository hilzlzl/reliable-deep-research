"""Blind pairwise LLM-as-Judge comparison for matched report runs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from open_deep_research.utils import ainvoke_structured_with_fallback


class ReportScores(BaseModel):
    relevance: float = Field(ge=1, le=5)
    completeness: float = Field(ge=1, le=5)
    structure: float = Field(ge=1, le=5)
    source_use: float = Field(ge=1, le=5)
    calibrated_claims: float = Field(ge=1, le=5)


class PairwiseJudgment(BaseModel):
    report_a: ReportScores
    report_b: ReportScores
    winner: Literal["A", "B", "tie"]
    rationale: str


def discover_by_case(paths: list[Path]) -> dict[str, Path]:
    result = {}
    for path in paths:
        for summary_path in path.rglob("summary.json"):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") in {"succeeded", "degraded"}:
                result[summary["case_id"]] = summary_path.parent
    return result


def truncate_report(report: str, max_characters: int) -> str:
    """Preserve both report arguments and trailing source lists."""
    if len(report) <= max_characters:
        return report
    marker = "\n\n[... middle omitted for blind evaluation ...]\n\n"
    available = max_characters - len(marker)
    head_size = int(available * 0.7)
    return report[:head_size] + marker + report[-(available - head_size) :]


def build_output(judgments: list[dict], model: str, max_report_characters: int) -> dict:
    dimensions = list(ReportScores.model_fields)
    aggregate = {
        "cases": len(judgments),
        "candidate_wins": sum(item["winner_profile"] == "candidate" for item in judgments),
        "baseline_wins": sum(item["winner_profile"] == "baseline" for item in judgments),
        "ties": sum(item["winner_profile"] == "tie" for item in judgments),
        "mean_baseline_scores": {},
        "mean_candidate_scores": {},
        "method": {
            "judge_model": model,
            "blind_order": True,
            "max_report_characters": max_report_characters,
            "truncation": "70% head + 30% tail to preserve source lists",
            "limitations": "Single LLM judge pass; not a substitute for factual human review.",
        },
    }
    if judgments:
        aggregate["mean_baseline_scores"] = {
            dimension: sum(item["baseline_scores"][dimension] for item in judgments) / len(judgments)
            for dimension in dimensions
        }
        aggregate["mean_candidate_scores"] = {
            dimension: sum(item["candidate_scores"][dimension] for item in judgments) / len(judgments)
            for dimension in dimensions
        }
    return {"aggregate": aggregate, "judgments": judgments}


def save_output(path: Path, judgments: list[dict], model: str, max_report_characters: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_output(judgments, model, max_report_characters),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-report-characters", type=int, default=12000)
    parser.add_argument("--request-timeout-seconds", type=float, default=180)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is required")
    baseline = discover_by_case(args.baseline)
    candidate = discover_by_case(args.candidate)
    case_ids = sorted(baseline.keys() & candidate.keys())
    model = ChatOpenAI(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        max_tokens=1024,
        timeout=args.request_timeout_seconds,
        max_retries=0,
    )
    judgments = []
    if args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        judgments = saved.get("judgments", [])
    completed_case_ids = {item["case_id"] for item in judgments}
    for case_id in case_ids:
        if case_id in completed_case_ids:
            print(f"{case_id}: already judged", flush=True)
            continue
        input_data = json.loads((baseline[case_id] / "input.json").read_text(encoding="utf-8"))
        baseline_report = truncate_report(
            (baseline[case_id] / "report.md").read_text(encoding="utf-8"),
            args.max_report_characters,
        )
        candidate_report = truncate_report(
            (candidate[case_id] / "report.md").read_text(encoding="utf-8"),
            args.max_report_characters,
        )
        candidate_is_a = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0
        report_a, report_b = (
            (candidate_report, baseline_report)
            if candidate_is_a
            else (baseline_report, candidate_report)
        )
        prompt = f"""You are a blind evaluator comparing two research reports for the same question.
Score each report from 1 to 5 on relevance, completeness, structure, source use,
and calibrated claims (distinguishing evidence from uncertainty). Judge only the
text shown. Do not assume that a longer report is better. Source-use scores must
reflect whether sources are connected to claims, not merely URL count.

QUESTION:
{input_data['prompt']}

REPORT A:
{report_a}

REPORT B:
{report_b}
"""
        judgment = await ainvoke_structured_with_fallback(
            model,
            PairwiseJudgment,
            [HumanMessage(content=prompt)],
            retries=1,
            native_structured_output=False,
        )
        winner_profile = (
            "tie" if judgment.winner == "tie" else
            ("candidate" if (judgment.winner == "A") == candidate_is_a else "baseline")
        )
        judgments.append({
            "case_id": case_id,
            "candidate_label": "A" if candidate_is_a else "B",
            "winner_label": judgment.winner,
            "winner_profile": winner_profile,
            "baseline_scores": (
                judgment.report_b.model_dump() if candidate_is_a else judgment.report_a.model_dump()
            ),
            "candidate_scores": (
                judgment.report_a.model_dump() if candidate_is_a else judgment.report_b.model_dump()
            ),
            "rationale": judgment.rationale,
        })
        save_output(args.output, judgments, args.model, args.max_report_characters)
        print(f"{case_id}: {winner_profile}", flush=True)

    output = build_output(judgments, args.model, args.max_report_characters)
    save_output(args.output, judgments, args.model, args.max_report_characters)
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
