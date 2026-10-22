"""Audit paired reports against fixed-corpus sources and reference answer points."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urldefrag, urlsplit

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from open_deep_research.local_benchmark import CorpusChunk, LocalBenchmarkCase, read_jsonl
from open_deep_research.utils import ainvoke_structured_with_fallback


_SOURCE_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)\.)\s+.*?((?:https?|local)://[^\s)]+)", re.MULTILINE)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_LINK_RE = re.compile(r"\[[^\]]+\]\(((?:https?|local)://[^)]+)\)")
_TERM_RE = re.compile(r"[a-z][a-z0-9-]{2,}", re.IGNORECASE)


class PointJudgment(BaseModel):
    point_id: int
    covered: bool
    reason: str = ""


class CitationJudgment(BaseModel):
    claim_id: str
    verdict: Literal["SUPPORTED", "PARTIAL", "CONTRADICTED", "NOT_SUPPORTED", "INSUFFICIENT_CONTEXT"]
    reason: str = ""


class ReportJudgment(BaseModel):
    points: list[PointJudgment] = Field(default_factory=list)
    citations: list[CitationJudgment] = Field(default_factory=list)


def source_index(report: str) -> dict[str, str]:
    """Resolve numbered references, including the upstream numbered-list form."""
    sources = {}
    for match in _SOURCE_RE.finditer(report):
        sources[match.group(1) or match.group(2)] = match.group(3).rstrip(".,;:")
    return sources


def cited_claim_candidates(report: str, case_id: str, limit: int) -> tuple[list[dict], int]:
    """Choose a stable sample of cited body statements without judging them."""
    body = re.split(r"(?im)^#{1,3}\s+(?:sources|references|来源|参考文献)\s*$", report, maxsplit=1)[0]
    sources = source_index(report)
    candidates = []
    seen = set()
    for line in body.splitlines():
        stripped = line.strip(" -*#\t")
        if not stripped or stripped.startswith("|") or len(stripped) < 35:
            continue
        for sentence in re.split(r"(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff*])", stripped):
            ids = list(dict.fromkeys(_CITATION_RE.findall(sentence)))
            inline_urls = list(dict.fromkeys(_LINK_RE.findall(sentence)))
            if (not ids and not inline_urls) or len(sentence) < 35:
                continue
            statement = sentence[:650].strip()
            source_urls = list(dict.fromkeys(
                [sources[number] for number in ids if number in sources] + inline_urls
            ))
            key = (statement, tuple(ids), tuple(source_urls))
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "claim_id": "claim-" + hashlib.sha256(statement.encode()).hexdigest()[:12],
                "statement": statement,
                "citation_numbers": ids,
                "source_urls": source_urls,
                "unresolved_citation_numbers": [number for number in ids if number not in sources],
            })
    candidates.sort(key=lambda item: hashlib.sha256(f"{case_id}:{item['statement']}".encode()).hexdigest())
    return candidates[:limit], len(candidates)


def corpus_index(chunks: list[CorpusChunk]) -> tuple[dict[str, CorpusChunk], dict[str, list[CorpusChunk]]]:
    by_id = {item.chunk_id: item for item in chunks}
    by_url: dict[str, list[CorpusChunk]] = defaultdict(list)
    for chunk in chunks:
        by_url[urldefrag(chunk.source_url).url].append(chunk)
    return by_id, by_url


def source_snippets(
    claim: dict,
    by_id: dict[str, CorpusChunk],
    by_url: dict[str, list[CorpusChunk]],
) -> list[dict]:
    """Use the cited exact chunk when known, otherwise find relevant cited-paper chunks."""
    snippets = []
    for url in claim["source_urls"][:3]:
        parsed = urlsplit(url)
        chunk_id = parse_qs(parsed.fragment).get("chunk", [None])[0]
        base_url = urldefrag(url).url
        if chunk_id and chunk_id in by_id and by_id[chunk_id].source_url == base_url:
            selected = [by_id[chunk_id]]
        else:
            terms = set(_TERM_RE.findall(claim["statement"].lower()))
            selected = sorted(
                by_url.get(base_url, []),
                key=lambda item: (-len(terms & set(_TERM_RE.findall(item.text.lower()))), item.chunk_id),
            )[:2]
        snippets.append({
            "url": url,
            "in_fixed_corpus": bool(selected),
            "chunks": [{"chunk_id": item.chunk_id, "text": item.text[:1400]} for item in selected],
        })
    return snippets


def latest_runs(root: Path) -> dict[str, Path]:
    runs = {}
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") in {"succeeded", "degraded"}:
            runs[summary["case_id"]] = path.parent
    return runs


def attempted_run_statuses(root: Path) -> dict[str, str]:
    statuses = {}
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        statuses[summary["case_id"]] = str(summary.get("status", "unknown"))
    return statuses


def sanitized_judgment(raw: ReportJudgment, points: list[str], claims: list[dict]) -> dict:
    point_by_id = {item.point_id: item for item in raw.points}
    claim_by_id = {item.claim_id: item for item in raw.citations}
    return {
        "points": [
            {
                "point_id": index,
                "reference_point": point,
                "covered": bool(point_by_id.get(index) and point_by_id[index].covered),
                "reason": point_by_id[index].reason if index in point_by_id else "Judge omitted this point.",
            }
            for index, point in enumerate(points, 1)
        ],
        "citations": [
            {
                **claim,
                "verdict": (
                    claim_by_id[claim["claim_id"]].verdict
                    if claim["claim_id"] in claim_by_id else "INSUFFICIENT_CONTEXT"
                ),
                "reason": (
                    claim_by_id[claim["claim_id"]].reason
                    if claim["claim_id"] in claim_by_id else "Judge omitted this claim."
                ),
            }
            for claim in claims
        ],
    }


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for profile in ("baseline", "reliable_v2"):
        group = [item for item in rows if item["profile"] == profile]
        points = [point for item in group for point in item["judgment"]["points"]]
        citations = [claim for item in group for claim in item["judgment"]["citations"]]
        source_links = [source for claim in citations for source in claim["source_snippets"]]
        output[profile] = {
            "reports": len(group),
            "reference_points": len(points),
            "covered_reference_points": sum(item["covered"] for item in points),
            "reference_point_coverage": sum(item["covered"] for item in points) / len(points) if points else None,
            "cited_claim_candidates": sum(item["cited_claim_candidates"] for item in group),
            "cited_claims_sampled": len(citations),
            "cited_claims_supported": sum(item["verdict"] == "SUPPORTED" for item in citations),
            "sampled_claim_support_rate": sum(item["verdict"] == "SUPPORTED" for item in citations) / len(citations) if citations else None,
            "sampled_claim_partial_rate": sum(item["verdict"] == "PARTIAL" for item in citations) / len(citations) if citations else None,
            "sampled_citation_links_in_corpus": sum(item["in_fixed_corpus"] for item in source_links),
            "sampled_citation_links": len(source_links),
            "sampled_fixed_corpus_source_rate": sum(item["in_fixed_corpus"] for item in source_links) / len(source_links) if source_links else None,
            "mean_generation_llm_calls": statistics.mean(item["generation_llm_calls"] for item in group) if group else None,
            "mean_generation_completion_tokens": statistics.mean(item["generation_completion_tokens"] for item in group) if group else None,
            "mean_generation_seconds": statistics.mean(item["generation_seconds"] for item in group) if group else None,
        }
    return output


async def run_audit(args: argparse.Namespace) -> dict:
    load_dotenv()
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"{args.api_key_env} is required")
    cases = {item.case_id: item for item in read_jsonl(args.questions, LocalBenchmarkCase)}
    chunks = read_jsonl(args.corpus / "chunks.jsonl", CorpusChunk)
    by_id, by_url = corpus_index(chunks)
    baseline = latest_runs(args.baseline)
    candidate = latest_runs(args.candidate)
    paired = sorted(set(cases) & baseline.keys() & candidate.keys())
    baseline_statuses = attempted_run_statuses(args.baseline)
    candidate_statuses = attempted_run_statuses(args.candidate)
    attempted = sorted(set(baseline_statuses) | set(candidate_statuses))
    if not paired:
        raise ValueError("No completed paired reports found")
    model = ChatOpenAI(
        model=args.judge_model,
        api_key=key,
        base_url=args.base_url,
        max_tokens=2500,
        timeout=90,
        max_retries=0,
    )
    rows = []
    if args.output.exists():
        rows = json.loads(args.output.read_text(encoding="utf-8")).get("runs", [])
    completed = {(item["case_id"], item["profile"]) for item in rows}
    for case_id in paired:
        for profile, run_dir in (("baseline", baseline[case_id]), ("reliable_v2", candidate[case_id])):
            if (case_id, profile) in completed:
                continue
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            claims, candidate_count = cited_claim_candidates(report, case_id, args.max_claims)
            for claim in claims:
                claim["source_snippets"] = source_snippets(claim, by_id, by_url)
            points = cases[case_id].reference_answer_points
            report_view = report[:args.max_report_characters]
            prompt = (
                "Audit this report using only the supplied report text and fixed-corpus excerpts. "
                "For each numbered reference answer point, mark covered only if the report clearly states it. "
                "For each cited statement, judge whether its cited excerpts entail the complete statement: "
                "SUPPORTED, PARTIAL, CONTRADICTED, NOT_SUPPORTED, or INSUFFICIENT_CONTEXT. "
                "A missing source, missing citation number, or irrelevant excerpt cannot support a claim. "
                "Do not use your own knowledge. Return every point_id and claim_id. Give a short reason.\n\n"
                f"Question: {cases[case_id].question}\n\n"
                f"Reference points: {json.dumps([{'point_id': index, 'text': point} for index, point in enumerate(points, 1)], ensure_ascii=False)}\n\n"
                f"Report: {report_view}\n\n"
                f"Cited statements and source excerpts: {json.dumps(claims, ensure_ascii=False)}"
            )
            raw = await ainvoke_structured_with_fallback(
                model,
                ReportJudgment,
                [HumanMessage(content=prompt)],
                retries=2,
                native_structured_output=False,
            )
            rows.append({
                "case_id": case_id,
                "profile": profile,
                "run_dir": str(run_dir),
                "cited_claim_candidates": candidate_count,
                "generation_llm_calls": (summary.get("budget") or {}).get("llm_calls", 0),
                "generation_completion_tokens": (summary.get("budget") or {}).get("observed_completion_tokens", 0),
                "generation_seconds": summary.get("elapsed_seconds", 0),
                "report_characters": len(report),
                "report_truncated_for_judge": len(report) > args.max_report_characters,
                "judgment": sanitized_judgment(raw, points, claims),
            })
            payload = {
                "method": {
                    "judge": args.judge_model,
                    "judge_type": "model_assisted_not_human",
                    "max_claims_per_report": args.max_claims,
                    "max_report_characters": args.max_report_characters,
                    "sample": "Stable hash order among body lines with numbered citations or Markdown source links",
                    "source_check": "Cited URL resolved to fixed corpus; exact chunk if named, otherwise two lexically closest chunks from the cited document",
                    "limits": "One generation per case/profile. Judge shares a model family with the writer; excerpt retrieval may miss support elsewhere in a long paper.",
                },
                "paired_case_ids": paired,
                "attempted_case_ids": attempted,
                "unpaired_case_statuses": {
                    case_id: {
                        "baseline": baseline_statuses.get(case_id, "not_run"),
                        "reliable_v2": candidate_statuses.get(case_id, "not_run"),
                    }
                    for case_id in attempted if case_id not in paired
                },
                "aggregate": aggregate(rows),
                "runs": rows,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{case_id} {profile}: {len(claims)} cited statements audited", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("benchmarks/researchbench_en/questions.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("benchmark_data/researchbench_en/derived"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-claims", type=int, default=5)
    parser.add_argument("--max-report-characters", type=int, default=45000)
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    args = parser.parse_args()
    result = asyncio.run(run_audit(args))
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
