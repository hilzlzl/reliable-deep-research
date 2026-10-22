"""Run a small, reproducible local baseline against fixed research questions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from open_deep_research.deep_researcher import deep_researcher_builder
from open_deep_research.observability import record_event
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.run_budget import BudgetExceeded, RunBudget


def load_cases(path: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
    raw_cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = []
    for raw_case in raw_cases:
        case = dict(raw_case)
        case.setdefault("id", case.get("case_id"))
        case.setdefault("prompt", case.get("question"))
        case.setdefault("category", case.get("domain", "unknown"))
        if not case.get("id") or not case.get("prompt"):
            raise ValueError(f"Case requires id/prompt or case_id/question: {raw_case}")
        cases.append(case)
    if selected_ids:
        cases = [case for case in cases if case["id"] in selected_ids]
        missing = selected_ids - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown case ids: {', '.join(sorted(missing))}")
    return cases


def report_metrics(report: str) -> dict[str, int]:
    return {
        "report_characters": len(report),
        "markdown_links": report.count("]("),
        "http_urls": report.count("http://") + report.count("https://"),
    }


def review_metrics(event_path: Path) -> dict[str, Any] | None:
    if not event_path.is_file():
        return None
    for line in reversed(event_path.read_text(encoding="utf-8").splitlines()):
        event = json.loads(line)
        if event.get("event_type") == "report_review_finished":
            data = event.get("data", {})
            return {
                "status": "succeeded",
                "final_score": data.get("final_score"),
                "stop_reason": data.get("stop_reason"),
                "rounds": data.get("rounds", []),
            }
        if event.get("event_type") == "report_review_failed":
            data = event.get("data", {})
            return {
                "status": "failed",
                "error_type": data.get("error_type"),
                "fallback": data.get("fallback"),
            }
    return None


def report_grounding_metrics(event_path: Path) -> dict[str, Any] | None:
    """Collect the final V2 plan, renderer, and source-audit measurements."""
    if not event_path.is_file():
        return None
    selected: dict[str, dict[str, Any]] = {}
    for line in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        event_type = event.get("event_type")
        if event_type in {
            "report_plan_created",
            "report_citations_rendered",
            "report_source_audited",
            "report_grounding_failed",
        }:
            selected[event_type] = event.get("data", {})
    if not selected:
        return None
    plan = selected.get("report_plan_created", {})
    render = selected.get("report_citations_rendered", {})
    audit = selected.get("report_source_audited", {})
    failure = selected.get("report_grounding_failed")
    return {
        "accepted": failure is None,
        "failure_reason": failure.get("reason") if failure else None,
        "sections": plan.get("sections"),
        "planned_claims": plan.get("planned_claims"),
        "claim_evidence_link_rate": plan.get("claim_evidence_link_rate"),
        "remaining_gaps": plan.get("remaining_gaps"),
        "cited_evidence_ids": render.get("cited_evidence_ids"),
        "unknown_evidence_ids": render.get("unknown_evidence_ids"),
        "cited_sources": render.get("cited_sources"),
        "citation_coverage": render.get("citation_coverage"),
        "report_sources": audit.get("report_sources"),
        "verified_sources": audit.get("verified_sources"),
        "source_precision": audit.get("source_precision"),
        "unsupported_sources": audit.get("unsupported_sources"),
    }


def grounding_acceptance_error(
    profile: str, grounding: dict[str, Any] | None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """Reject structured runs that completed without their grounding contract."""
    if profile in {"coverage_v1", "coverage_v2"}:
        # Completeness and semantic checks are recorded separately; citing all
        # planned evidence IDs is deliberately not an acceptance criterion.
        source_check_passed = bool(grounding and (
            grounding.get("source_precision") == 1.0
            if (grounding.get("cited_sources") or 0) > 0 else
            grounding.get("source_precision") is None
            and coverage and coverage.get("verified_blocks") == 0
        ))
        if (coverage and coverage.get("all_retained_blocks_verified")
                and coverage.get("requirements", 0) > 0
                and grounding and grounding.get("accepted")
                and source_check_passed
                and not grounding.get("unknown_evidence_ids")
                and not grounding.get("unsupported_sources")):
            # A complete abstention may have no citations. It is explicitly
            # recorded as incomplete_evidence, never as semantic/coverage gain.
            return None
        return {"type": "GroundingAcceptanceError",
                "message": "Coverage report lacks its completed semantic check"}
    if profile not in {"reliable_v2", "adaptive_v3"}:
        return None
    required = {
        "accepted": True,
        "claim_evidence_link_rate": 1.0,
        "citation_coverage": 1.0,
        "source_precision": 1.0,
    }
    if grounding and all(grounding.get(key) == value for key, value in required.items()):
        if not grounding.get("unknown_evidence_ids") and not grounding.get("unsupported_sources"):
            return None
    reason = (grounding or {}).get("failure_reason") or "grounding_contract_not_met"
    return {
        "type": "GroundingAcceptanceError",
        "message": f"Structured grounding acceptance failed: {reason}",
    }


def question_coverage_metrics(event_path: Path) -> dict[str, Any] | None:
    """Read the final internal completeness check, distinct from test gold coverage."""
    if not event_path.is_file():
        return None
    result = None
    for line in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") == "question_coverage_finished":
            result = event.get("data", {})
    return result


def adaptive_research_metrics(event_path: Path) -> dict[str, Any] | None:
    """Summarize evidence-aware research rounds and their stopping decision."""
    if not event_path.is_file():
        return None
    rounds = []
    final = None
    for line in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") == "adaptive_research_round_finished":
            rounds.append(event.get("data", {}))
        elif event.get("event_type") == "adaptive_research_finished":
            final = event.get("data", {})
    if final is None and not rounds:
        return None
    return {
        "rounds": len(rounds),
        "actions": sum(len(item.get("actions", [])) for item in rounds),
        "evidence_gain": sum(int(item.get("evidence_gain", 0)) for item in rounds),
        "stop_reason": (final or {}).get("stop_reason"),
        "objectives": (final or {}).get("objectives"),
        "decisions": (final or {}).get("decisions"),
        "trace": rounds,
    }


def sentence_audit_metrics(event_path: Path) -> dict[str, Any] | None:
    """Return the final sentence-level semantic audit, separately from URL checks."""
    if not event_path.is_file():
        return None
    for line in reversed(event_path.read_text(encoding="utf-8").splitlines()):
        event = json.loads(line)
        if event.get("event_type") == "report_sentence_audit_finished":
            return {"status": "succeeded", **event.get("data", {})}
        if event.get("event_type") == "report_sentence_audit_failed":
            return {"status": "failed", **event.get("data", {})}
    return None


def final_report_budget_error(event_path: Path) -> dict[str, Any] | None:
    """Return the terminal report-budget error hidden by graph fallback output."""
    if not event_path.is_file():
        return None
    for line in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if (
            event.get("event_type") == "run_budget_exhausted"
            and event.get("node") == "final_report_generation"
        ):
            data = event.get("data", {})
            return {
                "type": "BudgetExceeded",
                "message": (
                    "Final report generation exhausted "
                    f"{data.get('resource', 'a configured budget')}"
                ),
            }
    return None


def adaptive_retrieval_metrics(event_path: Path) -> dict[str, Any] | None:
    """Summarize auditable local-accept and web-escalation decisions."""
    if not event_path.is_file():
        return None
    decisions = []
    web_escalations = 0
    web_failures = 0
    marginal_gains = []
    fusion_events = []
    rerank_events = []
    stopped_low_gain = 0
    for line in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        event_type = event.get("event_type")
        if event_type == "retrieval_route_selected":
            decisions.append(event.get("data", {}))
        elif event_type == "retrieval_web_escalation_succeeded":
            web_escalations += 1
        elif event_type == "retrieval_web_escalation_failed":
            web_failures += 1
        elif event_type == "retrieval_marginal_gain":
            marginal_gains.append(event.get("data", {}))
        elif event_type == "retrieval_candidates_fused":
            fusion_events.append(event.get("data", {}))
        elif event_type == "retrieval_candidates_reranked":
            rerank_events.append(event.get("data", {}))
        elif event_type == "retrieval_stopped_low_gain":
            stopped_low_gain += 1
    if not decisions:
        return None
    return {
        "decisions": len(decisions),
        "local_accepts": sum(
            item.get("action") == "accept_local" for item in decisions
        ),
        "web_escalation_requests": sum(
            item.get("action") == "escalate_web" for item in decisions
        ),
        "web_escalations_succeeded": web_escalations,
        "web_escalations_failed": web_failures,
        "useful_web_supplements": sum(
            bool(item.get("useful")) for item in marginal_gains
        ),
        "low_gain_web_supplements": sum(
            not bool(item.get("useful")) for item in marginal_gains
        ),
        "searches_stopped_after_low_gain": stopped_low_gain,
        "mean_query_term_coverage_gain": (
            round(
                sum(float(item.get("query_term_coverage_gain", 0.0)) for item in marginal_gains)
                / len(marginal_gains),
                4,
            )
            if marginal_gains
            else None
        ),
        "mean_facet_coverage_gain": (
            round(
                sum(float(item.get("facet_coverage_gain", 0.0)) for item in marginal_gains)
                / len(marginal_gains),
                4,
            )
            if marginal_gains
            else None
        ),
        "relevance_rejected_candidates": sum(
            int(item.get("rejected_candidates", 0)) for item in rerank_events
        ),
        "deduplicated_candidates": sum(
            int(item.get("deduplicated", 0)) for item in fusion_events
        ),
        "mean_query_term_coverage": round(
            sum(float(item.get("query_term_coverage", 0.0)) for item in decisions)
            / len(decisions),
            4,
        ),
        "mean_facet_coverage": round(
            sum(float(item.get("facet_coverage", 0.0)) for item in decisions)
            / len(decisions),
            4,
        ),
        "decision_trace": decisions,
    }


def model_routing_metrics(event_path: Path) -> dict[str, Any] | None:
    """Summarize node-level local deployment and paid API selections."""
    if not event_path.is_file():
        return None
    routes = []
    escalations = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") == "model_route_selected":
            routes.append(
                {
                    "node": event.get("node"),
                    **event.get("data", {}),
                }
            )
        elif event.get("event_type") == "model_route_escalated":
            escalations.append(
                {"node": event.get("node"), **event.get("data", {})}
            )
    if not routes and not escalations:
        return None
    counts: dict[str, int] = {}
    for route in routes:
        tier = str(route.get("tier", "unknown"))
        counts[tier] = counts.get(tier, 0) + 1
    return {
        "selections": len(routes),
        "tier_counts": counts,
        "escalations": len(escalations),
        "trace": routes,
        "escalation_trace": escalations,
    }


def claim_validation_metrics(event_path: Path) -> dict[str, Any] | None:
    """Return the latest semantic claim-admission audit."""
    if not event_path.is_file():
        return None
    audits = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") == "claim_evidence_validation_finished":
            audits.append(event.get("data", {}))
    if not audits:
        return None
    return {**audits[-1], "validation_rounds": len(audits)}


async def run_case(case: dict[str, Any], args: argparse.Namespace, root: Path) -> dict[str, Any]:
    run_id = f"{case['id']}-{uuid4().hex[:8]}"
    case_dir = root / run_id
    case_dir.mkdir(parents=True, exist_ok=False)
    event_path = case_dir / "events.jsonl"
    configuration = {
        "thread_id": run_id,
        "run_id": run_id,
        "event_log_path": str(event_path),
        "runtime_db_path": str(case_dir / "runtime.db"),
        "observability_enabled": True,
        "allow_clarification": False,
        "max_concurrent_research_units": args.concurrency,
        "max_total_research_units": args.max_total_research_units,
        "max_initial_research_units": args.max_initial_research_units,
        "max_researcher_iterations": args.max_researcher_iterations,
        "max_react_tool_calls": args.max_react_tool_calls,
        "native_structured_output": not args.plain_json_output,
        "max_search_queries": args.max_search_queries,
        "max_search_results": args.max_search_results,
        "local_corpus_path": str(args.local_corpus_path) if args.local_corpus_path else None,
        "local_retrieval_strategy": args.local_retrieval_strategy,
        "local_retrieval_candidate_k": args.local_retrieval_candidate_k,
        "local_retrieval_rrf_k": args.local_retrieval_rrf_k,
        "local_retrieval_mmr_lambda": args.local_retrieval_mmr_lambda,
        "hybrid_local_min_results": args.hybrid_local_min_results,
        "hybrid_local_min_unique_documents": args.hybrid_local_min_unique_documents,
        "hybrid_local_min_query_term_coverage": args.hybrid_local_min_query_term_coverage,
        "hybrid_min_facet_coverage": args.hybrid_min_facet_coverage,
        "hybrid_min_candidate_relevance": args.hybrid_min_candidate_relevance,
        "hybrid_min_facet_relevance": args.hybrid_min_facet_relevance,
        "hybrid_min_source_quality": args.hybrid_min_source_quality,
        "hybrid_max_chunks_per_document": args.hybrid_max_chunks_per_document,
        "hybrid_min_new_documents": args.hybrid_min_new_documents,
        "hybrid_min_query_term_coverage_gain": args.hybrid_min_query_term_coverage_gain,
        "hybrid_max_consecutive_low_gain": args.hybrid_max_consecutive_low_gain,
        "hybrid_fusion_rrf_k": args.hybrid_fusion_rrf_k,
        "local_embedding_model": args.local_embedding_model,
        "local_embedding_threads": args.local_embedding_threads,
        "local_model_cache_path": str(args.local_model_cache_path),
        "local_embedding_cache_path": (
            str(args.local_embedding_cache_path) if args.local_embedding_cache_path else None
        ),
        "enable_claim_extraction": args.extract_claims,
        "enable_claim_evidence_validation": (
            args.profile in {"reliable_v2", "adaptive_v3"}
            and not args.skip_claim_evidence_validation
        ),
        "research_task_timeout_seconds": args.research_task_timeout,
        "claim_extraction_timeout_seconds": args.claim_extraction_timeout,
        "enable_evidence_context": args.use_evidence_context,
        "evidence_context_max_characters": args.evidence_context_max_characters,
        "enable_report_review": args.review_report,
        "enable_structured_report": args.profile == "reliable_v2",
        "enable_question_coverage": args.profile in {"coverage_v1", "coverage_v2"},
        "enable_coverage_feedback_repair": args.profile == "coverage_v2",
        "structured_json_repair_attempts": 1 if args.profile == "coverage_v2" else 0,
        "enable_gap_search": args.profile == "reliable_v2",
        "enable_adaptive_research": args.profile == "adaptive_v3",
        "enable_agentic_research_policy": (
            args.profile == "adaptive_v3"
            and not args.deterministic_adaptive_policy
        ),
        "max_adaptive_rounds": args.max_adaptive_rounds,
        "max_adaptive_actions_per_round": args.max_adaptive_actions_per_round,
        "max_objective_search_attempts": args.max_objective_search_attempts,
        "adaptive_min_action_utility": args.adaptive_min_action_utility,
        "adaptive_min_evidence_gain": args.adaptive_min_evidence_gain,
        "adaptive_budget_reserve_fraction": args.adaptive_budget_reserve_fraction,
        "enable_sentence_audit": args.profile == "adaptive_v3",
        "sentence_audit_apply_repairs": not args.no_sentence_audit_repairs,
        "enable_text_tool_call_fallback": args.text_tool_call_fallback,
        "report_review_max_rounds": args.review_max_rounds,
        "experiment_profile": args.profile,
        "enable_reliable_runtime": args.profile != "baseline",
        "max_llm_calls": args.max_llm_calls,
        "max_search_calls": args.max_search_calls,
        "max_task_attempts": args.max_task_attempts,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_completion_tokens": args.max_completion_tokens,
        "max_total_seconds": args.max_total_seconds,
        "model_api_key_env": args.api_key_env,
        "enable_adaptive_model_routing": args.adaptive_model_routing,
        "local_deployment_model": args.local_deployment_model,
        "local_deployment_base_url": args.local_deployment_base_url,
        "local_deployment_api_key_env": args.local_deployment_api_key_env,
        "local_deployment_max_tokens": args.local_deployment_max_tokens,
        "paid_api_model": args.paid_api_model,
        "paid_api_base_url": args.paid_api_base_url,
        "paid_api_key_env": args.paid_api_key_env,
        "paid_api_max_tokens": args.paid_api_max_tokens,
        "paid_api_nodes": args.paid_api_node or [
            "final_report_generation",
            "report_review",
            "report_sentence_audit",
        ],
        "research_model_max_tokens": args.research_max_tokens,
        "compression_model_max_tokens": args.compression_max_tokens,
        "final_report_model_max_tokens": args.final_report_max_tokens,
        "summarization_model_max_tokens": args.summarization_max_tokens,
    }
    if args.profile == "baseline":
        configuration.update({
            "enable_claim_extraction": False,
            "enable_evidence_context": False,
            "enable_report_review": False,
        })
    elif args.profile == "reviewed":
        configuration.update({
            "enable_claim_extraction": True,
            "enable_evidence_context": True,
            "enable_report_review": True,
        })
    elif args.profile in {"reliable_v2", "adaptive_v3", "coverage_v1", "coverage_v2"}:
        configuration.update({
            "enable_claim_extraction": args.profile not in {"coverage_v1", "coverage_v2"},
            "enable_evidence_context": True,
            "enable_structured_report": True,
            "enable_gap_search": args.profile in {"reliable_v2", "coverage_v1", "coverage_v2"},
            "enable_adaptive_research": args.profile == "adaptive_v3",
            "enable_sentence_audit": args.profile == "adaptive_v3",
        })
        # Keep the V2 defaults conservative, but never override an explicit
        # command-line budget. This is important for low-cost smoke runs.
        if args.max_initial_research_units is None:
            configuration["max_initial_research_units"] = 2
        if args.max_total_research_units == 10:
            configuration["max_total_research_units"] = (
                6 if args.profile == "adaptive_v3" else 4 if args.profile == "coverage_v2" else 3
            )
    for key in ("research_model", "compression_model", "final_report_model", "summarization_model", "search_api", "base_url"):
        value = getattr(args, key)
        if value is not None:
            configuration[key] = value
    budget = RunBudget(
        max_llm_calls=args.max_llm_calls,
        max_search_calls=args.max_search_calls,
        max_task_attempts=args.max_task_attempts,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
        max_total_seconds=args.max_total_seconds,
    )
    config = {"configurable": {**configuration, "run_budget": budget}}
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    runtime_store = RuntimeStore(configuration["runtime_db_path"])

    started = time.perf_counter()
    record_event(config, "run_started", "runtime", data={"case_id": case["id"], "category": case["category"]})
    status = "succeeded"
    error = None
    state: dict[str, Any] = {}
    try:
        invocation = graph.ainvoke(
            {"messages": [{"role": "user", "content": case["prompt"]}]}, config
        )
        state = await asyncio.wait_for(invocation, timeout=args.max_total_seconds)
    except (BudgetExceeded, asyncio.TimeoutError) as exception:
        if isinstance(exception, asyncio.TimeoutError):
            try:
                budget.check_time()
            except BudgetExceeded as budget_error:
                exception = budget_error
        status = "budget_exhausted"
        error = {"type": type(exception).__name__, "message": str(exception)}
        record_event(
            config,
            "run_budget_exhausted",
            "runtime",
            data={"error": error, "budget": budget.snapshot().model_dump()},
        )
    except Exception as exception:
        status = "failed"
        error = {"type": type(exception).__name__, "message": str(exception)}
    elapsed = round(time.perf_counter() - started, 3)
    report = str(state.get("final_report", ""))
    report_quality_status = str(state.get("report_quality_status", "not_audited"))
    research_tasks, successful_research_tasks = runtime_store.task_counts()
    evidence_counts = runtime_store.evidence_counts()
    report_budget_error = final_report_budget_error(event_path)
    if status == "succeeded" and report_budget_error:
        status = "budget_exhausted"
        error = report_budget_error
    if status == "succeeded" and (not report or report.startswith("Error generating final report:")):
        status = "failed"
        error = {"type": "ReportGenerationError", "message": report or "No final report was produced"}
    if status == "succeeded" and report_quality_status == "degraded_needs_review":
        status = "degraded"
        error = {
            "type": "ReportQualityBelowThreshold",
            "message": "Sentence-level semantic support is below the configured threshold",
        }
    grounding = report_grounding_metrics(event_path)
    question_coverage = question_coverage_metrics(event_path)
    grounding_error = grounding_acceptance_error(args.profile, grounding, question_coverage)
    if status == "succeeded" and grounding_error:
        status = "failed"
        error = grounding_error
    summary = {
        "run_id": run_id,
        "case_id": case["id"],
        "category": case["category"],
        "status": status,
        "report_quality_status": report_quality_status,
        "elapsed_seconds": elapsed,
        **report_metrics(report),
        "error": error,
        "research_tasks": research_tasks,
        "successful_research_tasks": successful_research_tasks,
        "research_results_saved": successful_research_tasks > 0,
        "evidence_collected": evidence_counts["evidence"] > 0,
        "evidence_counts": evidence_counts,
        "report_review": review_metrics(event_path),
        "report_grounding": grounding,
        "adaptive_retrieval": adaptive_retrieval_metrics(event_path),
        "model_routing": model_routing_metrics(event_path),
        "claim_evidence_validation": claim_validation_metrics(event_path),
        "adaptive_research": adaptive_research_metrics(event_path),
        "sentence_audit": sentence_audit_metrics(event_path),
        "question_coverage": question_coverage,
        "configuration": configuration,
        "budget": budget.snapshot().model_dump(),
    }
    (case_dir / "input.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "report.md").write_text(report, encoding="utf-8")
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    record_event(config, "run_finished", "runtime", data=summary)
    return summary


async def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "baseline_questions.jsonl",
    )
    parser.add_argument("--case", action="append", default=[], help="Case id to run; repeat to select multiple")
    parser.add_argument("--question", help="Run one question directly instead of loading a question file")
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-total-research-units", type=int, default=10)
    parser.add_argument("--max-initial-research-units", type=int)
    parser.add_argument("--max-researcher-iterations", type=int, default=6)
    parser.add_argument("--max-react-tool-calls", type=int, default=10)
    parser.add_argument(
        "--profile",
        choices=("baseline", "reliable", "reviewed", "reliable_v2", "adaptive_v3", "coverage_v1", "coverage_v2"),
        default="reliable",
    )
    parser.add_argument("--research-model")
    parser.add_argument("--compression-model")
    parser.add_argument("--final-report-model")
    parser.add_argument("--summarization-model")
    parser.add_argument(
        "--search-api",
        choices=("tavily", "openai", "anthropic", "local", "hybrid", "none"),
    )
    parser.add_argument("--local-corpus-path", type=Path)
    parser.add_argument(
        "--local-retrieval-strategy",
        choices=(
            "bm25",
            "tfidf_vector",
            "rrf_hybrid",
            "rrf_mmr",
            "semantic_vector",
            "bm25_semantic_rrf",
            "semantic_rrf_mmr",
        ),
        default="bm25",
    )
    parser.add_argument("--local-embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--local-embedding-threads", type=int, default=8)
    parser.add_argument("--local-model-cache-path", type=Path, default=Path("benchmark_data/models"))
    parser.add_argument("--local-embedding-cache-path", type=Path)
    parser.add_argument("--local-retrieval-candidate-k", type=int, default=40)
    parser.add_argument("--local-retrieval-rrf-k", type=int, default=60)
    parser.add_argument("--local-retrieval-mmr-lambda", type=float, default=0.7)
    parser.add_argument("--hybrid-local-min-results", type=int, default=3)
    parser.add_argument("--hybrid-local-min-unique-documents", type=int, default=2)
    parser.add_argument(
        "--hybrid-local-min-query-term-coverage", type=float, default=0.75
    )
    parser.add_argument("--hybrid-min-facet-coverage", type=float, default=1.0)
    parser.add_argument("--hybrid-min-candidate-relevance", type=float, default=0.35)
    parser.add_argument("--hybrid-min-facet-relevance", type=float, default=0.55)
    parser.add_argument("--hybrid-min-source-quality", type=float, default=0.35)
    parser.add_argument("--hybrid-max-chunks-per-document", type=int, default=2)
    parser.add_argument("--hybrid-min-new-documents", type=int, default=1)
    parser.add_argument(
        "--hybrid-min-query-term-coverage-gain", type=float, default=0.05
    )
    parser.add_argument(
        "--hybrid-max-consecutive-low-gain", type=int, default=2
    )
    parser.add_argument("--hybrid-fusion-rrf-k", type=int, default=60)
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--api-key-env",
        help="Environment variable containing the model API key; the key itself is never persisted",
    )
    parser.add_argument("--adaptive-model-routing", action="store_true")
    parser.add_argument("--local-deployment-model")
    parser.add_argument("--local-deployment-base-url")
    parser.add_argument("--local-deployment-api-key-env")
    parser.add_argument("--local-deployment-max-tokens", type=int)
    parser.add_argument("--paid-api-model")
    parser.add_argument("--paid-api-base-url")
    parser.add_argument("--paid-api-key-env")
    parser.add_argument("--paid-api-max-tokens", type=int)
    parser.add_argument(
        "--paid-api-node",
        action="append",
        help="High-risk workflow node assigned to the paid API; repeat as needed",
    )
    parser.add_argument("--plain-json-output", action="store_true", help="Avoid provider-native structured output")
    parser.add_argument("--disable-thinking", action="store_true", help="Use non-thinking mode for an OpenAI-compatible comparison")
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"))
    parser.add_argument("--streaming", action="store_true", help="Aggregate streamed responses to avoid gateway idle timeouts")
    parser.add_argument(
        "--text-tool-call-fallback",
        action="store_true",
        help="Parse native call:Tool{args} output from local models",
    )
    parser.add_argument("--max-search-queries", type=int, default=3)
    parser.add_argument("--max-search-results", type=int, default=5)
    parser.add_argument("--extract-claims", action="store_true", help="Add one claim-extraction call per research task")
    parser.add_argument(
        "--skip-claim-evidence-validation",
        action="store_true",
        help="Disable semantic claim admission in the reliable_v2 profile",
    )
    parser.add_argument("--research-task-timeout", type=float, default=120.0)
    parser.add_argument("--claim-extraction-timeout", type=float, default=30.0)
    parser.add_argument("--use-evidence-context", action="store_true", help="Write the report from persisted evidence")
    parser.add_argument("--evidence-context-max-characters", type=int, default=30000)
    parser.add_argument("--review-report", action="store_true", help="Run bounded Red-Blue report review")
    parser.add_argument("--review-max-rounds", type=int, default=1)
    parser.add_argument("--max-adaptive-rounds", type=int, default=2)
    parser.add_argument("--max-adaptive-actions-per-round", type=int, default=2)
    parser.add_argument("--max-objective-search-attempts", type=int, default=2)
    parser.add_argument("--adaptive-min-action-utility", type=float, default=0.15)
    parser.add_argument("--adaptive-min-evidence-gain", type=int, default=1)
    parser.add_argument("--adaptive-budget-reserve-fraction", type=float, default=0.2)
    parser.add_argument(
        "--deterministic-adaptive-policy",
        action="store_true",
        help="Use the legacy utility formula instead of the structured policy agent",
    )
    parser.add_argument("--no-sentence-audit-repairs", action="store_true")
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--max-search-calls", type=int)
    parser.add_argument("--max-task-attempts", type=int)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--max-total-seconds", type=float, default=600)
    parser.add_argument("--research-max-tokens", type=int, default=10000)
    parser.add_argument("--compression-max-tokens", type=int, default=8192)
    parser.add_argument("--final-report-max-tokens", type=int, default=10000)
    parser.add_argument("--summarization-max-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.question is not None and (not args.question.strip() or args.case):
        parser.error("--question must be nonempty and cannot be combined with --case")

    if args.disable_thinking or args.reasoning_effort or args.streaming:
        if not args.research_model or not args.api_key_env or not args.base_url:
            parser.error("Model transport overrides require --research-model, --api-key-env, and --base-url")
        from open_deep_research import deep_researcher

        deep_researcher.configurable_model = init_chat_model(
            model=args.research_model,
            api_key=os.environ[args.api_key_env],
            base_url=args.base_url,
            **({"extra_body": {"thinking": {"type": "disabled"}}} if args.disable_thinking else {}),
            **({"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}),
            **({"streaming": True, "stream_usage": True} if args.streaming else {}),
            configurable_fields=("model", "max_tokens", "api_key", "base_url"),
        )

    cases = ([{"id": "question", "category": "custom", "prompt": args.question}]
             if args.question is not None else load_cases(args.questions, set(args.case)))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output / f"{timestamp}-{uuid4().hex[:6]}"
    root.mkdir(parents=True, exist_ok=False)
    summaries = []
    for case in cases:
        summary = await run_case(case, args, root)
        summaries.append(summary)
        print(f"{summary['case_id']}: {summary['status']} ({summary['elapsed_seconds']}s)")
    (root / "manifest.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(summary["status"] != "succeeded" for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
