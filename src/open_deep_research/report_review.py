"""Bounded Red-Blue review loop for evidence-grounded reports."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.utils import ainvoke_structured_with_fallback


class IssueType(str, Enum):
    FACTUALITY = "FACTUALITY"
    CITATION = "CITATION"
    LOGIC = "LOGIC"
    COVERAGE = "COVERAGE"
    SOURCE_QUALITY = "SOURCE_QUALITY"


class RepairOperation(str, Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    VERIFY = "VERIFY"


class ReviewIssue(BaseModel):
    issue_type: IssueType
    severity: int = Field(ge=1, le=5)
    description: str = Field(min_length=1)
    report_excerpt: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ReportReview(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    issues: list[ReviewIssue] = Field(default_factory=list)


class RepairAction(BaseModel):
    operation: RepairOperation
    target_excerpt: str = ""
    replacement: str = ""
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class RepairedReport(BaseModel):
    actions: list[RepairAction] = Field(default_factory=list)


class RejectedRepairAction(BaseModel):
    action: RepairAction
    rejection_reason: str


class ReviewRound(BaseModel):
    round_number: int
    score: float
    issue_count: int
    actions: list[RepairAction] = Field(default_factory=list)
    rejected_actions: list[RejectedRepairAction] = Field(default_factory=list)


class ReviewOutcome(BaseModel):
    report: str
    final_score: float
    stop_reason: str
    rounds: list[ReviewRound]


def _report_hash(report: str) -> str:
    return hashlib.sha256(report.encode("utf-8")).hexdigest()


def _evidence_ids(context: str) -> set[str]:
    return set(re.findall(r"\bev(?::|-)[A-Za-z0-9]+(?:[._:-][A-Za-z0-9]+)*", context))


def apply_repair_actions(
    report: str,
    actions: list[RepairAction],
    valid_evidence_ids: set[str],
) -> tuple[str, list[RepairAction], list[RejectedRepairAction]]:
    """Validate a model-proposed patch plan and apply accepted edits in order."""
    revised = report
    accepted: list[RepairAction] = []
    rejected: list[RejectedRepairAction] = []

    for action in actions:
        reason: str | None = None
        if action.operation in {RepairOperation.ADD, RepairOperation.MODIFY}:
            if not action.evidence_ids:
                reason = "ADD and MODIFY require at least one evidence_id"
            elif unknown := set(action.evidence_ids) - valid_evidence_ids:
                reason = f"Unknown evidence IDs: {', '.join(sorted(unknown))}"
        if reason is None and action.operation != RepairOperation.VERIFY:
            if not action.target_excerpt:
                reason = "Mutation requires a non-empty target_excerpt"
            elif action.target_excerpt not in revised:
                reason = "target_excerpt was not found in the current report"
        if reason is None and action.operation in {RepairOperation.ADD, RepairOperation.MODIFY}:
            if not action.replacement:
                reason = "ADD and MODIFY require a non-empty replacement"

        if reason is not None:
            rejected.append(RejectedRepairAction(action=action, rejection_reason=reason))
            continue

        if action.operation == RepairOperation.ADD:
            revised = revised.replace(
                action.target_excerpt,
                action.target_excerpt + action.replacement,
                1,
            )
        elif action.operation == RepairOperation.DELETE:
            revised = revised.replace(action.target_excerpt, "", 1)
        elif action.operation == RepairOperation.MODIFY:
            revised = revised.replace(action.target_excerpt, action.replacement, 1)
        accepted.append(action)

    return revised, accepted, rejected


async def run_report_review(
    model: BaseChatModel,
    *,
    report: str,
    evidence_context: str,
    max_rounds: int = 2,
    quality_threshold: float = 0.8,
    min_improvement: float = 0.02,
    native_structured_output: bool = True,
    retries: int = 3,
    config: RunnableConfig | None = None,
) -> ReviewOutcome:
    """Review and repair until quality converges or a hard guard stops it."""
    current = report
    hashes = {_report_hash(current)}
    rounds: list[ReviewRound] = []
    previous_score: float | None = None
    best_report = current
    best_score = -1.0
    valid_evidence_ids = _evidence_ids(evidence_context)

    for round_number in range(1, max_rounds + 2):
        review_prompt = (
            "Act as the Red reviewer. Find only actionable factuality, citation, "
            "logic, coverage, and source-quality issues. Evidence IDs must come "
            "from the supplied context.\n\n"
            f"EVIDENCE:\n{evidence_context}\n\nREPORT:\n{current}"
        )
        review = await ainvoke_structured_with_fallback(
            model,
            ReportReview,
            [HumanMessage(content=review_prompt)],
            retries=retries,
            native_structured_output=native_structured_output,
            config=config,
            node="report_red_review",
        )
        current_round = ReviewRound(
            round_number=round_number,
            score=review.score,
            issue_count=len(review.issues),
        )
        rounds.append(current_round)
        if review.score > best_score:
            best_report = current
            best_score = review.score

        if review.score >= quality_threshold or not review.issues:
            return ReviewOutcome(
                report=current,
                final_score=review.score,
                stop_reason="quality_threshold",
                rounds=rounds,
            )
        if previous_score is not None and review.score - previous_score < min_improvement:
            return ReviewOutcome(
                report=best_report,
                final_score=best_score,
                stop_reason="insufficient_improvement",
                rounds=rounds,
            )
        if round_number > max_rounds:
            return ReviewOutcome(
                report=best_report,
                final_score=best_score,
                stop_reason="max_rounds",
                rounds=rounds,
            )

        repair_prompt = (
            "Act as the Blue repair planner. Propose only ADD, DELETE, MODIFY, or "
            "VERIFY actions needed to fix the supplied issues. Do not return a "
            "rewritten report: code will validate and apply your actions. ADD and "
            "MODIFY must list one or more evidence_ids copied exactly from the "
            "evidence context. For ADD, replacement is inserted immediately after "
            "target_excerpt. For MODIFY, replacement substitutes target_excerpt.\n\n"
            f"EVIDENCE:\n{evidence_context}\n\nREPORT:\n{current}\n\n"
            f"ISSUES:\n{json.dumps(review.model_dump(), ensure_ascii=False)}"
        )
        repaired = await ainvoke_structured_with_fallback(
            model,
            RepairedReport,
            [HumanMessage(content=repair_prompt)],
            retries=retries,
            native_structured_output=native_structured_output,
            config=config,
            node="report_blue_repair",
        )
        revised_report, accepted, rejected = apply_repair_actions(
            current, repaired.actions, valid_evidence_ids
        )
        current_round.actions = accepted
        current_round.rejected_actions = rejected
        revised_hash = _report_hash(revised_report)
        if revised_hash in hashes:
            return ReviewOutcome(
                report=current,
                final_score=review.score,
                stop_reason="oscillation_detected",
                rounds=rounds,
            )
        hashes.add(revised_hash)
        previous_score = review.score
        current = revised_report

    raise AssertionError("Review loop exited unexpectedly")
