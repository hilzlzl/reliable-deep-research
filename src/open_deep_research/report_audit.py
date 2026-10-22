"""Sentence-level semantic audit and constrained repair for grounded reports."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.evidence import ClaimStatus
from open_deep_research.utils import ainvoke_structured_with_fallback


_MARKER_RE = re.compile(r"\[EVIDENCE:([^\]]+)\]", re.IGNORECASE)


class AssertionRepair(str, Enum):
    KEEP = "KEEP"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    SEARCH = "SEARCH"


class ReportAssertion(BaseModel):
    assertion_id: str
    sentence: str
    evidence_ids: list[str] = Field(default_factory=list)
    verdict: ClaimStatus
    reason: str
    repair_action: AssertionRepair = AssertionRepair.KEEP
    replacement: str = ""


class ReportAssertionBatch(BaseModel):
    assertions: list[ReportAssertion] = Field(default_factory=list)


class ReportAuditOutcome(BaseModel):
    report: str
    assertions: list[ReportAssertion]
    total_assertions: int
    supported: int
    partially_supported: int
    contradicted: int
    not_supported: int
    support_rate: float | None
    cited_assertions: int = 0
    citation_coverage: float | None = None
    audit_coverage: float | None = None
    available_evidence_markers: int = 0
    repairs_applied: int
    repairs_verified: int = 0
    repairs_rejected: int = 0


def _assertion_id(sentence: str) -> str:
    digest = hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]
    return f"assertion:{digest}"


def _adjacent_evidence_ids(
    report: str, sentence: str, known_ids: set[str]
) -> list[str]:
    """Resolve markers in the assertion or later on the same Markdown line."""
    direct = set(_MARKER_RE.findall(sentence)) & known_ids
    if direct:
        return sorted(direct)
    start = report.find(sentence)
    if start < 0:
        return []
    block_start = report.rfind("\n\n", 0, start)
    block_start = 0 if block_start < 0 else block_start + 2
    block_end = report.find("\n\n", start + len(sentence))
    if block_end < 0:
        block_end = len(report)
    # Models commonly preserve the sentence exactly but omit a trailing
    # citation marker from the extracted sentence. The marker is still
    # adjacent when it appears later in the same bullet/line.
    block = report[block_start:block_end]
    return sorted(set(_MARKER_RE.findall(block)) & known_ids)


def _sanitize_assertions(
    report: str,
    evidence: list[dict],
    proposed: list[ReportAssertion],
) -> list[ReportAssertion]:
    known_ids = {str(item["evidence_id"]) for item in evidence}
    output: list[ReportAssertion] = []
    seen_sentences: set[str] = set()
    for item in proposed:
        sentence = item.sentence.strip()
        if not sentence or sentence not in report or sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        cited_ids = _adjacent_evidence_ids(report, sentence, known_ids)
        admitted_ids = cited_ids
        verdict = item.verdict
        action = item.repair_action
        replacement = item.replacement.strip()
        if not admitted_ids and verdict == ClaimStatus.SUPPORTED:
            verdict = ClaimStatus.NOT_SUPPORTED
            action = AssertionRepair.SEARCH
        if action == AssertionRepair.MODIFY:
            replacement_ids = set(_MARKER_RE.findall(replacement))
            if (
                not replacement
                or "http://" in replacement
                or "https://" in replacement
                or not replacement_ids
                or not replacement_ids.issubset(set(admitted_ids))
            ):
                action = AssertionRepair.SEARCH
                replacement = ""
        elif action == AssertionRepair.DELETE:
            replacement = ""
        else:
            replacement = ""
        output.append(
            ReportAssertion(
                assertion_id=_assertion_id(sentence),
                sentence=sentence,
                evidence_ids=admitted_ids,
                verdict=verdict,
                reason=item.reason,
                repair_action=action,
                replacement=replacement,
            )
        )
    return output


def _audit_body(report: str) -> str:
    """Remove bibliography and non-assertive gap sections before semantic audit."""
    # The renderer owns the bibliography.  Auditing source-list URLs as prose
    # assertions creates false NOT_SUPPORTED findings and dilutes the metric.
    body = re.split(r"(?im)^#{1,3}\s+(?:sources|references)\s*$", report, maxsplit=1)[0]
    # Gaps are deliberately unresolved; they are tracked by the report plan and
    # should not be counted as failed factual assertions.
    body = re.split(r"(?im)^#{1,3}\s+evidence gaps\s*$", body, maxsplit=1)[0]
    return body.strip()


def apply_assertion_repairs(
    report: str, assertions: list[ReportAssertion]
) -> tuple[str, int]:
    """Apply only exact, locally validated MODIFY and DELETE operations."""
    revised = report
    applied = 0
    for item in assertions:
        if item.sentence not in revised:
            continue
        if item.repair_action == AssertionRepair.MODIFY and item.replacement:
            revised = revised.replace(item.sentence, item.replacement, 1)
            applied += 1
        elif item.repair_action == AssertionRepair.DELETE:
            revised = revised.replace(item.sentence, "", 1)
            applied += 1
    return revised, applied


async def audit_report_assertions(
    model: BaseChatModel,
    *,
    report: str,
    evidence: list[dict],
    apply_repairs: bool,
    native_structured_output: bool,
    retries: int,
    config: RunnableConfig | None = None,
) -> ReportAuditOutcome:
    """Audit each factual assertion only against its adjacent cited Evidence IDs."""
    compact_evidence = {
        str(item["evidence_id"]): {
            "source_title": item.get("source_title", ""),
            "excerpt": str(item.get("excerpt", ""))[:1200],
        }
        for item in evidence
    }
    audit_body = _audit_body(report)
    prompt = (
        "Extract every externally verifiable factual assertion from the report body. "
        "Ignore headings, bibliography/source lists, and the Evidence gaps section. "
        "Preserve "
        "the exact sentence text, including [EVIDENCE:id] markers. Judge each assertion "
        "only against the evidence IDs cited in that sentence. Use SUPPORTED only when "
        "the complete assertion follows, PARTIALLY_SUPPORTED for overstatement or partial "
        "coverage, CONTRADICTED for opposition, and NOT_SUPPORTED otherwise. Choose KEEP, "
        "MODIFY, DELETE, or SEARCH. A MODIFY replacement must be a complete weaker sentence "
        "using only supplied evidence markers. Do not repair stylistic text.\n\n"
        f"EVIDENCE:\n{json.dumps(compact_evidence, ensure_ascii=False)}\n\n"
        f"REPORT BODY:\n{audit_body}"
    )
    response = await ainvoke_structured_with_fallback(
        model,
        ReportAssertionBatch,
        [HumanMessage(content=prompt)],
        retries=retries,
        native_structured_output=native_structured_output,
        config=config,
        node="report_sentence_audit",
    )
    assertions = _sanitize_assertions(report, evidence, response.assertions)
    initial_assertions = assertions
    revised, tentative_repairs = (
        apply_assertion_repairs(report, assertions) if apply_repairs else (report, 0)
    )
    repairs_verified = 0
    repairs_rejected = 0
    repairs_applied = tentative_repairs
    if apply_repairs and tentative_repairs:
        final_body = _audit_body(revised)
        verification_prompt = (
            "Independently audit every externally verifiable factual assertion in this "
            "revised report. Do not rely on a previous audit. Preserve exact sentence "
            "text and judge only against Evidence IDs cited in that sentence. Use "
            "SUPPORTED only when the complete sentence follows from the evidence. Set "
            "repair_action to KEEP and replacement to an empty string.\n\n"
            f"EVIDENCE:\n{json.dumps(compact_evidence, ensure_ascii=False)}\n\n"
            f"REVISED REPORT BODY:\n{final_body}"
        )
        verification = await ainvoke_structured_with_fallback(
            model,
            ReportAssertionBatch,
            [HumanMessage(content=verification_prompt)],
            retries=retries,
            native_structured_output=native_structured_output,
            config=config,
            node="report_repair_verification",
        )
        final_assertions = _sanitize_assertions(
            revised, evidence, verification.assertions
        )
        final_verdicts = {item.sentence: item.verdict for item in final_assertions}
        modified = [
            item for item in initial_assertions
            if item.repair_action == AssertionRepair.MODIFY and item.replacement
        ]
        rejected_replacements = {
            item.replacement
            for item in modified
            if final_verdicts.get(item.replacement) != ClaimStatus.SUPPORTED
        }
        repairs_rejected = len(rejected_replacements)
        repairs_verified = len(modified) - repairs_rejected
        for replacement in rejected_replacements:
            revised = revised.replace(replacement, "", 1)
        final_assertions = [
            item for item in final_assertions if item.sentence not in rejected_replacements
        ]
        assertions = final_assertions
        deleted = sum(
            item.repair_action == AssertionRepair.DELETE for item in initial_assertions
        )
        repairs_applied = deleted + repairs_verified
    counts = {status.value: 0 for status in ClaimStatus}
    for item in assertions:
        counts[item.verdict.value] += 1
    supported = counts[ClaimStatus.SUPPORTED.value]
    total = len(assertions)
    cited_assertions = sum(bool(item.evidence_ids) for item in assertions)
    return ReportAuditOutcome(
        report=revised,
        assertions=assertions,
        total_assertions=total,
        supported=supported,
        partially_supported=counts[ClaimStatus.PARTIALLY_SUPPORTED.value],
        contradicted=counts[ClaimStatus.CONTRADICTED.value],
        not_supported=counts[ClaimStatus.NOT_SUPPORTED.value],
        support_rate=(round(supported / total, 4) if total else None),
        cited_assertions=cited_assertions,
        citation_coverage=(round(cited_assertions / total, 4) if total else None),
        # This is populated against the extracted assertion set.  It is kept
        # separate from semantic support so an empty model extraction is visible.
        audit_coverage=(round(total / max(1, len(audit_body.splitlines())), 4)),
        available_evidence_markers=len(_MARKER_RE.findall(_audit_body(revised))),
        repairs_applied=repairs_applied,
        repairs_verified=repairs_verified,
        repairs_rejected=repairs_rejected,
    )
