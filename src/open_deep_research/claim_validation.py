"""Semantic admission control for evidence-linked research claims."""

from __future__ import annotations

import json
import re
from collections import Counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.evidence import ClaimStatus, ResearchClaim
from open_deep_research.observability import record_event
from open_deep_research.run_budget import BudgetExceeded
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.utils import ainvoke_structured_with_fallback


class ClaimEvidenceJudgment(BaseModel):
    claim_id: str
    verdict: ClaimStatus
    supported_evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class ClaimEvidenceValidation(BaseModel):
    judgments: list[ClaimEvidenceJudgment] = Field(default_factory=list)


class ClaimValidationAudit(BaseModel):
    total_claims: int
    supported: int = 0
    partially_supported: int = 0
    contradicted: int = 0
    not_supported: int = 0
    fallback_used: bool = False
    supported_rate: float | None = None


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_NEGATIONS = {"no", "not", "never", "without", "cannot", "can't"}
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "with",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOP_WORDS and token.lower() not in _NEGATIONS
    }


def deterministic_claim_judgment(
    claim: dict, evidence_by_id: dict[str, dict]
) -> ClaimEvidenceJudgment:
    """Conservative fallback used when semantic validation is unavailable."""
    linked = [
        evidence_by_id[evidence_id]
        for evidence_id in claim.get("evidence_ids", [])
        if evidence_id in evidence_by_id
    ]
    if not linked:
        return ClaimEvidenceJudgment(
            claim_id=claim["claim_id"],
            verdict=ClaimStatus.NOT_SUPPORTED,
            reason="No known linked evidence.",
        )
    statement = str(claim["statement"])
    evidence_text = " ".join(
        f"{item.get('claim_text', '')} {item.get('excerpt', '')}" for item in linked
    )
    claim_tokens = _content_tokens(statement)
    evidence_tokens = _content_tokens(evidence_text)
    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens) if claim_tokens else 0.0
    claim_numbers = set(_NUMBER_RE.findall(statement))
    evidence_numbers = set(_NUMBER_RE.findall(evidence_text))
    numbers_match = claim_numbers.issubset(evidence_numbers)
    claim_negated = bool(set(token.lower() for token in _TOKEN_RE.findall(statement)) & _NEGATIONS)
    evidence_negated = bool(set(token.lower() for token in _TOKEN_RE.findall(evidence_text)) & _NEGATIONS)
    evidence_ids = [item["evidence_id"] for item in linked]

    if overlap >= 0.8 and numbers_match and claim_negated != evidence_negated:
        verdict = ClaimStatus.CONTRADICTED
        reason = "Claim and evidence have opposing negation with high lexical alignment."
    elif overlap >= 0.8 and numbers_match:
        verdict = ClaimStatus.SUPPORTED
        reason = "High lexical coverage and matching numeric facts."
    elif overlap >= 0.5 and numbers_match:
        verdict = ClaimStatus.PARTIALLY_SUPPORTED
        reason = "Evidence covers only part of the claim."
    else:
        verdict = ClaimStatus.NOT_SUPPORTED
        reason = "Linked evidence does not cover enough of the claim."
    return ClaimEvidenceJudgment(
        claim_id=claim["claim_id"],
        verdict=verdict,
        supported_evidence_ids=(evidence_ids if verdict == ClaimStatus.SUPPORTED else []),
        reason=reason,
    )


def _sanitize_judgments(
    claims: list[dict],
    evidence: list[dict],
    judgments: list[ClaimEvidenceJudgment],
) -> list[ClaimEvidenceJudgment]:
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    judgments_by_id = {item.claim_id: item for item in judgments}
    output = []
    for claim in claims:
        linked_ids = set(claim.get("evidence_ids", [])) & set(evidence_by_id)
        judgment = judgments_by_id.get(claim["claim_id"])
        if judgment is None:
            output.append(deterministic_claim_judgment(claim, evidence_by_id))
            continue
        supported_ids = sorted(set(judgment.supported_evidence_ids) & linked_ids)
        verdict = judgment.verdict
        if verdict == ClaimStatus.SUPPORTED and not supported_ids:
            verdict = ClaimStatus.NOT_SUPPORTED
        if verdict not in {
            ClaimStatus.SUPPORTED,
            ClaimStatus.PARTIALLY_SUPPORTED,
            ClaimStatus.CONTRADICTED,
            ClaimStatus.NOT_SUPPORTED,
        }:
            verdict = ClaimStatus.NOT_SUPPORTED
        output.append(
            ClaimEvidenceJudgment(
                claim_id=claim["claim_id"],
                verdict=verdict,
                supported_evidence_ids=(supported_ids if verdict == ClaimStatus.SUPPORTED else []),
                reason=judgment.reason,
            )
        )
    return output


async def validate_and_persist_claims(
    model: BaseChatModel,
    *,
    store: RuntimeStore,
    native_structured_output: bool,
    retries: int,
    config: RunnableConfig | None = None,
) -> ClaimValidationAudit:
    """Batch-validate claims and persist only semantically admissible links."""
    claims = store.list_claims()
    evidence = store.list_evidence()
    if not claims:
        return ClaimValidationAudit(total_claims=0, supported_rate=None)
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    compact_evidence = {
        evidence_id: {
            "source_title": item.get("source_title", ""),
            "excerpt": item.get("excerpt", "")[:900],
        }
        for evidence_id, item in evidence_by_id.items()
    }
    compact_claims = [
        {
            "claim_id": item["claim_id"],
            "statement": item["statement"],
            "linked_evidence_ids": [
                evidence_id
                for evidence_id in item.get("evidence_ids", [])
                if evidence_id in evidence_by_id
            ],
        }
        for item in claims
    ]
    prompt = (
        "Judge whether each claim is entailed by only its linked evidence excerpts. "
        "Use exactly one verdict: SUPPORTED when the full claim follows directly; "
        "PARTIALLY_SUPPORTED when only part follows; CONTRADICTED when the evidence "
        "opposes it; NOT_SUPPORTED when it does not follow. For SUPPORTED, return only "
        "linked evidence IDs that directly entail the complete claim. Be strict about "
        "numbers, comparisons, causality, and scope. Judge every claim.\n\n"
        f"Claims:\n{json.dumps(compact_claims, ensure_ascii=False)}\n\n"
        f"Evidence:\n{json.dumps(compact_evidence, ensure_ascii=False)}"
    )
    fallback_used = False
    try:
        response = await ainvoke_structured_with_fallback(
            model,
            ClaimEvidenceValidation,
            [HumanMessage(content=prompt)],
            retries=retries,
            native_structured_output=native_structured_output,
            config=config,
            node="claim_evidence_validation",
        )
        judgments = _sanitize_judgments(claims, evidence, response.judgments)
    except BudgetExceeded:
        raise
    except Exception as error:
        fallback_used = True
        judgments = [
            deterministic_claim_judgment(claim, evidence_by_id) for claim in claims
        ]
        record_event(
            config,
            "claim_evidence_validation_fallback",
            "claim_evidence_validation",
            data={"error_type": type(error).__name__, "claims": len(claims)},
        )

    claims_by_id = {item["claim_id"]: item for item in claims}
    counts: Counter[str] = Counter()
    for judgment in judgments:
        original = claims_by_id[judgment.claim_id]
        persisted_evidence_ids = (
            judgment.supported_evidence_ids
            if judgment.verdict == ClaimStatus.SUPPORTED
            else [
                evidence_id
                for evidence_id in original.get("evidence_ids", [])
                if evidence_id in evidence_by_id
            ]
        )
        store.save_claim(
            ResearchClaim(
                claim_id=judgment.claim_id,
                statement=original["statement"],
                evidence_ids=persisted_evidence_ids,
                status=judgment.verdict,
            )
        )
        counts[judgment.verdict.value] += 1
    supported = counts[ClaimStatus.SUPPORTED.value]
    audit = ClaimValidationAudit(
        total_claims=len(claims),
        supported=supported,
        partially_supported=counts[ClaimStatus.PARTIALLY_SUPPORTED.value],
        contradicted=counts[ClaimStatus.CONTRADICTED.value],
        not_supported=counts[ClaimStatus.NOT_SUPPORTED.value],
        fallback_used=fallback_used,
        supported_rate=round(supported / len(claims), 4),
    )
    record_event(
        config,
        "claim_evidence_validation_finished",
        "claim_evidence_validation",
        data=audit.model_dump(),
    )
    return audit
