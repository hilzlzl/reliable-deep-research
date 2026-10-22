"""Task-level claim extraction backed by persisted source evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.evidence import (
    ClaimExtraction,
    ClaimStatus,
    ExtractedClaim,
    ResearchClaim,
)
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.run_budget import BudgetExceeded
from open_deep_research.observability import record_event
from open_deep_research.utils import ainvoke_structured_with_fallback


def persist_evidence_claims(
    evidence: list[dict], store: RuntimeStore
) -> list[ResearchClaim]:
    """Create deterministic evidence-linked claims without a model call."""
    claims = []
    for item in evidence:
        statement = item["claim_text"].strip()
        if not statement:
            continue
        claim = ResearchClaim(
            claim_id="claim:" + hashlib.sha256(statement.encode("utf-8")).hexdigest()[:20],
            statement=statement,
            evidence_ids=[item["evidence_id"]],
            status=ClaimStatus.SUPPORTED,
        )
        store.save_claim(claim)
        claims.append(claim)
    return claims


async def extract_and_persist_claims(
    model: BaseChatModel,
    *,
    task_id: str,
    store: RuntimeStore,
    native_structured_output: bool,
    retries: int,
    timeout_seconds: float = 30.0,
    config: RunnableConfig | None = None,
) -> list[ResearchClaim]:
    """Extract claims once per task and link them only to known evidence IDs."""
    evidence = store.list_evidence(task_id)
    if not evidence:
        return []
    compact_evidence = [
        {
            "evidence_id": item["evidence_id"],
            "excerpt": item["excerpt"][:1200],
            "candidate_claim": item["claim_text"][:500],
        }
        for item in evidence
    ]
    prompt = (
        "Extract concise factual claims from the evidence below. Every claim must "
        "reference one or more evidence_id values from the input. Do not invent IDs.\n\n"
        + json.dumps(compact_evidence, ensure_ascii=False)
    )
    known_ids = {item["evidence_id"] for item in evidence}
    try:
        extraction = await asyncio.wait_for(
            ainvoke_structured_with_fallback(
                model,
                ClaimExtraction,
                [HumanMessage(content=prompt)],
                retries=retries,
                native_structured_output=native_structured_output,
                config=config,
                node="claim_extraction",
            ),
            timeout=timeout_seconds,
        )
        extracted_claims = extraction.claims
    except BudgetExceeded:
        raise
    except Exception as error:
        record_event(
            config,
            "claim_extraction_fallback",
            "claim_extraction",
            data={"error_type": type(error).__name__, "evidence_count": len(evidence)},
        )
        return persist_evidence_claims(evidence, store)
    claims = []
    for extracted in extracted_claims:
        linked_ids = sorted(set(extracted.evidence_ids).intersection(known_ids))
        if not linked_ids:
            continue
        claim_id = "claim:" + hashlib.sha256(
            extracted.statement.strip().encode("utf-8")
        ).hexdigest()[:20]
        claim = ResearchClaim(
            claim_id=claim_id,
            statement=extracted.statement.strip(),
            evidence_ids=linked_ids,
            status=ClaimStatus.SUPPORTED,
        )
        store.save_claim(claim)
        claims.append(claim)
    return claims
