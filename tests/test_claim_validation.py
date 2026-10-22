"""Tests for semantic Claim-Evidence admission control."""

import asyncio
import json

from langchain_core.messages import AIMessage

from open_deep_research.claim_validation import (
    deterministic_claim_judgment,
    validate_and_persist_claims,
)
from open_deep_research.evidence import ClaimStatus, Document, Evidence, ResearchClaim
from open_deep_research.report_planning import deterministic_report_plan
from open_deep_research.runtime_store import RuntimeStore


class ValidationModel:
    async def ainvoke(self, _messages):
        return AIMessage(
            content=json.dumps(
                {
                    "judgments": [
                        {
                            "claim_id": "claim-supported",
                            "verdict": "SUPPORTED",
                            "supported_evidence_ids": ["ev-1", "invented-id"],
                            "reason": "The excerpt states the complete claim.",
                        },
                        {
                            "claim_id": "claim-partial",
                            "verdict": "PARTIALLY_SUPPORTED",
                            "supported_evidence_ids": ["ev-1"],
                            "reason": "The speed comparison is absent.",
                        },
                    ]
                }
            )
        )


def populated_store(tmp_path):
    store = RuntimeStore(tmp_path / "runtime.db")
    store.save_document(
        Document(
            document_id="doc-1",
            url="local://papers/deepcad",
            title="DeepCAD",
            content_hash="sha256:deepcad",
            retrieved_at="2026-10-11T00:00:00+00:00",
        )
    )
    store.save_evidence(
        Evidence(
            evidence_id="ev-1",
            task_id="task-1",
            document_id="doc-1",
            excerpt="DeepCAD represents a model as a sequence of CAD operations.",
            claim_text="DeepCAD uses a sequence of CAD operations.",
            confidence=0.9,
        )
    )
    store.save_claim(
        ResearchClaim(
            claim_id="claim-supported",
            statement="DeepCAD represents a model as a sequence of CAD operations.",
            evidence_ids=["ev-1"],
            status=ClaimStatus.SUPPORTED,
        )
    )
    store.save_claim(
        ResearchClaim(
            claim_id="claim-partial",
            statement="DeepCAD is faster and uses a sequence of CAD operations.",
            evidence_ids=["ev-1"],
            status=ClaimStatus.SUPPORTED,
        )
    )
    return store


def test_semantic_validation_only_admits_fully_supported_claims(tmp_path):
    store = populated_store(tmp_path)

    audit = asyncio.run(
        validate_and_persist_claims(
            ValidationModel(),
            store=store,
            native_structured_output=False,
            retries=1,
        )
    )

    claims = {item["claim_id"]: item for item in store.list_claims()}
    assert audit.supported == 1
    assert audit.partially_supported == 1
    assert claims["claim-supported"]["evidence_ids"] == ["ev-1"]
    assert claims["claim-partial"]["status"] == "PARTIALLY_SUPPORTED"
    assert claims["claim-partial"]["evidence_ids"] == ["ev-1"]
    plan = deterministic_report_plan("Compare CAD methods", store.list_claims(), store.list_evidence())
    assert plan.sections[0].claim_ids == ["claim-supported"]


def test_deterministic_validation_is_conservative_about_unsupported_numbers():
    judgment = deterministic_claim_judgment(
        {
            "claim_id": "claim-1",
            "statement": "The benchmark contains 100 questions.",
            "evidence_ids": ["ev-1"],
        },
        {
            "ev-1": {
                "evidence_id": "ev-1",
                "claim_text": "The benchmark contains 32 questions.",
                "excerpt": "The benchmark contains 32 questions.",
            }
        },
    )

    assert judgment.verdict == ClaimStatus.NOT_SUPPORTED
    assert judgment.supported_evidence_ids == []
