"""Tests for evidence-linked claim extraction."""

import asyncio

from langchain_core.messages import AIMessage

from open_deep_research.claim_extraction import extract_and_persist_claims
from open_deep_research.evidence import Document, Evidence
from open_deep_research.runtime_store import RuntimeStore


class FakeClaimModel:
    async def ainvoke(self, _messages):
        return AIMessage(
            content=(
                '{"claims":['
                '{"statement":"MCP uses JSON-RPC.","evidence_ids":["ev-1"]},'
                '{"statement":"Unsupported claim.","evidence_ids":["unknown"]}'
                "]}"
            )
        )


class SlowClaimModel:
    async def ainvoke(self, _messages):
        await asyncio.sleep(1)


def test_claim_extraction_links_only_known_evidence(tmp_path):
    store = RuntimeStore(tmp_path / "runtime.db")
    store.save_document(
        Document(
            document_id="doc-1",
            url="https://example.com",
            content_hash="sha256:1",
            retrieved_at="2026-10-09T00:00:00+00:00",
        )
    )
    store.save_evidence(
        Evidence(
            evidence_id="ev-1",
            task_id="task-1",
            document_id="doc-1",
            excerpt="MCP messages follow JSON-RPC 2.0.",
            claim_text="MCP uses JSON-RPC.",
            confidence=0.9,
        )
    )

    claims = asyncio.run(
        extract_and_persist_claims(
            FakeClaimModel(),
            task_id="task-1",
            store=store,
            native_structured_output=False,
            retries=1,
        )
    )

    assert len(claims) == 1
    assert claims[0].evidence_ids == ["ev-1"]
    assert store.evidence_counts()["claims"] == 1


def test_claim_extraction_timeout_uses_evidence_fallback(tmp_path):
    store = RuntimeStore(tmp_path / "runtime.db")
    store.save_document(
        Document(
            document_id="doc-1",
            url="https://example.com",
            content_hash="sha256:1",
            retrieved_at="2026-10-09T00:00:00+00:00",
        )
    )
    store.save_evidence(
        Evidence(
            evidence_id="ev-1",
            task_id="task-1",
            document_id="doc-1",
            excerpt="MCP messages follow JSON-RPC 2.0.",
            claim_text="MCP uses JSON-RPC.",
            confidence=0.9,
        )
    )

    claims = asyncio.run(
        extract_and_persist_claims(
            SlowClaimModel(),
            task_id="task-1",
            store=store,
            native_structured_output=False,
            retries=1,
            timeout_seconds=0.01,
            config={"configurable": {"observability_enabled": False}},
        )
    )

    assert [claim.statement for claim in claims] == ["MCP uses JSON-RPC."]
    assert claims[0].evidence_ids == ["ev-1"]
