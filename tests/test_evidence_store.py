"""Tests for structured evidence persistence and source deduplication."""

from open_deep_research.evidence import (
    ClaimStatus,
    Document,
    Evidence,
    EvidenceConflict,
    ResearchClaim,
    audit_report_sources,
)
from open_deep_research.runtime_store import RuntimeStore


def test_evidence_graph_is_persisted(tmp_path):
    store = RuntimeStore(tmp_path / "evidence.db")
    document = Document(
        document_id="doc-1",
        url="https://example.com/source",
        title="Source",
        content_hash="sha256:abc",
        retrieved_at="2026-10-09T00:00:00+00:00",
    )
    evidence_a = Evidence(
        evidence_id="ev-1",
        task_id="task-1",
        document_id="doc-1",
        excerpt="Documented statement",
        claim_text="The protocol uses JSON-RPC.",
        confidence=0.9,
    )
    evidence_b = evidence_a.model_copy(
        update={"evidence_id": "ev-2", "excerpt": "Contradictory statement"}
    )
    claim = ResearchClaim(
        claim_id="claim-1",
        statement="The protocol uses JSON-RPC.",
        evidence_ids=["ev-1", "ev-2"],
        status=ClaimStatus.CONFLICTED,
    )
    conflict = EvidenceConflict(
        conflict_id="conflict-1",
        claim_id="claim-1",
        evidence_id_a="ev-1",
        evidence_id_b="ev-2",
        reason="Sources disagree on protocol version",
    )

    assert store.save_document(document)
    assert not store.save_document(document.model_copy(update={"document_id": "duplicate"}))
    store.save_evidence(evidence_a)
    store.save_evidence(evidence_b)
    store.save_claim(claim)
    store.save_conflict(conflict)

    assert store.evidence_counts() == {
        "documents": 1,
        "evidence": 2,
        "claims": 1,
        "conflicts": 1,
    }


def test_report_source_audit_rejects_urls_outside_persisted_evidence():
    result = audit_report_sources(
        "Supported https://example.test/a and invented https://example.test/b.",
        {"https://example.test/a"},
    )

    assert result.report_sources == 2
    assert result.verified_sources == 1
    assert result.source_precision == 0.5
    assert result.unsupported_sources == ["https://example.test/b"]
