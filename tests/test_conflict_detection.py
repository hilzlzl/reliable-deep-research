"""Tests for deterministic contradiction candidate detection."""

from open_deep_research.conflict_detection import detect_and_persist_conflicts
from open_deep_research.evidence import Document, Evidence, ResearchClaim
from open_deep_research.runtime_store import RuntimeStore


def test_opposite_polarity_evidence_creates_conflict(tmp_path):
    store = RuntimeStore(tmp_path / "runtime.db")
    store.save_document(Document(document_id="doc", url="https://example.com", content_hash="h", retrieved_at="now"))
    store.save_evidence(Evidence(evidence_id="yes", task_id="t", document_id="doc", excerpt="The protocol supports resumable streaming", claim_text="streaming"))
    store.save_evidence(Evidence(evidence_id="no", task_id="t", document_id="doc", excerpt="The protocol does not support resumable streaming", claim_text="streaming"))
    store.save_claim(ResearchClaim(claim_id="claim", statement="Streaming support", evidence_ids=["yes", "no"]))

    conflicts = detect_and_persist_conflicts(store, similarity_threshold=0.5)

    assert len(conflicts) == 1
    assert store.evidence_counts()["conflicts"] == 1


def test_unrelated_evidence_is_not_marked_as_conflict(tmp_path):
    store = RuntimeStore(tmp_path / "runtime.db")
    store.save_document(Document(document_id="doc", url="https://example.com", content_hash="h", retrieved_at="now"))
    store.save_evidence(Evidence(evidence_id="a", task_id="t", document_id="doc", excerpt="The protocol supports streaming", claim_text="streaming"))
    store.save_evidence(Evidence(evidence_id="b", task_id="t", document_id="doc", excerpt="The database has no vector index", claim_text="database"))
    store.save_claim(ResearchClaim(claim_id="claim", statement="Capability", evidence_ids=["a", "b"]))

    assert detect_and_persist_conflicts(store) == []
