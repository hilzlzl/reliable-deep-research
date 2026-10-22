"""Tests for budgeted evidence compression metrics."""

import pytest

from open_deep_research.context_compression import compress_evidence


def test_compression_prioritizes_required_and_relevant_evidence():
    evidence = [
        {"evidence_id": "required", "claim_text": "MCP security", "excerpt": "OAuth security requirements", "confidence": 0.9},
        {"evidence_id": "relevant", "claim_text": "MCP tools", "excerpt": "Tool invocation protocol", "confidence": 0.8},
        {"evidence_id": "other", "claim_text": "CAD geometry", "excerpt": "B-rep topology", "confidence": 1.0},
    ]

    result = compress_evidence(
        evidence,
        query="MCP protocol security and tools",
        max_characters=70,
        required_evidence_ids={"required"},
    )

    assert result.selected_evidence_ids[0] == "required"
    assert "other" not in result.selected_evidence_ids
    assert result.metrics.required_evidence_retention == 1.0
    assert result.metrics.compression_ratio < 1.0


def test_compression_rejects_unknown_required_evidence():
    with pytest.raises(ValueError, match="Unknown required evidence ids"):
        compress_evidence(
            [{"evidence_id": "known", "claim_text": "x", "excerpt": "y"}],
            query="x",
            max_characters=20,
            required_evidence_ids={"missing"},
        )
