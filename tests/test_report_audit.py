"""Tests for sentence-level semantic report auditing."""

import asyncio
import json

from langchain_core.messages import AIMessage

from open_deep_research.report_audit import audit_report_assertions


class AuditModel:
    def __init__(self, payload, verification_payload=None):
        self.payloads = [payload]
        if verification_payload is not None:
            self.payloads.append(verification_payload)
        self.calls = 0

    async def ainvoke(self, _messages):
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return AIMessage(content=json.dumps(payload))


def test_partial_assertion_is_repaired_with_known_evidence_marker():
    sentence = "Diffusion models are always valid [EVIDENCE:ev-1]."
    replacement = "One diffusion model improved validity on one benchmark [EVIDENCE:ev-1]."
    model = AuditModel(
        {
            "assertions": [
                {
                    "assertion_id": "model-chosen-id",
                    "sentence": sentence,
                    "evidence_ids": ["ev-1"],
                    "verdict": "PARTIALLY_SUPPORTED",
                    "reason": "The evidence covers one benchmark, not all settings.",
                    "repair_action": "MODIFY",
                    "replacement": replacement,
                }
            ]
        },
        {
            "assertions": [
                {
                    "assertion_id": "verification",
                    "sentence": replacement,
                    "evidence_ids": ["ev-1"],
                    "verdict": "SUPPORTED",
                    "reason": "The weaker sentence follows from the excerpt.",
                    "repair_action": "KEEP",
                    "replacement": "",
                }
            ]
        },
    )

    outcome = asyncio.run(
        audit_report_assertions(
            model,
            report=f"# Report\n\n{sentence}",
            evidence=[
                {
                    "evidence_id": "ev-1",
                    "source_title": "Paper",
                    "excerpt": "The method improved validity on Benchmark A.",
                }
            ],
            apply_repairs=True,
            native_structured_output=False,
            retries=1,
        )
    )

    assert sentence not in outcome.report
    assert replacement in outcome.report
    assert outcome.supported == 1
    assert outcome.partially_supported == 0
    assert outcome.repairs_applied == 1
    assert outcome.repairs_verified == 1
    assert outcome.repairs_rejected == 0


def test_modify_is_rejected_when_second_entailment_pass_does_not_support_it():
    sentence = "The method is always valid [EVIDENCE:ev-1]."
    replacement = "The method improved validity [EVIDENCE:ev-1]."
    model = AuditModel(
        {"assertions": [{
            "assertion_id": "first",
            "sentence": sentence,
            "evidence_ids": ["ev-1"],
            "verdict": "PARTIALLY_SUPPORTED",
            "reason": "The universal statement is too strong.",
            "repair_action": "MODIFY",
            "replacement": replacement,
        }]},
        {"assertions": [{
            "assertion_id": "second",
            "sentence": replacement,
            "evidence_ids": ["ev-1"],
            "verdict": "NOT_SUPPORTED",
            "reason": "The evidence does not establish improvement.",
            "repair_action": "KEEP",
            "replacement": "",
        }]},
    )
    outcome = asyncio.run(audit_report_assertions(
        model,
        report=sentence,
        evidence=[{"evidence_id": "ev-1", "source_title": "Paper", "excerpt": "Validity was measured."}],
        apply_repairs=True,
        native_structured_output=False,
        retries=1,
    ))
    assert sentence not in outcome.report
    assert replacement not in outcome.report
    assert outcome.repairs_applied == 0
    assert outcome.repairs_verified == 0
    assert outcome.repairs_rejected == 1
    assert outcome.assertions == []


def test_supported_verdict_without_adjacent_citation_is_rejected():
    sentence = "The method is universally superior."
    model = AuditModel(
        {
            "assertions": [
                {
                    "assertion_id": "a",
                    "sentence": sentence,
                    "evidence_ids": ["ev-1"],
                    "verdict": "SUPPORTED",
                    "reason": "Claimed support.",
                    "repair_action": "KEEP",
                }
            ]
        }
    )

    outcome = asyncio.run(
        audit_report_assertions(
            model,
            report=sentence,
            evidence=[
                {
                    "evidence_id": "ev-1",
                    "source_title": "Paper",
                    "excerpt": "Limited result.",
                }
            ],
            apply_repairs=True,
            native_structured_output=False,
            retries=1,
        )
    )

    assert outcome.not_supported == 1
    assert outcome.assertions[0].repair_action.value == "SEARCH"
    assert outcome.repairs_applied == 0


def test_trailing_marker_is_admitted_when_model_extracts_sentence_without_it():
    sentence = "ReAct alternates reasoning and action steps."
    model = AuditModel({"assertions": [{
        "assertion_id": "a",
        "sentence": sentence,
        "evidence_ids": [],
        "verdict": "SUPPORTED",
        "reason": "The evidence states the mechanism.",
        "repair_action": "KEEP",
        "replacement": "",
    }]})
    outcome = asyncio.run(audit_report_assertions(
        model,
        report=f"- {sentence} [EVIDENCE:ev-1]",
        evidence=[{
            "evidence_id": "ev-1",
            "source_title": "ReAct",
            "excerpt": "ReAct interleaves reasoning traces and actions.",
        }],
        apply_repairs=False,
        native_structured_output=False,
        retries=1,
    ))
    assert outcome.supported == 1
    assert outcome.cited_assertions == 1
    assert outcome.assertions[0].evidence_ids == ["ev-1"]


def test_marker_on_next_line_of_same_bullet_is_adjacent():
    sentence = "ReAct uses observations to update later actions."
    model = AuditModel({"assertions": [{
        "assertion_id": "a",
        "sentence": sentence,
        "evidence_ids": [],
        "verdict": "SUPPORTED",
        "reason": "Directly supported.",
        "repair_action": "KEEP",
        "replacement": "",
    }]})
    outcome = asyncio.run(audit_report_assertions(
        model,
        report=f"- {sentence}\n  [EVIDENCE:ev-1]",
        evidence=[{
            "evidence_id": "ev-1",
            "source_title": "ReAct",
            "excerpt": "An observation updates the next reasoning and action step.",
        }],
        apply_repairs=False,
        native_structured_output=False,
        retries=1,
    ))
    assert outcome.supported == 1
    assert outcome.assertions[0].evidence_ids == ["ev-1"]


def test_sources_and_evidence_gaps_are_not_sent_as_assertions():
    sentence = "The method preserves constraints [EVIDENCE:ev-1]."
    model = AuditModel(
        {
            "assertions": [
                {
                    "assertion_id": "a",
                    "sentence": sentence,
                    "evidence_ids": ["ev-1"],
                    "verdict": "SUPPORTED",
                    "reason": "The excerpt states this directly.",
                    "repair_action": "KEEP",
                }
            ]
        }
    )
    report = (
        f"# Report\n\n{sentence}\n\n## Evidence gaps\n\n"
        "- No universal guarantee was identified.\n\n## Sources\n\n"
        "[1] Paper: https://example.com/paper"
    )
    outcome = asyncio.run(
        audit_report_assertions(
            model,
            report=report,
            evidence=[
                {
                    "evidence_id": "ev-1",
                    "source_title": "Paper",
                    "excerpt": "The method preserves constraints.",
                }
            ],
            apply_repairs=False,
            native_structured_output=False,
            retries=1,
        )
    )

    assert outcome.total_assertions == 1
    assert outcome.cited_assertions == 1
    assert outcome.citation_coverage == 1.0
