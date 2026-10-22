"""Tests for bounded Red-Blue review and convergence guards."""

import asyncio

from langchain_core.messages import AIMessage

from open_deep_research.report_review import run_report_review


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def ainvoke(self, _messages):
        return AIMessage(content=next(self.responses))


def test_review_repairs_then_reaches_threshold():
    model = SequenceModel([
        '{"score":0.5,"issues":[{"issue_type":"CITATION","severity":4,"description":"Unsupported","report_excerpt":"claim","evidence_ids":["ev-1"]}]}',
        '{"actions":[{"operation":"MODIFY","target_excerpt":"claim","replacement":"Supported report [ev-1]","reason":"Add support","evidence_ids":["ev-1"]}]}',
        '{"score":0.9,"issues":[]}',
    ])

    outcome = asyncio.run(run_report_review(
        model,
        report="claim",
        evidence_context="[ev-1] evidence",
        max_rounds=2,
        quality_threshold=0.95,
        native_structured_output=False,
    ))

    assert outcome.report == "Supported report [ev-1]"
    assert outcome.stop_reason == "quality_threshold"
    assert len(outcome.rounds) == 2


def test_review_stops_when_repair_oscillates():
    model = SequenceModel([
        '{"score":0.4,"issues":[{"issue_type":"LOGIC","severity":3,"description":"Problem"}]}',
        '{"actions":[{"operation":"VERIFY","reason":"No valid change"}]}',
    ])

    outcome = asyncio.run(run_report_review(
        model,
        report="original",
        evidence_context="evidence",
        max_rounds=2,
        native_structured_output=False,
    ))

    assert outcome.stop_reason == "oscillation_detected"
    assert outcome.report == "original"


def test_review_rejects_unknown_evidence_id_and_keeps_report_unchanged():
    model = SequenceModel([
        '{"score":0.4,"issues":[{"issue_type":"FACTUALITY","severity":5,"description":"Unsupported"}]}',
        '{"actions":[{"operation":"MODIFY","target_excerpt":"claim","replacement":"invented","reason":"fix","evidence_ids":["ev-fake"]}]}',
    ])

    outcome = asyncio.run(run_report_review(
        model,
        report="claim",
        evidence_context="[ev-real] supported source",
        max_rounds=2,
        native_structured_output=False,
    ))

    assert outcome.report == "claim"
    assert outcome.stop_reason == "oscillation_detected"
    assert outcome.rounds[0].actions == []
    assert "Unknown evidence IDs" in outcome.rounds[0].rejected_actions[0].rejection_reason


def test_review_rejects_modify_without_evidence_reference():
    model = SequenceModel([
        '{"score":0.4,"issues":[{"issue_type":"CITATION","severity":4,"description":"Missing"}]}',
        '{"actions":[{"operation":"MODIFY","target_excerpt":"claim","replacement":"changed","reason":"fix"}]}',
    ])

    outcome = asyncio.run(run_report_review(
        model,
        report="claim",
        evidence_context="[ev-real] supported source",
        max_rounds=2,
        native_structured_output=False,
    ))

    assert outcome.report == "claim"
    assert "require at least one evidence_id" in outcome.rounds[0].rejected_actions[0].rejection_reason


def test_review_rolls_back_when_repair_score_regresses():
    model = SequenceModel([
        '{"score":0.8,"issues":[{"issue_type":"LOGIC","severity":3,"description":"Problem"}]}',
        '{"actions":[{"operation":"MODIFY","target_excerpt":"claim","replacement":"weaker [ev-1]","reason":"attempted fix","evidence_ids":["ev-1"]}]}',
        '{"score":0.5,"issues":[{"issue_type":"LOGIC","severity":4,"description":"Worse"}]}',
    ])

    outcome = asyncio.run(run_report_review(
        model,
        report="claim",
        evidence_context="[ev-1] evidence",
        max_rounds=2,
        quality_threshold=0.95,
        native_structured_output=False,
    ))

    assert outcome.stop_reason == "insufficient_improvement"
    assert outcome.report == "claim"
    assert outcome.final_score == 0.8
