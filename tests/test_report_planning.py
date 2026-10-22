"""Tests for claim-first planning and deterministic citation rendering."""

from open_deep_research.report_planning import (
    PlannedSection,
    ReportPlan,
    deterministic_report_from_plan,
    deterministic_report_plan,
    plan_context,
    render_grounded_report,
)


def records():
    evidence = [
        {
            "evidence_id": "ev-1",
            "source_url": "local://docs/paper.md",
            "source_title": "Paper",
            "excerpt": "The method improves exact-match accuracy.",
        },
        {
            "evidence_id": "ev-2",
            "source_url": "https://example.org/result",
            "source_title": "Result",
            "excerpt": "The evaluation contains 32 questions.",
        },
    ]
    claims = [
        {
            "claim_id": "claim-1",
            "statement": "The method improves exact-match accuracy.",
            "evidence_ids": ["ev-1"],
            "status": "SUPPORTED",
        },
        {
            "claim_id": "claim-2",
            "statement": "The evaluation contains 32 questions.",
            "evidence_ids": ["ev-2"],
            "status": "SUPPORTED",
        },
    ]
    return claims, evidence


def test_deterministic_plan_contains_only_supported_claims():
    claims, evidence = records()
    claims.append(
        {
            "claim_id": "claim-bad",
            "statement": "Unsupported",
            "evidence_ids": ["missing"],
            "status": "SUPPORTED",
        }
    )

    plan = deterministic_report_plan("Compare the methods", claims, evidence)

    assert plan.sections[0].claim_ids == ["claim-1", "claim-2"]
    assert plan.gaps == []


def test_plan_rejects_partial_and_unverified_claims():
    claims, evidence = records()
    claims[0]["status"] = "PARTIALLY_SUPPORTED"
    claims[1]["status"] = "UNVERIFIED"

    plan = deterministic_report_plan("Compare the methods", claims, evidence)

    assert plan.sections[0].claim_ids == []
    assert plan.gaps == ["Compare the methods"]


def test_plan_context_preserves_claim_evidence_links():
    claims, evidence = records()
    plan = ReportPlan(
        title="Evaluation",
        sections=[PlannedSection(heading="Results", claim_ids=["claim-2"])],
    )

    context = plan_context(plan, claims, evidence)

    assert "CLAIM claim-2" in context
    assert "EVIDENCE ev-2" in context
    assert "https://example.org/result" in context
    assert "claim-1" not in context


def test_deterministic_report_covers_all_planned_evidence():
    claims, evidence = records()
    plan = ReportPlan(
        title="Evaluation",
        sections=[
            PlannedSection(
                heading="Results", claim_ids=["claim-1", "claim-2"]
            )
        ],
        gaps=["A remaining question"],
    )

    draft = deterministic_report_from_plan(plan, claims)
    report, audit = render_grounded_report(
        draft, evidence, required_evidence_ids={"ev-1", "ev-2"}
    )

    assert audit.citation_coverage == 1.0
    assert audit.source_precision == 1.0
    assert "A remaining question" in report


def test_renderer_uses_exact_urls_and_removes_model_urls():
    _, evidence = records()
    draft = (
        "# Report\n\nThe benchmark has 32 questions [EVIDENCE:ev-2]. "
        "A fabricated [link](https://invented.example/x) must not survive. "
        "The method improved [EVIDENCE:ev-1]. Unknown [EVIDENCE:not-real].\n\n"
        "## Sources\nhttps://invented.example/x"
    )

    report, audit = render_grounded_report(
        draft, evidence, required_evidence_ids={"ev-1", "ev-2"}
    )

    assert "[1]" in report and "[2]" in report
    assert "https://invented.example/x" not in report
    assert "https://example.org/result" in report
    assert "local://docs/paper.md" in report
    assert audit.unknown_evidence_ids == ["not-real"]
    assert audit.source_precision == 1.0
    assert audit.citation_coverage == 1.0
