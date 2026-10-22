"""Claim-first report planning and deterministic evidence citation rendering."""

from __future__ import annotations

import json
import re
from collections import OrderedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.utils import ainvoke_structured_with_fallback


class PlannedSection(BaseModel):
    heading: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)


class ReportPlan(BaseModel):
    title: str = Field(min_length=1)
    sections: list[PlannedSection] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class RenderAudit(BaseModel):
    cited_evidence_ids: list[str] = Field(default_factory=list)
    unknown_evidence_ids: list[str] = Field(default_factory=list)
    cited_sources: int = 0
    source_precision: float = 1.0
    citation_coverage: float | None = None


def deterministic_report_plan(
    research_brief: str, claims: list[dict], evidence: list[dict]
) -> ReportPlan:
    """Build a valid fallback plan without another model call."""
    known_evidence = {item["evidence_id"] for item in evidence}
    supported_ids = [
        claim["claim_id"]
        for claim in claims
        if claim.get("evidence_ids")
        and set(claim["evidence_ids"]).issubset(known_evidence)
        and claim.get("status") == "SUPPORTED"
    ]
    gaps = [] if supported_ids else [research_brief]
    return ReportPlan(
        title=research_brief.strip().splitlines()[0][:160] or "Research Report",
        sections=[PlannedSection(heading="Findings", claim_ids=supported_ids)],
        gaps=gaps,
    )


def _validate_plan(
    plan: ReportPlan, research_brief: str, claims: list[dict], evidence: list[dict]
) -> ReportPlan:
    claims_by_id = {item["claim_id"]: item for item in claims}
    known_evidence = {item["evidence_id"] for item in evidence}
    seen: set[str] = set()
    sections: list[PlannedSection] = []
    for section in plan.sections:
        valid_ids = []
        for claim_id in section.claim_ids:
            claim = claims_by_id.get(claim_id)
            if (
                claim
                and claim_id not in seen
                and claim.get("evidence_ids")
                and set(claim["evidence_ids"]).issubset(known_evidence)
                and claim.get("status") == "SUPPORTED"
            ):
                valid_ids.append(claim_id)
                seen.add(claim_id)
        if valid_ids:
            sections.append(PlannedSection(heading=section.heading, claim_ids=valid_ids))
    if not sections:
        return deterministic_report_plan(research_brief, claims, evidence)
    return ReportPlan(title=plan.title, sections=sections, gaps=plan.gaps[:3])


async def create_report_plan(
    model: BaseChatModel,
    *,
    research_brief: str,
    claims: list[dict],
    evidence: list[dict],
    native_structured_output: bool,
    retries: int,
    config: RunnableConfig | None = None,
) -> ReportPlan:
    """Ask the model to organize only persisted, evidence-linked claims."""
    known_evidence = {item["evidence_id"] for item in evidence}
    compact_claims = [
        {
            "claim_id": item["claim_id"],
            "statement": item["statement"],
            "evidence_ids": item["evidence_ids"],
        }
        for item in claims
        if item.get("evidence_ids")
        and set(item["evidence_ids"]).issubset(known_evidence)
        and item.get("status") == "SUPPORTED"
    ]
    if not compact_claims:
        return deterministic_report_plan(research_brief, claims, evidence)
    prompt = (
        "Create a concise report outline for the research brief using only the supplied "
        "claims. Put claim IDs under the most relevant section. Do not invent or alter IDs. "
        "List at most three important unanswered questions in gaps; use an empty list when "
        "the claims adequately answer the brief.\n\n"
        f"Research brief:\n{research_brief}\n\n"
        f"Claims:\n{json.dumps(compact_claims, ensure_ascii=False)}"
    )
    plan = await ainvoke_structured_with_fallback(
        model,
        ReportPlan,
        [HumanMessage(content=prompt)],
        retries=retries,
        native_structured_output=native_structured_output,
        config=config,
        node="report_planning",
    )
    return _validate_plan(plan, research_brief, claims, evidence)


def plan_context(plan: ReportPlan, claims: list[dict], evidence: list[dict]) -> str:
    """Serialize the validated plan and evidence cards for the report writer."""
    claims_by_id = {item["claim_id"]: item for item in claims}
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    lines = [f"# {plan.title}"]
    for section in plan.sections:
        lines.append(f"\n## {section.heading}")
        for claim_id in section.claim_ids:
            claim = claims_by_id[claim_id]
            lines.append(f"\nCLAIM {claim_id}: {claim['statement']}")
            for evidence_id in claim["evidence_ids"]:
                item = evidence_by_id[evidence_id]
                lines.extend(
                    [
                        f"EVIDENCE {evidence_id}",
                        f"Source title: {item['source_title']}",
                        f"Exact URL: {item['source_url']}",
                        f"Quote: {item['excerpt']}",
                    ]
                )
    return "\n".join(lines)


def deterministic_report_from_plan(plan: ReportPlan, claims: list[dict]) -> str:
    """Render a readable claim-only draft when a model ignores evidence markers."""
    claims_by_id = {item["claim_id"]: item for item in claims}
    lines = [f"# {plan.title}"]
    for section in plan.sections:
        lines.append(f"\n## {section.heading}")
        for claim_id in section.claim_ids:
            claim = claims_by_id[claim_id]
            markers = " ".join(
                f"[EVIDENCE:{evidence_id}]"
                for evidence_id in claim["evidence_ids"]
            )
            lines.append(f"\n- {claim['statement']} {markers}".rstrip())
    if plan.gaps:
        lines.extend(
            [
                "\n## Evidence gaps",
                "",
                *[f"- {gap}" for gap in plan.gaps],
            ]
        )
    return "\n".join(lines)


_MARKER_RE = re.compile(r"\[EVIDENCE:([^\]]+)\]", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?|local)://[^)]+\)")
_RAW_URL_RE = re.compile(r"<?(?:https?|local)://[^\s)>\]]+>?")
_SOURCES_HEADING_RE = re.compile(r"(?im)^#{1,3}\s+(?:sources|references)\s*$")


def render_grounded_report(
    draft: str,
    evidence: list[dict],
    required_evidence_ids: set[str] | None = None,
) -> tuple[str, RenderAudit]:
    """Replace evidence markers with citations and append exact persisted URLs."""
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    source_numbers: OrderedDict[str, int] = OrderedDict()
    cited_ids: list[str] = []
    unknown_ids: list[str] = []

    sources_heading = _SOURCES_HEADING_RE.search(draft)
    body = draft[: sources_heading.start()] if sources_heading else draft
    body = _MARKDOWN_LINK_RE.sub(r"\1", body)
    body = _RAW_URL_RE.sub("", body)

    def replace_marker(match: re.Match[str]) -> str:
        evidence_id = match.group(1).strip()
        item = evidence_by_id.get(evidence_id)
        if item is None:
            unknown_ids.append(evidence_id)
            return ""
        if evidence_id not in cited_ids:
            cited_ids.append(evidence_id)
        url = item["source_url"]
        if url not in source_numbers:
            source_numbers[url] = len(source_numbers) + 1
        return f"[{source_numbers[url]}]"

    body = _MARKER_RE.sub(replace_marker, body).rstrip()
    if source_numbers:
        by_url = {item["source_url"]: item for item in evidence}
        source_lines = ["", "## Sources", ""]
        for url, number in source_numbers.items():
            title = by_url[url].get("source_title") or "Source"
            source_lines.append(f"[{number}] {title}: {url}")
        body += "\n" + "\n".join(source_lines)
    required = required_evidence_ids or set()
    return body.strip(), RenderAudit(
        cited_evidence_ids=cited_ids,
        unknown_evidence_ids=sorted(set(unknown_ids)),
        cited_sources=len(source_numbers),
        source_precision=1.0,
        citation_coverage=(len(set(cited_ids) & required) / len(required) if required else None),
    )
