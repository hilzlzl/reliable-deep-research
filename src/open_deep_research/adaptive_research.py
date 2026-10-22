"""Evidence-aware policy for bounded, incremental research rounds."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import SearchAPI
from open_deep_research.run_budget import BudgetSnapshot
from open_deep_research.utils import ainvoke_structured_with_fallback


class ObjectiveStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTED = "CONFLICTED"
    ABANDONED = "ABANDONED"


class ResearchActionType(str, Enum):
    SEARCH_LOCAL = "SEARCH_LOCAL"
    SEARCH_WEB = "SEARCH_WEB"
    DIVERSIFY_SOURCES = "DIVERSIFY_SOURCES"
    RESOLVE_CONFLICT = "RESOLVE_CONFLICT"
    ESCALATE_MODEL = "ESCALATE_MODEL"
    WRITE_WITH_CAVEAT = "WRITE_WITH_CAVEAT"
    DROP_OBJECTIVE = "DROP_OBJECTIVE"
    STOP = "STOP"


class ResearchObjective(BaseModel):
    objective_id: str
    question: str
    importance: float = Field(ge=0.0, le=1.0)
    status: ObjectiveStatus = ObjectiveStatus.INSUFFICIENT
    linked_claim_ids: list[str] = Field(default_factory=list)
    search_attempts: int = Field(default=0, ge=0)


class ProposedObjective(BaseModel):
    objective_id: str | None = None
    question: str = Field(min_length=1)
    importance: float = Field(ge=0.0, le=1.0)
    linked_claim_ids: list[str] = Field(default_factory=list)


class ResearchObjectiveProposal(BaseModel):
    objectives: list[ProposedObjective] = Field(default_factory=list)


class ObjectiveAssessment(BaseModel):
    objective: ResearchObjective
    semantic_support: float = Field(ge=0.0, le=1.0)
    source_diversity: int = Field(ge=0)
    contradiction_level: float = Field(ge=0.0, le=1.0)
    estimated_search_gain: float = Field(ge=0.0, le=1.0)
    reason: str


class ResearchAction(BaseModel):
    objective_id: str
    action: ResearchActionType
    retrieval_route: str
    model_tier: str
    utility: float
    reason: str
    query: str
    candidate_id: str | None = None
    expected_quality_gain: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_cost: float | None = Field(default=None, ge=0.0, le=1.0)
    success_condition: str | None = None
    policy_source: str = "deterministic_fallback"


class PolicyCandidate(BaseModel):
    """One action considered by the model policy before execution."""

    candidate_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    action: ResearchActionType
    query: str = ""
    expected_quality_gain: float = Field(ge=0.0, le=1.0)
    expected_cost: float = Field(ge=0.0, le=1.0)
    success_condition: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class AgentPolicyProposal(BaseModel):
    """Auditable output contract for one model-policy decision."""

    situation_summary: str = Field(min_length=1)
    candidates: list[PolicyCandidate] = Field(min_length=1, max_length=12)
    selected_candidate_ids: list[str] = Field(default_factory=list)


class ResearchPolicyResult(BaseModel):
    """Validated policy result consumed by the deterministic executor."""

    actions: list[ResearchAction] = Field(default_factory=list)
    candidates: list[PolicyCandidate] = Field(default_factory=list)
    situation_summary: str
    policy_source: str


class PolicyContext(BaseModel):
    search_api: SearchAPI
    local_corpus_available: bool = False
    local_model_available: bool = False
    paid_model_available: bool = False
    remaining_task_slots: int = Field(ge=0)
    max_actions: int = Field(default=1, ge=1)
    min_utility: float = 0.15
    budget_reserve_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    max_search_attempts_per_objective: int = Field(default=2, ge=1)
    budget: BudgetSnapshot | None = None


def _assessment_payload(assessment: ObjectiveAssessment) -> dict[str, Any]:
    return {
        "objective_id": assessment.objective.objective_id,
        "question": assessment.objective.question,
        "importance": assessment.objective.importance,
        "status": assessment.objective.status.value,
        "search_attempts": assessment.objective.search_attempts,
        "semantic_support": assessment.semantic_support,
        "source_diversity": assessment.source_diversity,
        "contradiction_level": assessment.contradiction_level,
        "estimated_search_gain": assessment.estimated_search_gain,
        "assessment_reason": assessment.reason,
    }


def _route_agent_candidate(
    candidate: PolicyCandidate,
    assessment: ObjectiveAssessment | None,
    context: PolicyContext,
) -> ResearchAction | None:
    """Validate a model proposal and map it onto trusted execution settings."""
    terminal = {
        ResearchActionType.STOP,
        ResearchActionType.WRITE_WITH_CAVEAT,
        ResearchActionType.DROP_OBJECTIVE,
    }
    if candidate.action not in terminal:
        if context.remaining_task_slots <= 0:
            return None
        if assessment is None:
            return None
        objective = assessment.objective
        if objective.status == ObjectiveStatus.SUFFICIENT:
            return None
        if objective.search_attempts >= context.max_search_attempts_per_objective:
            return None

    if candidate.action == ResearchActionType.SEARCH_LOCAL:
        if not context.local_corpus_available:
            return None
        route = SearchAPI.LOCAL.value
        tier = "local_deployment" if context.local_model_available else "legacy"
    elif candidate.action in {
        ResearchActionType.SEARCH_WEB,
        ResearchActionType.DIVERSIFY_SOURCES,
    }:
        if context.search_api in {SearchAPI.LOCAL, SearchAPI.NONE}:
            return None
        route = context.search_api.value
        tier = "local_deployment" if context.local_model_available else "legacy"
    elif candidate.action == ResearchActionType.RESOLVE_CONFLICT:
        if context.search_api == SearchAPI.NONE:
            return None
        route = context.search_api.value
        tier = (
            "paid_api"
            if context.paid_model_available
            else "local_deployment" if context.local_model_available else "legacy"
        )
    elif candidate.action == ResearchActionType.ESCALATE_MODEL:
        if not context.paid_model_available or context.search_api == SearchAPI.NONE:
            return None
        route = context.search_api.value
        tier = "paid_api"
    else:
        route = "none"
        tier = "none"

    query = candidate.query.strip()
    if candidate.action not in terminal and not query:
        return None
    return ResearchAction(
        objective_id=candidate.objective_id,
        action=candidate.action,
        retrieval_route=route,
        model_tier=tier,
        utility=round(candidate.expected_quality_gain - candidate.expected_cost, 4),
        reason=candidate.reason,
        query=query,
        candidate_id=candidate.candidate_id,
        expected_quality_gain=candidate.expected_quality_gain,
        expected_cost=candidate.expected_cost,
        success_condition=candidate.success_condition,
        policy_source="agent",
    )


async def decide_research_actions(
    model: BaseChatModel,
    *,
    research_brief: str,
    assessments: list[ObjectiveAssessment],
    context: PolicyContext,
    decision_history: list[dict[str, Any]],
    native_structured_output: bool,
    retries: int,
    config: RunnableConfig | None = None,
) -> ResearchPolicyResult:
    """Let a model policy choose the next action from state and observed outcomes.

    The model owns strategy selection. This function owns only protocol safety:
    known objectives, available tools, attempt limits, routing credentials, and
    the maximum number of actions.
    """
    available_actions = [
        ResearchActionType.WRITE_WITH_CAVEAT.value,
        ResearchActionType.DROP_OBJECTIVE.value,
        ResearchActionType.STOP.value,
    ]
    if context.remaining_task_slots > 0 and context.local_corpus_available:
        available_actions.append(ResearchActionType.SEARCH_LOCAL.value)
    if (
        context.remaining_task_slots > 0
        and context.search_api not in {SearchAPI.LOCAL, SearchAPI.NONE}
    ):
        available_actions.extend(
            [
                ResearchActionType.SEARCH_WEB.value,
                ResearchActionType.DIVERSIFY_SOURCES.value,
            ]
        )
    if context.remaining_task_slots > 0 and context.search_api != SearchAPI.NONE:
        available_actions.append(ResearchActionType.RESOLVE_CONFLICT.value)
    if (
        context.remaining_task_slots > 0
        and context.paid_model_available
        and context.search_api != SearchAPI.NONE
    ):
        available_actions.append(ResearchActionType.ESCALATE_MODEL.value)

    selection_limit = max(
        1, min(context.max_actions, max(1, context.remaining_task_slots))
    )

    state = {
        "research_brief": research_brief,
        "objectives": [_assessment_payload(item) for item in assessments],
        "available_actions": available_actions,
        "remaining_task_slots": context.remaining_task_slots,
        "max_selected_actions": selection_limit,
        "budget": context.budget.model_dump(mode="json") if context.budget else None,
        "recent_action_outcomes": decision_history[-12:],
    }
    prompt = (
        "You are the control policy for a deep-research agent. Decide the next "
        "research action from the current evidence state; do not write the report. "
        "Generate distinct candidates, estimate each candidate's marginal quality "
        "gain and normalized cost, then select only the best executable candidates. "
        "The query may refine the objective but must stay within the user's brief. "
        "React to observed outcomes: do not repeat a route/query that produced no new "
        "evidence unless you explain what materially changed. Prefer DIVERSIFY_SOURCES "
        "for source monoculture and RESOLVE_CONFLICT for contradictory evidence. Use "
        "ESCALATE_MODEL only when stronger reasoning is worth its cost. Choose "
        "WRITE_WITH_CAVEAT when an important unresolved point should remain explicit, "
        "DROP_OBJECTIVE when it is peripheral or exhausted, and STOP only when no "
        "research action has positive marginal value. Include at least one terminal "
        "candidate. Never mix a selected terminal action with a selected executable "
        "research action. Candidate and selected IDs must be unique.\n\n"
        f"CURRENT STATE:\n{json.dumps(state, ensure_ascii=False)}"
    )
    proposal = await ainvoke_structured_with_fallback(
        model,
        AgentPolicyProposal,
        [HumanMessage(content=prompt)],
        retries=retries,
        native_structured_output=native_structured_output,
        config=config,
        node="research_policy_agent",
    )

    candidate_ids = [item.candidate_id for item in proposal.candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Policy candidate IDs must be unique")
    selected_ids = proposal.selected_candidate_ids
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selected policy candidate IDs must be unique")
    if not selected_ids:
        raise ValueError("Policy must select an explicit action")
    if len(selected_ids) > selection_limit:
        raise ValueError("Policy selected more actions than the executor allows")

    assessments_by_id = {
        item.objective.objective_id: item for item in assessments
    }
    available_action_set = set(available_actions)
    for candidate in proposal.candidates:
        if candidate.action.value not in available_action_set:
            raise ValueError(
                f"Policy proposed an unavailable action: {candidate.candidate_id}"
            )
        if (
            candidate.objective_id not in assessments_by_id
            and candidate.objective_id != "__run__"
        ):
            raise ValueError(
                "Policy candidate references unknown objective: "
                f"{candidate.objective_id}"
            )
    candidates_by_id = {item.candidate_id: item for item in proposal.candidates}
    actions: list[ResearchAction] = []
    for candidate_id in selected_ids:
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Policy selected unknown candidate: {candidate_id}")
        action = _route_agent_candidate(
            candidate,
            assessments_by_id.get(candidate.objective_id),
            context,
        )
        if action is None:
            raise ValueError(f"Policy selected an unavailable action: {candidate_id}")
        actions.append(action)

    terminal = {
        ResearchActionType.STOP,
        ResearchActionType.WRITE_WITH_CAVEAT,
        ResearchActionType.DROP_OBJECTIVE,
    }
    has_terminal = any(item.action in terminal for item in actions)
    if has_terminal and not all(item.action in terminal for item in actions):
        raise ValueError("Policy cannot mix terminal and executable actions")
    return ResearchPolicyResult(
        actions=actions,
        candidates=proposal.candidates,
        situation_summary=proposal.situation_summary,
        policy_source="agent",
    )


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def build_research_objectives(
    research_brief: str,
    claims: list[dict[str, Any]],
    gaps: list[str],
    persisted: dict[str, dict[str, Any]] | None = None,
) -> list[ResearchObjective]:
    """Create actionable objectives from report gaps and rejected claims."""
    persisted = persisted or {}
    objectives: dict[str, ResearchObjective] = {}
    for objective_id, item in persisted.items():
        if (
            objective_id.startswith("objective-gap:")
            and str(item.get("question", "")).strip() not in gaps
        ):
            continue
        try:
            status = ObjectiveStatus(item.get("status", ObjectiveStatus.INSUFFICIENT))
        except ValueError:
            status = ObjectiveStatus.INSUFFICIENT
        objectives[objective_id] = ResearchObjective(
            objective_id=objective_id,
            question=str(item.get("question", "")).strip(),
            importance=float(item.get("importance", 0.5)),
            status=status,
            linked_claim_ids=list(item.get("linked_claim_ids", [])),
            search_attempts=int(item.get("search_attempts", 0)),
        )
    for gap in gaps:
        question = str(gap).strip()
        if not question:
            continue
        objective_id = _stable_id("objective-gap", question)
        previous = persisted.get(objective_id, {})
        objectives[objective_id] = ResearchObjective(
            objective_id=objective_id,
            question=question,
            importance=0.8,
            status=ObjectiveStatus(previous.get("status", ObjectiveStatus.INSUFFICIENT)),
            search_attempts=int(previous.get("search_attempts", 0)),
        )

    importance_by_status = {
        "CONTRADICTED": 0.95,
        "CONFLICTED": 0.95,
        "PARTIALLY_SUPPORTED": 0.75,
        "NOT_SUPPORTED": 0.65,
        "UNVERIFIED": 0.6,
        "REJECTED": 0.5,
    }
    for claim in claims:
        status = str(claim.get("status", "UNVERIFIED"))
        if status not in importance_by_status:
            continue
        question = (
            "Find direct evidence that verifies, qualifies, or refutes this candidate "
            f"claim in the context of the research brief '{research_brief}': "
            f"{claim.get('statement', '')}"
        )
        objective_id = _stable_id("objective-claim", str(claim.get("claim_id", question)))
        previous = persisted.get(objective_id, {})
        objectives[objective_id] = ResearchObjective(
            objective_id=objective_id,
            question=question,
            importance=importance_by_status[status],
            status=(
                ObjectiveStatus.CONFLICTED
                if status in {"CONTRADICTED", "CONFLICTED"}
                else ObjectiveStatus.INSUFFICIENT
            ),
            linked_claim_ids=[str(claim.get("claim_id", ""))],
            search_attempts=int(previous.get("search_attempts", 0)),
        )
    return sorted(objectives.values(), key=lambda item: (-item.importance, item.objective_id))


async def propose_research_objectives(
    model: BaseChatModel,
    *,
    research_brief: str,
    claims: list[dict[str, Any]],
    gaps: list[str],
    persisted: dict[str, dict[str, Any]] | None,
    native_structured_output: bool,
    retries: int,
    config: RunnableConfig | None = None,
    max_objectives: int = 6,
) -> list[ResearchObjective]:
    """Identify the necessary answer conditions before choosing more research."""
    persisted = persisted or {}
    compact_claims = [
        {
            "claim_id": str(item.get("claim_id", "")),
            "statement": str(item.get("statement", ""))[:500],
            "status": str(item.get("status", "UNVERIFIED")),
        }
        for item in claims[:80]
    ]
    prompt = (
        "Decompose the research brief into 2 to 5 necessary answer objectives. An "
        "objective is a question that must be resolved to answer the user's exact "
        "request, not merely a related fact. Link only claim IDs that directly resolve "
        "that objective. Leave linked_claim_ids empty when current claims do not resolve "
        "it. Assign high importance to objectives that determine the final conclusion. "
        "Do not treat the quantity of related claims as proof that an objective is "
        "answered. Preserve an existing objective_id and its exact question when updating "
        "its claim links; leave objective_id null only for a genuinely new objective.\n\n"
        f"RESEARCH BRIEF:\n{research_brief}\n\n"
        f"EXISTING OBJECTIVES:\n{json.dumps(list(persisted.values()), ensure_ascii=False)}\n\n"
        f"CURRENT REPORT GAPS:\n{json.dumps(gaps, ensure_ascii=False)}\n\n"
        f"VALIDATED CLAIMS:\n{json.dumps(compact_claims, ensure_ascii=False)}"
    )
    proposal = await ainvoke_structured_with_fallback(
        model,
        ResearchObjectiveProposal,
        [HumanMessage(content=prompt)],
        retries=retries,
        native_structured_output=native_structured_output,
        config=config,
        node="research_objective_planning",
    )
    known_claim_ids = {str(item.get("claim_id", "")) for item in claims}
    objectives: list[ResearchObjective] = []
    seen_questions: set[str] = set()
    for item in proposal.objectives:
        question = item.question.strip()
        normalized = " ".join(question.lower().split())
        if not question or normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        objective_id = (
            item.objective_id
            if item.objective_id in persisted
            else _stable_id("objective-required", question)
        )
        previous = persisted.get(objective_id, {})
        objectives.append(
            ResearchObjective(
                objective_id=objective_id,
                question=question,
                importance=item.importance,
                linked_claim_ids=sorted(
                    set(item.linked_claim_ids).intersection(known_claim_ids)
                ),
                search_attempts=int(previous.get("search_attempts", 0)),
            )
        )
        if len(objectives) >= max_objectives:
            break
    return objectives


def assess_objectives(
    objectives: list[ResearchObjective],
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[ObjectiveAssessment]:
    """Measure support, diversity, and expected value without another model call."""
    claims_by_id = {str(item["claim_id"]): item for item in claims}
    evidence_by_id = {str(item["evidence_id"]): item for item in evidence}
    output: list[ObjectiveAssessment] = []
    for objective in objectives:
        linked_claims = [
            claims_by_id[claim_id]
            for claim_id in objective.linked_claim_ids
            if claim_id in claims_by_id
        ]
        statuses = [str(item.get("status", "UNVERIFIED")) for item in linked_claims]
        supported = sum(status == "SUPPORTED" for status in statuses)
        semantic_support = supported / len(statuses) if statuses else 0.0
        contradiction_level = (
            sum(status in {"CONTRADICTED", "CONFLICTED"} for status in statuses)
            / len(statuses)
            if statuses
            else 0.0
        )
        evidence_ids = {
            str(evidence_id)
            for claim in linked_claims
            for evidence_id in claim.get("evidence_ids", [])
        }
        source_diversity = len(
            {
                evidence_by_id[evidence_id].get("document_id")
                for evidence_id in evidence_ids
                if evidence_id in evidence_by_id
            }
        )
        if semantic_support == 1.0 and source_diversity >= 1:
            objective.status = ObjectiveStatus.SUFFICIENT
            gain = 0.0
            reason = "The linked claim is supported by persisted evidence."
        elif contradiction_level > 0:
            objective.status = ObjectiveStatus.CONFLICTED
            gain = max(0.25, 0.9 - 0.2 * objective.search_attempts)
            reason = "A high-impact claim is contradicted or conflicted."
        else:
            objective.status = ObjectiveStatus.INSUFFICIENT
            gain = max(0.1, 0.75 - 0.25 * objective.search_attempts)
            reason = "The objective lacks a fully supported evidence-linked claim."
        output.append(
            ObjectiveAssessment(
                objective=objective,
                semantic_support=semantic_support,
                source_diversity=source_diversity,
                contradiction_level=contradiction_level,
                estimated_search_gain=round(gain, 4),
                reason=reason,
            )
        )
    return output


def _remaining_fraction(snapshot: BudgetSnapshot | None, resource: str, limit: int | None) -> float:
    if snapshot is None or limit is None:
        return 1.0
    used = int(getattr(snapshot, resource))
    return max(0.0, min(1.0, (limit - used) / limit))


def select_research_actions(
    assessments: list[ObjectiveAssessment],
    context: PolicyContext,
    *,
    llm_call_limit: int | None = None,
    search_call_limit: int | None = None,
    prompt_token_limit: int | None = None,
    completion_token_limit: int | None = None,
    total_seconds_limit: float | None = None,
) -> list[ResearchAction]:
    """Choose the highest-value bounded actions from the current research state."""
    if context.remaining_task_slots <= 0:
        return []
    remaining_fractions = [
        _remaining_fraction(context.budget, "llm_calls", llm_call_limit),
        _remaining_fraction(context.budget, "search_calls", search_call_limit),
        _remaining_fraction(
            context.budget, "estimated_prompt_tokens", prompt_token_limit
        ),
        _remaining_fraction(
            context.budget, "observed_completion_tokens", completion_token_limit
        ),
    ]
    if context.budget is not None and total_seconds_limit is not None:
        remaining_fractions.append(
            max(
                0.0,
                min(
                    1.0,
                    (total_seconds_limit - context.budget.elapsed_seconds)
                    / total_seconds_limit,
                ),
            )
        )
    raw_budget_factor = min(remaining_fractions)
    reserve = context.budget_reserve_fraction
    budget_factor = max(0.0, (raw_budget_factor - reserve) / (1.0 - reserve))
    candidates: list[ResearchAction] = []
    for assessment in assessments:
        objective = assessment.objective
        if objective.status == ObjectiveStatus.SUFFICIENT:
            continue
        if objective.search_attempts >= context.max_search_attempts_per_objective:
            continue

        if objective.status == ObjectiveStatus.CONFLICTED and context.paid_model_available:
            action_type = ResearchActionType.ESCALATE_MODEL
            retrieval_route = context.search_api.value
            model_tier = "paid_api"
            cost_penalty = 0.3
            action_reason = "Resolve a consequential evidence conflict with stronger synthesis."
        elif objective.search_attempts == 0 and context.local_corpus_available:
            action_type = ResearchActionType.SEARCH_LOCAL
            retrieval_route = SearchAPI.LOCAL.value
            model_tier = (
                "local_deployment" if context.local_model_available else "legacy"
            )
            cost_penalty = 0.05
            action_reason = "Try the low-cost local corpus before external escalation."
        else:
            action_type = (
                ResearchActionType.SEARCH_LOCAL
                if context.search_api == SearchAPI.LOCAL
                else ResearchActionType.SEARCH_WEB
            )
            retrieval_route = context.search_api.value
            model_tier = (
                "local_deployment" if context.local_model_available else "legacy"
            )
            cost_penalty = 0.15
            action_reason = "Broaden retrieval after evidence remained insufficient."

        utility = round(
            objective.importance
            * assessment.estimated_search_gain
            * budget_factor
            - cost_penalty,
            4,
        )
        if utility < context.min_utility:
            continue
        candidates.append(
            ResearchAction(
                objective_id=objective.objective_id,
                action=action_type,
                retrieval_route=retrieval_route,
                model_tier=model_tier,
                utility=utility,
                reason=f"{assessment.reason} {action_reason}",
                query=objective.question,
            )
        )

    candidates.sort(key=lambda item: (-item.utility, item.objective_id))
    # Terminal decisions are part of the policy, not decorative enum values.
    # Emit them only when no executable search action clears the utility gate;
    # the supervisor records the decision and turns the unresolved objective
    # into an explicit caveat instead of silently stopping.
    if not candidates:
        unresolved = [
            item for item in assessments
            if item.objective.status != ObjectiveStatus.SUFFICIENT
        ]
        if not unresolved:
            return [ResearchAction(
                objective_id="__run__",
                action=ResearchActionType.STOP,
                retrieval_route="none",
                model_tier="none",
                utility=0.0,
                reason="All research objectives are sufficiently supported.",
                query="",
            )]
        terminal: list[ResearchAction] = []
        for assessment in unresolved:
            objective = assessment.objective
            if objective.search_attempts >= context.max_search_attempts_per_objective:
                action = ResearchActionType.DROP_OBJECTIVE
                reason = "Search budget for this objective is exhausted; keep it as an evidence gap."
            else:
                action = ResearchActionType.WRITE_WITH_CAVEAT
                reason = "No positive-utility action remains; report the unresolved point with a caveat."
            terminal.append(ResearchAction(
                objective_id=objective.objective_id,
                action=action,
                retrieval_route="none",
                model_tier="none",
                utility=0.0,
                reason=reason,
                query=objective.question,
            ))
        return terminal[: min(context.max_actions, context.remaining_task_slots)]
    limit = min(context.max_actions, context.remaining_task_slots)
    return candidates[:limit]
