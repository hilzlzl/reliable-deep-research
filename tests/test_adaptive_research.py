"""Tests for evidence-aware research policy decisions."""

import asyncio
import json

from langchain_core.messages import AIMessage

from open_deep_research.adaptive_research import (
    PolicyContext,
    ResearchActionType,
    assess_objectives,
    build_research_objectives,
    decide_research_actions,
    select_research_actions,
    propose_research_objectives,
)
from open_deep_research.configuration import SearchAPI
from open_deep_research.run_budget import RunBudget


def test_conflicted_high_impact_claim_escalates_when_paid_model_exists():
    claims = [
        {
            "claim_id": "claim-1",
            "statement": "Diffusion is always more constraint-valid.",
            "evidence_ids": ["ev-1"],
            "status": "CONTRADICTED",
        }
    ]
    evidence = [{"evidence_id": "ev-1", "document_id": "doc-1"}]
    objectives = build_research_objectives("Compare CAD methods", claims, [])
    assessments = assess_objectives(objectives, claims, evidence)

    actions = select_research_actions(
        assessments,
        PolicyContext(
            search_api=SearchAPI.HYBRID,
            local_corpus_available=True,
            local_model_available=True,
            paid_model_available=True,
            remaining_task_slots=2,
        ),
    )

    assert actions[0].action == ResearchActionType.ESCALATE_MODEL
    assert actions[0].model_tier == "paid_api"


def test_gap_uses_local_first_then_broadens_on_next_attempt():
    gap = "Find direct editability evaluations."
    objectives = build_research_objectives("Compare CAD methods", [], [gap])
    first = select_research_actions(
        assess_objectives(objectives, [], []),
        PolicyContext(
            search_api=SearchAPI.HYBRID,
            local_corpus_available=True,
            local_model_available=True,
            remaining_task_slots=2,
        ),
    )
    assert first[0].action == ResearchActionType.SEARCH_LOCAL
    assert first[0].retrieval_route == "local"

    persisted = {
        objectives[0].objective_id: {
            "status": "INSUFFICIENT",
            "search_attempts": 1,
        }
    }
    second_objectives = build_research_objectives(
        "Compare CAD methods", [], [gap], persisted
    )
    second = select_research_actions(
        assess_objectives(second_objectives, [], []),
        PolicyContext(
            search_api=SearchAPI.HYBRID,
            local_corpus_available=True,
            local_model_available=True,
            remaining_task_slots=2,
        ),
    )
    assert second[0].action == ResearchActionType.SEARCH_WEB
    assert second[0].retrieval_route == "hybrid"


def test_policy_stops_when_no_task_capacity_remains():
    objectives = build_research_objectives("Question", [], ["Missing evidence"])
    actions = select_research_actions(
        assess_objectives(objectives, [], []),
        PolicyContext(
            search_api=SearchAPI.LOCAL,
            local_corpus_available=True,
            remaining_task_slots=0,
        ),
    )
    assert actions == []


class ObjectiveModel:
    async def ainvoke(self, _messages):
        return AIMessage(
            content=json.dumps(
                {
                    "objectives": [
                        {
                            "question": "Is there a universal validity guarantee?",
                            "importance": 1.0,
                            "linked_claim_ids": ["claim-known", "claim-invented"],
                        }
                    ]
                }
            )
        )


def test_objective_planning_keeps_only_known_claim_links():
    objectives = asyncio.run(
        propose_research_objectives(
            ObjectiveModel(),
            research_brief="Which method is guaranteed to be best?",
            claims=[
                {
                    "claim_id": "claim-known",
                    "statement": "One benchmark reports a result.",
                    "status": "SUPPORTED",
                }
            ],
            gaps=[],
            persisted=None,
            native_structured_output=False,
            retries=1,
        )
    )

    assert objectives[0].importance == 1.0
    assert objectives[0].linked_claim_ids == ["claim-known"]


def test_policy_preserves_configured_final_stage_budget():
    objectives = build_research_objectives("Question", [], ["Missing evidence"])
    budget = RunBudget(max_completion_tokens=100)
    budget.consume("observed_completion_tokens", 85)

    actions = select_research_actions(
        assess_objectives(objectives, [], []),
        PolicyContext(
            search_api=SearchAPI.LOCAL,
            local_corpus_available=True,
            remaining_task_slots=1,
            budget=budget.snapshot(),
            budget_reserve_fraction=0.2,
        ),
        completion_token_limit=100,
    )

    assert actions[0].action == ResearchActionType.WRITE_WITH_CAVEAT


def test_policy_emits_explicit_caveat_when_search_has_no_positive_utility():
    objectives = build_research_objectives("Question", [], ["Missing evidence"])
    actions = select_research_actions(
        assess_objectives(objectives, [], []),
        PolicyContext(
            search_api=SearchAPI.LOCAL,
            local_corpus_available=False,
            remaining_task_slots=1,
            min_utility=0.9,
        ),
    )
    assert actions[0].action == ResearchActionType.WRITE_WITH_CAVEAT


def test_policy_emits_stop_when_all_objectives_are_supported():
    objectives = build_research_objectives(
        "Question",
        [{"claim_id": "c1", "statement": "Fact", "status": "SUPPORTED", "evidence_ids": ["e1"]}],
        [],
    )
    actions = select_research_actions(
        assess_objectives(objectives, [{"claim_id": "c1", "status": "SUPPORTED", "evidence_ids": ["e1"]}], [{"evidence_id": "e1", "document_id": "d1"}]),
        PolicyContext(search_api=SearchAPI.LOCAL, remaining_task_slots=1),
    )
    assert actions[0].action == ResearchActionType.STOP


class AgentPolicyModel:
    def __init__(self, action="SEARCH_LOCAL"):
        self.action = action

    async def ainvoke(self, messages):
        prompt = str(messages[-1].content)
        state = json.loads(prompt.split("CURRENT STATE:\n", 1)[1])
        objective_id = state["objectives"][0]["objective_id"]
        action = self.action
        if state["recent_action_outcomes"]:
            action = "WRITE_WITH_CAVEAT"
        return AIMessage(
            content=json.dumps(
                {
                    "situation_summary": "The next action depends on observed gain.",
                    "candidates": [
                        {
                            "candidate_id": "continue",
                            "objective_id": objective_id,
                            "action": action,
                            "query": state["objectives"][0]["question"],
                            "expected_quality_gain": 0.7,
                            "expected_cost": 0.1,
                            "success_condition": "Add direct evidence.",
                            "reason": "Best marginal value in the current state.",
                        },
                        {
                            "candidate_id": "stop",
                            "objective_id": objective_id,
                            "action": "WRITE_WITH_CAVEAT",
                            "query": "",
                            "expected_quality_gain": 0.1,
                            "expected_cost": 0.0,
                            "success_condition": "Preserve the unresolved gap.",
                            "reason": "Terminal alternative.",
                        },
                    ],
                    "selected_candidate_ids": ["continue"],
                }
            )
        )


def test_agent_policy_changes_action_after_zero_gain_feedback():
    objectives = build_research_objectives("Question", [], ["Missing evidence"])
    assessments = assess_objectives(objectives, [], [])
    context = PolicyContext(
        search_api=SearchAPI.LOCAL,
        local_corpus_available=True,
        remaining_task_slots=1,
    )

    first = asyncio.run(
        decide_research_actions(
            AgentPolicyModel(),
            research_brief="Question",
            assessments=assessments,
            context=context,
            decision_history=[],
            native_structured_output=False,
            retries=1,
        )
    )
    second = asyncio.run(
        decide_research_actions(
            AgentPolicyModel(),
            research_brief="Question",
            assessments=assessments,
            context=context,
            decision_history=[
                {
                    "action": "SEARCH_LOCAL",
                    "query": "Missing evidence",
                    "evidence_gain": 0,
                    "supported_claim_gain": 0,
                }
            ],
            native_structured_output=False,
            retries=1,
        )
    )

    assert first.actions[0].action == ResearchActionType.SEARCH_LOCAL
    assert second.actions[0].action == ResearchActionType.WRITE_WITH_CAVEAT
    assert first.actions[0].policy_source == "agent"
    assert len(first.candidates) == 2


def test_agent_policy_rejects_unavailable_web_action():
    objectives = build_research_objectives("Question", [], ["Missing evidence"])
    assessments = assess_objectives(objectives, [], [])

    try:
        asyncio.run(
            decide_research_actions(
                AgentPolicyModel(action="SEARCH_WEB"),
                research_brief="Question",
                assessments=assessments,
                context=PolicyContext(
                    search_api=SearchAPI.LOCAL,
                    local_corpus_available=True,
                    remaining_task_slots=1,
                ),
                decision_history=[],
                native_structured_output=False,
                retries=1,
            )
        )
    except ValueError as error:
        assert "unavailable action" in str(error)
    else:
        raise AssertionError("Unavailable policy action was accepted")
