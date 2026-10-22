"""Offline integration test for the evidence-aware research loop."""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage

import open_deep_research.deep_researcher as researcher
from open_deep_research.configuration import Configuration
from open_deep_research.evidence import ClaimStatus, Document, Evidence, ResearchClaim
from open_deep_research.report_planning import ReportPlan
from open_deep_research.run_budget import RunBudget
from open_deep_research.runtime_store import RuntimeStore


class PlanModel:
    async def ainvoke(self, _messages):
        prompt = str(_messages[-1].content)
        if "control policy for a deep-research agent" in prompt:
            state = json.loads(prompt.split("CURRENT STATE:\n", 1)[1])
            objective = state["objectives"][0]
            sufficient = objective["status"] == "SUFFICIENT"
            selected = "stop" if sufficient else "search"
            return AIMessage(
                content=json.dumps(
                    {
                        "situation_summary": (
                            "Evidence is sufficient."
                            if sufficient
                            else "A targeted local search has positive value."
                        ),
                        "candidates": [
                            {
                                "candidate_id": "search",
                                "objective_id": objective["objective_id"],
                                "action": "SEARCH_LOCAL",
                                "query": objective["question"],
                                "expected_quality_gain": 0.8,
                                "expected_cost": 0.1,
                                "success_condition": "Add direct evidence.",
                                "reason": "The objective is unresolved.",
                            },
                            {
                                "candidate_id": "stop",
                                "objective_id": "__run__",
                                "action": "STOP",
                                "query": "",
                                "expected_quality_gain": 0.0,
                                "expected_cost": 0.0,
                                "success_condition": "Proceed to the report.",
                                "reason": "No further research is needed.",
                            },
                        ],
                        "selected_candidate_ids": [selected],
                    }
                )
            )
        if "Decompose the research brief" in prompt:
            linked_ids = ["claim-1"] if "claim-1" in prompt else []
            return AIMessage(
                content=json.dumps(
                    {
                        "objectives": [
                            {
                                "question": "Find a direct comparison.",
                                "importance": 0.9,
                                "linked_claim_ids": linked_ids,
                            }
                        ]
                    }
                )
            )
        return AIMessage(
            content=json.dumps(
                {
                    "title": "Adaptive report",
                    "sections": [{"heading": "Finding", "claim_ids": ["claim-1"]}],
                    "gaps": [],
                }
            )
        )


class EvidenceProducingResearcher:
    def __init__(self, store):
        self.store = store

    async def ainvoke(self, _state, config):
        task_id = config["configurable"]["current_task_id"]
        self.store.save_document(
            Document(
                document_id="doc-1",
                url="local://paper",
                title="Paper",
                content_hash="paper-hash",
                retrieved_at="2026-10-12T00:00:00+00:00",
            )
        )
        self.store.save_evidence(
            Evidence(
                evidence_id="ev-1",
                task_id=task_id,
                document_id="doc-1",
                excerpt="Direct evidence for the missing comparison.",
                claim_text="The comparison is supported.",
                confidence=0.9,
            )
        )
        self.store.save_claim(
            ResearchClaim(
                claim_id="claim-1",
                statement="The comparison is supported.",
                evidence_ids=["ev-1"],
                status=ClaimStatus.SUPPORTED,
            )
        )
        return {"compressed_research": "done", "raw_notes": []}


class AdaptivePipelineModel:
    def with_config(self, _config):
        return self

    async def ainvoke(self, messages):
        prompt = str(messages[-1].content)
        if "Decompose the research brief" in prompt:
            return AIMessage(
                content=json.dumps(
                    {
                        "objectives": [
                            {
                                "question": "Determine whether constraints are preserved.",
                                "importance": 0.9,
                                "linked_claim_ids": ["claim-1"],
                            }
                        ]
                    }
                )
            )
        if "Create a concise report outline" in prompt:
            return AIMessage(
                content=json.dumps(
                    {
                        "title": "CAD report",
                        "sections": [
                            {"heading": "Finding", "claim_ids": ["claim-1"]}
                        ],
                        "gaps": [],
                    }
                )
            )
        if "Extract every externally verifiable factual assertion" in prompt:
            sentence = "The method preserves constraints [EVIDENCE:ev-1]."
            return AIMessage(
                content=json.dumps(
                    {
                        "assertions": [
                            {
                                "assertion_id": "a",
                                "sentence": sentence,
                                "evidence_ids": ["ev-1"],
                                "verdict": "SUPPORTED",
                                "reason": "Direct support.",
                                "repair_action": "KEEP",
                            }
                        ]
                    }
                )
            )
        return AIMessage(
            content="# CAD report\n\nThe method preserves constraints [EVIDENCE:ev-1]."
        )


def test_adaptive_loop_turns_a_gap_into_a_persisted_decision_and_evidence(
    tmp_path, monkeypatch
):
    store = RuntimeStore(tmp_path / "runtime.db")
    monkeypatch.setattr(
        researcher, "researcher_subgraph", EvidenceProducingResearcher(store)
    )
    config = {
        "configurable": {
            "runtime_db_path": str(tmp_path / "runtime.db"),
            "observability_enabled": False,
            "search_api": "local",
            "local_corpus_path": str(tmp_path),
            "max_total_research_units": 2,
            "max_adaptive_rounds": 2,
            "max_adaptive_actions_per_round": 1,
            "native_structured_output": False,
            "max_llm_calls": 20,
            "run_budget": RunBudget(max_llm_calls=20),
        }
    }
    configurable = Configuration.from_runnable_config(config)

    evidence, claims, plan = asyncio.run(
        researcher._run_adaptive_research_loop(
            research_brief="Compare two CAD generation methods",
            report_plan=ReportPlan(
                title="Comparison",
                sections=[],
                gaps=["Find a direct comparison."],
            ),
            claims=[],
            evidence=[],
            store=store,
            writer_model=PlanModel(),
            validation_model=None,
            config=config,
            configurable=configurable,
        )
    )

    assert [item["evidence_id"] for item in evidence] == ["ev-1"]
    assert [item["claim_id"] for item in claims] == ["claim-1"]
    assert plan.gaps == []
    decisions = store.list_adaptive_decisions()
    assert len(decisions) == 2
    assert decisions[0]["payload"]["evidence_gain_for_round"] == 1
    assert decisions[0]["payload"]["policy_source"] == "agent"
    assert len(decisions[0]["payload"]["policy_candidates"]) == 2
    assert decisions[0]["payload"]["actual_cost"]["llm_calls"] == 2
    assert decisions[1]["action"] == "STOP"


def test_adaptive_profile_persists_sentence_audit_before_rendering(
    tmp_path, monkeypatch
):
    database = tmp_path / "runtime.db"
    store = RuntimeStore(database)
    store.save_document(
        Document(
            document_id="doc-1",
            url="local://paper",
            title="Paper",
            content_hash="paper-hash",
            retrieved_at="2026-10-12T00:00:00+00:00",
        )
    )
    store.save_evidence(
        Evidence(
            evidence_id="ev-1",
            task_id="research:1",
            document_id="doc-1",
            excerpt="The method preserves constraints.",
            claim_text="The method preserves constraints.",
            confidence=0.9,
        )
    )
    store.save_claim(
        ResearchClaim(
            claim_id="claim-1",
            statement="The method preserves constraints.",
            evidence_ids=["ev-1"],
            status=ClaimStatus.SUPPORTED,
        )
    )
    monkeypatch.setattr(researcher, "configurable_model", AdaptivePipelineModel())
    config = {
        "configurable": {
            "runtime_db_path": str(database),
            "observability_enabled": False,
            "enable_evidence_context": True,
            "enable_structured_report": True,
            "enable_adaptive_research": True,
            "enable_sentence_audit": True,
            "native_structured_output": False,
            "max_total_research_units": 1,
            "final_report_model": "openai:fake",
            "research_model": "openai:fake",
        }
    }

    result = asyncio.run(
        researcher.final_report_generation(
            {
                "messages": [HumanMessage(content="Assess the method")],
                "research_brief": "Assess the method",
                "notes": [],
            },
            config,
        )
    )

    assert "local://paper" in result["final_report"]
    assertions = store.list_report_assertions()
    assert len(assertions) == 1
    assert assertions[0]["verdict"] == "SUPPORTED"


def test_structured_profile_rejects_research_complete_before_any_research(
    tmp_path, monkeypatch
):
    store = RuntimeStore(tmp_path / "runtime.db")
    monkeypatch.setattr(
        researcher, "researcher_subgraph", EvidenceProducingResearcher(store)
    )
    config = {
        "configurable": {
            "runtime_db_path": str(tmp_path / "runtime.db"),
            "observability_enabled": False,
            "experiment_profile": "adaptive_v3",
            "enable_reliable_runtime": True,
            "search_api": "local",
            "local_corpus_path": str(tmp_path),
            "max_total_research_units": 2,
            "max_initial_research_units": 1,
            "max_researcher_iterations": 1,
        }
    }
    premature = AIMessage(
        content="",
        tool_calls=[{"id": "done", "name": "ResearchComplete", "args": {}}],
    )

    command = asyncio.run(
        researcher.supervisor_tools(
            {
                "supervisor_messages": [premature],
                "research_brief": "Compare CAD methods",
                "research_iterations": 1,
                "research_units_started": 0,
                "raw_notes": [],
            },
            config,
        )
    )

    assert command.goto == "supervisor"
    assert store.task_counts() == (1, 1)


def test_structured_profile_forces_research_when_thinking_exhausts_iterations(
    tmp_path, monkeypatch
):
    store = RuntimeStore(tmp_path / "runtime.db")
    monkeypatch.setattr(
        researcher, "researcher_subgraph", EvidenceProducingResearcher(store)
    )
    config = {
        "configurable": {
            "runtime_db_path": str(tmp_path / "runtime.db"),
            "observability_enabled": False,
            "experiment_profile": "adaptive_v3",
            "enable_reliable_runtime": True,
            "search_api": "local",
            "local_corpus_path": str(tmp_path),
            "max_total_research_units": 2,
            "max_initial_research_units": 1,
            "max_researcher_iterations": 1,
        }
    }
    thinking = AIMessage(
        content="",
        tool_calls=[
            {"id": "think", "name": "think_tool", "args": {"reflection": "plan"}}
        ],
    )

    command = asyncio.run(
        researcher.supervisor_tools(
            {
                "supervisor_messages": [thinking],
                "research_brief": "Compare CAD methods",
                "research_iterations": 2,
                "research_units_started": 0,
                "raw_notes": [],
            },
            config,
        )
    )

    assert command.goto == "supervisor"
    assert store.task_counts() == (1, 1)
