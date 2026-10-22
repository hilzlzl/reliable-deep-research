"""Offline integration test for the Reliable V2 report path."""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage

import open_deep_research.deep_researcher as researcher
from open_deep_research.evidence import (
    ClaimStatus,
    Document,
    Evidence,
    ResearchClaim,
)
from open_deep_research.runtime_store import RuntimeStore


class FakeV2Model:
    def with_config(self, _config):
        return self

    async def ainvoke(self, messages):
        prompt = str(messages[-1].content)
        if "Return only one valid JSON object" in prompt:
            return AIMessage(
                content=json.dumps(
                    {
                        "title": "CAD Generation Evaluation",
                        "sections": [
                            {"heading": "Results", "claim_ids": ["claim-1"]}
                        ],
                        "gaps": [],
                    }
                )
            )
        return AIMessage(
            content=(
                "# CAD Generation Evaluation\n\n"
                "The benchmark contains 32 questions [EVIDENCE:ev-1]. "
                "A fabricated [source](https://invented.invalid/paper) is ignored."
            )
        )


def test_final_report_node_renders_only_persisted_sources(tmp_path, monkeypatch):
    database = tmp_path / "runtime.db"
    events = tmp_path / "events.jsonl"
    store = RuntimeStore(database)
    store.save_document(
        Document(
            document_id="doc-1",
            url="local://researchbench/benchmark.md",
            title="ResearchBench",
            content_hash="sha256:benchmark",
            retrieved_at="2026-10-10T00:00:00+00:00",
        )
    )
    store.save_evidence(
        Evidence(
            evidence_id="ev-1",
            task_id="research:1",
            document_id="doc-1",
            excerpt="The benchmark contains 32 questions.",
            claim_text="The benchmark contains 32 questions.",
            confidence=0.9,
        )
    )
    store.save_claim(
        ResearchClaim(
            claim_id="claim-1",
            statement="The benchmark contains 32 questions.",
            evidence_ids=["ev-1"],
            status=ClaimStatus.SUPPORTED,
        )
    )
    monkeypatch.setattr(researcher, "configurable_model", FakeV2Model())
    config = {
        "configurable": {
            "runtime_db_path": str(database),
            "event_log_path": str(events),
            "run_id": "v2-offline-test",
            "allow_clarification": False,
            "enable_evidence_context": True,
            "enable_structured_report": True,
            "enable_gap_search": False,
            "native_structured_output": False,
            "final_report_model": "openai:fake",
        }
    }

    result = asyncio.run(
        researcher.final_report_generation(
            {
                "messages": [HumanMessage(content="Evaluate CAD generation systems")],
                "research_brief": "Evaluate CAD generation systems",
                "notes": [],
            },
            config,
        )
    )

    report = result["final_report"]
    assert "[1]" in report
    assert "local://researchbench/benchmark.md" in report
    assert "https://invented.invalid/paper" not in report

    event_records = [
        json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()
    ]
    event_data = {item["event_type"]: item["data"] for item in event_records}
    assert event_data["report_plan_created"]["claim_evidence_link_rate"] == 1.0
    assert event_data["report_citations_rendered"]["citation_coverage"] == 1.0
    assert event_data["report_source_audited"]["source_precision"] == 1.0
    assert event_data["report_source_audited"]["unsupported_sources"] == []


def test_final_report_falls_back_when_model_omits_evidence_markers(
    tmp_path, monkeypatch
):
    database = tmp_path / "runtime.db"
    events = tmp_path / "events.jsonl"
    store = RuntimeStore(database)
    store.save_document(
        Document(
            document_id="doc-1",
            url="local://researchbench/paper.md",
            title="Paper",
            content_hash="sha256:paper",
            retrieved_at="2026-10-10T00:00:00+00:00",
        )
    )
    store.save_evidence(
        Evidence(
            evidence_id="ev-1",
            task_id="research:1",
            document_id="doc-1",
            excerpt="DeepCAD uses an operation sequence.",
            claim_text="DeepCAD uses an operation sequence.",
            confidence=0.9,
        )
    )
    store.save_claim(
        ResearchClaim(
            claim_id="claim-1",
            statement="DeepCAD uses an operation sequence.",
            evidence_ids=["ev-1"],
            status=ClaimStatus.SUPPORTED,
        )
    )
    monkeypatch.setattr(researcher, "configurable_model", FakeV2Model())
    original = FakeV2Model.ainvoke

    async def omit_markers(self, messages):
        prompt = str(messages[-1].content)
        if "Return only one valid JSON object" in prompt:
            return await original(self, messages)
        return AIMessage(content="# Report\n\nAn uncited model draft.")

    monkeypatch.setattr(FakeV2Model, "ainvoke", omit_markers)
    config = {
        "configurable": {
            "runtime_db_path": str(database),
            "event_log_path": str(events),
            "run_id": "v2-fallback-test",
            "enable_evidence_context": True,
            "enable_structured_report": True,
            "enable_gap_search": False,
            "native_structured_output": False,
            "final_report_model": "openai:fake",
        }
    }

    result = asyncio.run(
        researcher.final_report_generation(
            {
                "messages": [HumanMessage(content="Compare CAD models")],
                "research_brief": "Compare CAD models",
                "notes": [],
            },
            config,
        )
    )

    assert "DeepCAD uses an operation sequence." in result["final_report"]
    assert "local://researchbench/paper.md" in result["final_report"]
    event_types = {
        json.loads(line)["event_type"]
        for line in events.read_text(encoding="utf-8").splitlines()
    }
    assert "report_draft_grounding_fallback" in event_types
