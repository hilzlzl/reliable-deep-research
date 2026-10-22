"""Persistence tests for task recovery groundwork."""

import asyncio
from datetime import datetime, timezone

from open_deep_research.evidence import Document, Evidence, ResearchClaim, ClaimStatus
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.adaptive_research import (
    ObjectiveStatus,
    ResearchObjective,
)
from open_deep_research.report_audit import AssertionRepair, ReportAssertion
from open_deep_research.task_runtime import DagTaskScheduler, ResearchTask, TaskStatus


def test_scheduler_persists_task_lifecycle_and_result(tmp_path):
    db_path = tmp_path / "runtime.db"

    async def worker(task):
        return {"content": task.description}

    task = ResearchTask(task_id="persisted", description="store me")
    config = {"configurable": {"runtime_db_path": str(db_path), "observability_enabled": False}}
    result = asyncio.run(DagTaskScheduler().run([task], worker, config))

    store = RuntimeStore(db_path)
    snapshot = store.task_snapshot("persisted")
    assert result.succeeded
    assert snapshot["status"] == "SUCCEEDED"
    assert snapshot["attempt_count"] == 1
    assert store.event_count("persisted") == 2
    assert store.task_counts() == (1, 1)


def test_empty_runtime_store_has_zero_task_counts(tmp_path):
    store = RuntimeStore(tmp_path / "empty.db")

    assert store.task_counts() == (0, 0)


def test_scheduler_restores_successful_tasks_without_reexecution(tmp_path):
    db_path = tmp_path / "resume.db"
    calls = []

    async def first_worker(task):
        calls.append(task.task_id)
        return {"value": task.task_id}

    config = {"configurable": {"runtime_db_path": str(db_path), "observability_enabled": False}}
    first_tasks = [ResearchTask(task_id="root", description="root")]
    asyncio.run(DagTaskScheduler().run(first_tasks, first_worker, config))

    async def resumed_worker(task):
        calls.append(task.task_id)
        return {"value": task.task_id}

    resumed_tasks = [
        ResearchTask(task_id="root", description="root"),
        ResearchTask(task_id="child", description="child", dependencies=["root"]),
    ]
    result = asyncio.run(DagTaskScheduler().run(resumed_tasks, resumed_worker, config))

    assert calls == ["root", "child"]
    assert result.results["root"].content == {"value": "root"}
    assert result.results["child"].content == {"value": "child"}
    assert result.succeeded


def test_replay_invalidates_derived_evidence_and_claims(tmp_path):
    store = RuntimeStore(tmp_path / "lineage.db")
    store.save_document(Document(
        document_id="doc",
        url="https://example.test/doc",
        title="Doc",
        content_hash="hash",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    ))
    store.save_evidence(Evidence(
        evidence_id="ev-old",
        task_id="root",
        document_id="doc",
        excerpt="old evidence",
        claim_text="old claim",
    ))
    store.save_claim(ResearchClaim(
        claim_id="claim-old",
        statement="old claim",
        evidence_ids=["ev-old"],
        status=ClaimStatus.SUPPORTED,
    ))

    invalidated = store.reset_tasks_for_replay({"root"})

    assert invalidated["evidence"] == 1
    assert invalidated["claims"] == 1
    assert store.list_evidence() == []
    assert store.list_claims() == []


def test_resume_invalidates_evidence_from_uncommitted_task(tmp_path):
    database = tmp_path / "crash.db"
    store = RuntimeStore(database)
    task = ResearchTask(task_id="root", description="root")
    task.status = TaskStatus.RUNNING
    task.attempt_count = 1
    store.save_task(task)
    store.save_document(Document(
        document_id="doc",
        url="https://example.test/crash",
        title="Crash",
        content_hash="crash-hash",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    ))
    store.save_evidence(Evidence(
        evidence_id="ev-uncommitted",
        task_id="root",
        document_id="doc",
        excerpt="written before process exit",
        claim_text="uncommitted",
    ))

    calls = []

    async def worker(resumed_task):
        calls.append(resumed_task.task_id)
        return "fresh"

    result = asyncio.run(DagTaskScheduler().run(
        [ResearchTask(task_id="root", description="root", retry_limit=1)],
        worker,
        {"configurable": {"runtime_db_path": str(database), "observability_enabled": False}},
    ))

    assert result.succeeded
    assert calls == ["root"]
    assert store.list_evidence() == []


def test_retry_invalidates_evidence_from_failed_attempt(tmp_path):
    database = tmp_path / "retry.db"
    store = RuntimeStore(database)
    config = {"configurable": {"runtime_db_path": str(database), "observability_enabled": False}}

    async def worker(task):
        attempt = task.attempt_count
        store.save_document(Document(
            document_id=f"doc-{attempt}",
            url=f"https://example.test/{attempt}",
            title="Attempt",
            content_hash=f"hash-{attempt}",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        ))
        store.save_evidence(Evidence(
            evidence_id=f"ev-{attempt}",
            task_id=task.task_id,
            document_id=f"doc-{attempt}",
            excerpt=f"attempt {attempt}",
            claim_text=f"attempt {attempt}",
        ))
        if attempt == 1:
            store.save_claim(ResearchClaim(
                claim_id="claim-old",
                statement="attempt 1",
                evidence_ids=["ev-1"],
                status=ClaimStatus.SUPPORTED,
            ))
            raise RuntimeError("injected failure after evidence write")
        return "fresh"

    result = asyncio.run(DagTaskScheduler().run(
        [ResearchTask(task_id="root", description="root", retry_limit=1)],
        worker,
        config,
    ))

    assert result.succeeded
    assert result.results["root"].attempts == 2
    assert [item["evidence_id"] for item in store.list_evidence()] == ["ev-2"]
    assert store.list_claims() == []


def test_adaptive_decisions_and_report_assertions_are_persisted(tmp_path):
    store = RuntimeStore(tmp_path / "adaptive.db")
    objective = ResearchObjective(
        objective_id="objective-1",
        question="Find constraint-validity evidence",
        importance=0.9,
        status=ObjectiveStatus.INSUFFICIENT,
        search_attempts=1,
    )
    store.save_research_objective(objective)
    store.save_adaptive_decision(
        decision_id="decision-1",
        round_number=1,
        objective_id=objective.objective_id,
        action="SEARCH_LOCAL",
        payload={"utility": 0.6},
    )
    store.replace_report_assertions([
        ReportAssertion(
            assertion_id="assertion-1",
            sentence="A supported sentence [EVIDENCE:ev-1].",
            evidence_ids=["ev-1"],
            verdict=ClaimStatus.SUPPORTED,
            reason="Directly entailed.",
            repair_action=AssertionRepair.KEEP,
        )
    ])

    assert store.list_research_objectives()[0]["search_attempts"] == 1
    assert store.list_adaptive_decisions()[0]["payload"]["utility"] == 0.6
    assert store.list_report_assertions()[0]["verdict"] == "SUPPORTED"
