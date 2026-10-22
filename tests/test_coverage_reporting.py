"""Behavioral regressions for answer completeness and verified local repair."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import open_deep_research.coverage_reporting as coverage
import open_deep_research.deep_researcher as researcher
from open_deep_research.configuration import Configuration
from open_deep_research.evidence import Document, Evidence
from open_deep_research.run_budget import BudgetExceeded, RunBudget
from open_deep_research.runtime_store import RuntimeStore


def evidence_records():
    return [
        {
            "evidence_id": "ev-1",
            "source_url": "local://paper",
            "source_title": "Paper",
            "excerpt": "Reflections are stored in memory and added to the next actor prompt.",
        },
        {
            "evidence_id": "ev-2",
            "source_url": "local://appendix",
            "source_title": "Appendix",
            "excerpt": "The method improves through language feedback without updating weights.",
        },
        {
            "evidence_id": "ev-3",
            "source_url": "local://extra",
            "source_title": "Extra",
            "excerpt": "The evaluation used three trials.",
        },
    ]


def questions():
    return [
        coverage.QuestionRequirement(
            question_id="q1", question="How is feedback used?"
        ),
        coverage.QuestionRequirement(question_id="q2", question="Are weights updated?"),
    ]


FEEDBACK = "Stored reflections enter the next actor prompt [EVIDENCE:ev-1]."
WEIGHTS = "Improvement uses language feedback without updating weights [EVIDENCE:ev-2]."


def plan():
    return coverage.CoveragePlan(
        title="Reflexion",
        answers=[
            coverage.PlannedAnswer(
                question_id="q1",
                status="SUPPORTED",
                answer=FEEDBACK,
                evidence_ids=["ev-1"],
            ),
            coverage.PlannedAnswer(
                question_id="q2",
                status="SUPPORTED",
                answer=WEIGHTS,
                evidence_ids=["ev-2"],
            ),
        ],
    )


def complete_check(report):
    checks = []
    for block_id, text in coverage.report_blocks(report):
        if text.startswith("#"):
            continue
        ids = coverage.MARKER.findall(text)
        checks.append(
            coverage.BlockCheck(
                block_id=block_id,
                verdict="SUPPORTED",
                evidence_ids=ids,
                reason="Direct entailment.",
            )
        )
    return coverage.DraftCheck(
        blocks=checks,
        questions=[
            coverage.QuestionCheck(
                question_id="q1",
                status="ANSWERED" if FEEDBACK in report else "UNANSWERED",
                answer_excerpt=FEEDBACK if FEEDBACK in report else "",
            ),
            coverage.QuestionCheck(
                question_id="q2",
                status="ANSWERED" if WEIGHTS in report else "UNANSWERED",
                answer_excerpt=WEIGHTS if WEIGHTS in report else "",
            ),
        ],
    )


class Responses:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def with_config(self, _):
        return self

    def with_structured_output(self, _):
        return self

    def with_retry(self, **_):
        return self

    async def ainvoke(self, messages):
        self.prompts.append(str(messages[-1].content))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return AIMessage(
            content=response
            if isinstance(response, str)
            else response.model_dump_json()
        )


def run_pipeline(
    model, *, gap_search=None, evidence=None, reload_evidence=None, budget=None,
    feedback=False,
):
    async def no_search(_):
        return False

    return asyncio.run(
        coverage.generate_coverage_report(
            model,
            questions=questions(),
            evidence=evidence or evidence_records(),
            config={
                "configurable": {"observability_enabled": False, "run_budget": budget}
            },
            configurable=Configuration(
                native_structured_output=False,
                max_structured_output_retries=1,
                enable_gap_search=gap_search is not None,
                enable_coverage_feedback_repair=feedback,
            ),
            gap_search=gap_search or no_search,
            reload_evidence=reload_evidence or evidence_records,
        )
    )


def test_unused_evidence_does_not_replace_explanatory_report():
    draft = "# Reflexion\n\n" + FEEDBACK + "\n\n" + WEIGHTS
    model = Responses(plan(), draft, complete_check(draft))
    result = run_pipeline(model)
    assert "Stored reflections enter the next actor prompt" in result["final_report"]
    assert "without updating weights" in result["final_report"]
    assert "local://extra" not in result["final_report"]
    assert result["report_quality_status"] == "coverage_audited_pass"
    assert len(model.prompts) == 3  # plan, write, joint support/completeness check


def test_missing_answer_is_added_and_independently_verified():
    draft = "# Reflexion\n\n" + FEEDBACK
    check = complete_check(draft)
    candidate = coverage.local_repair_candidate(
        draft, check, plan(), questions(), {"ev-1", "ev-2", "ev-3"}
    )
    model = Responses(plan(), draft, check, complete_check(candidate))
    result = run_pipeline(model)
    assert "without updating weights" in result["final_report"]
    assert "Evidence gaps" not in result["final_report"]
    assert len(model.prompts) == 4


def test_unknown_marker_repairs_only_affected_block():
    bad = "Stored reflections enter the next actor prompt [EVIDENCE:invented]."
    draft = "# Reflexion\n\n" + bad + "\n\n" + WEIGHTS
    check = complete_check(draft)
    check.blocks[0].evidence_ids = ["ev-1"]
    candidate = coverage.local_repair_candidate(
        draft, check, plan(), questions(), {"ev-1", "ev-2", "ev-3"}
    )
    model = Responses(plan(), draft, check, complete_check(candidate))
    result = run_pipeline(model)
    assert "invented" not in result["final_report"]
    assert "without updating weights" in result["final_report"]
    assert "local://paper" in result["final_report"]


def test_failed_repair_verification_keeps_only_previously_verified_blocks():
    draft = "# Reflexion\n\n" + FEEDBACK
    model = Responses(plan(), draft, complete_check(draft), RuntimeError("unavailable"))
    result = run_pipeline(model)
    assert "Stored reflections" in result["final_report"]
    assert "without updating weights" not in result["final_report"]
    assert "Are weights updated?" in result["final_report"]
    assert result["report_quality_status"] == "degraded_needs_review"


def test_rejected_addition_cannot_be_resurrected_after_audit():
    draft = "# Reflexion\n\n" + FEEDBACK
    first = complete_check(draft)
    candidate = coverage.local_repair_candidate(
        draft, first, plan(), questions(), {"ev-1", "ev-2", "ev-3"}
    )
    final = complete_check(candidate)
    final.blocks[-1].verdict = "UNSUPPORTED"
    model = Responses(plan(), draft, first, final)
    result = run_pipeline(model)
    assert "without updating weights" not in result["final_report"]
    assert "Evidence gaps" in result["final_report"]
    assert result["report_quality_status"] == "degraded_needs_review"


@pytest.mark.parametrize(
    "mode", ["missing", "duplicate", "unknown_evidence", "uncited"]
)
def test_incomplete_or_invalid_support_judgments_fail_closed(mode):
    report = FEEDBACK
    check = complete_check(report)
    if mode == "missing":
        check.blocks = []
    elif mode == "duplicate":
        check.blocks.append(check.blocks[0])
    elif mode == "unknown_evidence":
        check.blocks[0].evidence_ids = ["ev-invented"]
    else:
        report = "Stored reflections enter the next actor prompt."
    safe, removed = coverage.filter_verified_blocks(report, check, {"ev-1"})
    assert safe == ""
    assert removed == 1


def test_related_facts_or_fabricated_excerpt_are_not_complete_answers():
    check = coverage.DraftCheck(
        questions=[
            coverage.QuestionCheck(
                question_id="q1", status="ANSWERED", answer_excerpt="Not in report"
            ),
            coverage.QuestionCheck(
                question_id="q2", status="PARTIAL", answer_excerpt=WEIGHTS
            ),
        ]
    )
    assert (
        coverage.covered_questions(FEEDBACK + "\n\n" + WEIGHTS, check, questions())
        == []
    )


def test_plan_cannot_hide_omitted_question_or_admit_unknown_evidence():
    bad = coverage.CoveragePlan(
        title="Report",
        answers=[
            coverage.PlannedAnswer(
                question_id="q1",
                status="SUPPORTED",
                answer="Fabricated answer",
                evidence_ids=["invented"],
            )
        ],
    )
    model = Responses(bad)
    result = asyncio.run(
        coverage.plan_answers(
            model,
            questions(),
            evidence_records(),
            native=False,
            retries=1,
            config={"configurable": {"observability_enabled": False}},
        )
    )
    assert [a.status for a in result.answers] == ["INSUFFICIENT", "INSUFFICIENT"]
    assert [a.question_id for a in result.answers] == ["q1", "q2"]


def test_question_decomposition_sees_only_user_request():
    model = Responses(
        coverage.QuestionList(
            questions=["How is feedback used?", "Are weights updated?"]
        )
    )
    requirements = asyncio.run(
        coverage.decompose_questions(
            model,
            "Explain Reflexion and its parameter updates",
            native=False,
            retries=1,
            config={"configurable": {"observability_enabled": False}},
        )
    )
    assert [q.question_id for q in requirements] == ["q1", "q2"]
    assert "Reflections are stored in memory" not in model.prompts[0]
    assert "reference answers" in model.prompts[0]


def test_targeted_search_reuses_same_question_ids_and_checks_gap_again():
    partial = plan()
    partial.answers[1] = coverage.PlannedAnswer(
        question_id="q2", status="INSUFFICIENT", search_query="Reflexion weight updates"
    )
    calls = []

    async def search(gaps):
        calls.append(gaps)
        return True

    draft = "# Reflexion\n\n" + FEEDBACK + "\n\n" + WEIGHTS
    model = Responses(partial, plan(), draft, complete_check(draft))
    result = run_pipeline(model, gap_search=search, evidence=evidence_records()[:1])
    assert calls == [["Reflexion weight updates"]]
    assert result["report_quality_status"] == "coverage_audited_pass"
    assert '"q2"' in model.prompts[1]


def test_unanswerable_question_remains_explicit_instead_of_getting_guessed():
    partial = plan()
    partial.answers[1] = coverage.PlannedAnswer(question_id="q2", status="INSUFFICIENT")
    draft = "# Reflexion\n\n" + FEEDBACK
    model = Responses(partial, draft, complete_check(draft))
    result = run_pipeline(model)
    assert "Evidence gaps" in result["final_report"]
    assert "Are weights updated?" in result["final_report"]
    assert result["report_quality_status"] == "incomplete_evidence"


def test_budget_exhaustion_is_propagated_not_silently_reported_as_success():
    model = Responses()
    with pytest.raises(BudgetExceeded):
        run_pipeline(model, budget=RunBudget(max_llm_calls=1, max_prompt_tokens=1))
    assert model.prompts == []


def test_new_profile_runs_in_final_graph_node_without_legacy_plan(
    tmp_path, monkeypatch
):
    store = RuntimeStore(tmp_path / "runtime.db")
    for index, item in enumerate(evidence_records(), 1):
        doc_id = f"doc-{index}"
        store.save_document(
            Document(
                document_id=doc_id,
                url=item["source_url"],
                title=item["source_title"],
                content_hash=doc_id,
                retrieved_at="2026-10-21T00:00:00Z",
            )
        )
        store.save_evidence(
            Evidence(
                evidence_id=item["evidence_id"],
                task_id="research:1",
                document_id=doc_id,
                excerpt=item["excerpt"],
                claim_text=item["excerpt"],
                confidence=0.9,
            )
        )
    draft = "# Reflexion\n\n" + FEEDBACK + "\n\n" + WEIGHTS
    model = Responses(plan(), draft, complete_check(draft))
    monkeypatch.setattr(researcher, "configurable_model", model)
    result = asyncio.run(
        researcher.final_report_generation(
            {
                "messages": [HumanMessage(content="Explain Reflexion")],
                "notes": [],
                "research_brief": "Explain Reflexion",
                "answer_requirements": [q.model_dump() for q in questions()],
            },
            {
                "configurable": {
                    "enable_question_coverage": True,
                    "enable_structured_report": True,
                    "enable_evidence_context": True,
                    "enable_gap_search": False,
                    "native_structured_output": False,
                    "final_report_model": "openai:fake",
                    "runtime_db_path": str(store.path),
                    "event_log_path": str(tmp_path / "events.jsonl"),
                }
            },
        )
    )
    assert result["report_quality_status"] == "coverage_audited_pass"
    assert "local://paper" in result["final_report"]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["event_type"] == "question_coverage_finished" for e in events)
    assert not any(e["event_type"] == "report_draft_grounding_fallback" for e in events)


def test_replay_reads_database_without_modifying_original(tmp_path):
    import hashlib
    from experiments.runners.replay_coverage import load_evidence_readonly

    database = tmp_path / "runtime.db"
    store = RuntimeStore(database)
    store.save_document(
        Document(
            document_id="doc",
            url="local://paper",
            title="Paper",
            content_hash="hash",
            retrieved_at="2026-10-21T00:00:00Z",
        )
    )
    store.save_evidence(
        Evidence(
            evidence_id="ev-1",
            task_id="research:1",
            document_id="doc",
            excerpt="Evidence",
            claim_text="Evidence",
            confidence=0.9,
        )
    )
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    records = load_evidence_readonly(database)
    assert records == store.list_evidence()
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_coverage_runner_requires_semantic_check_and_allows_explicit_abstention():
    from experiments.runners.run_baseline import grounding_acceptance_error

    grounding = {
        "accepted": True,
        "source_precision": None,
        "cited_sources": 0,
        "unknown_evidence_ids": [],
        "unsupported_sources": [],
    }
    assert grounding_acceptance_error("coverage_v1", grounding) is not None
    audit = {
        "requirements": 2,
        "all_retained_blocks_verified": True,
        "verified_blocks": 0,
    }
    assert grounding_acceptance_error("coverage_v1", grounding, audit) is None


def test_empty_headings_are_removed_without_losing_populated_sections():
    report = "# Report\n\n## Empty\n\n## Populated\n\n" + FEEDBACK + "\n\n## Also empty"
    result = coverage.prune_empty_sections(report)
    assert "# Report" in result and "## Populated" in result
    assert FEEDBACK in result
    assert "## Empty" not in result and "## Also empty" not in result


def test_report_explicitly_declines_to_invent_an_unresolved_answer():
    partial = plan()
    partial.answers[1] = coverage.PlannedAnswer(question_id="q2", status="INSUFFICIENT")
    model = Responses(partial, FEEDBACK, complete_check(FEEDBACK))
    result = run_pipeline(model)
    assert (
        "cannot supply the requested conclusion or value without inventing it"
        in result["final_report"]
    )
    assert result["report_quality_status"] == "incomplete_evidence"


def test_initial_research_receives_atomic_requirements_without_reference_answers(
    monkeypatch,
):
    from open_deep_research.state import ResearchQuestion

    model = Responses(
        ResearchQuestion(research_brief="Explain the feedback mechanism"),
        coverage.QuestionList(
            questions=["How does feedback affect later actions?"],
            constraints=["Use only local evidence"],
        ),
    )
    monkeypatch.setattr(researcher, "configurable_model", model)
    command = asyncio.run(
        researcher.write_research_brief(
            {
                "messages": [
                    HumanMessage(content="Use only local evidence to explain feedback")
                ]
            },
            {
                "configurable": {
                    "enable_question_coverage": True,
                    "native_structured_output": False,
                    "research_model": "openai:fake",
                    "observability_enabled": False,
                }
            },
        )
    )
    requirements = command.update["answer_requirements"]
    assert requirements == [
        {"question_id": "q1", "question": "How does feedback affect later actions?"}
    ]
    assert (
        "q1: How does feedback affect later actions?"
        in command.update["research_brief"]
    )
    assert "Use only local evidence to explain feedback" in model.prompts[1]
    assert "EVIDENCE CARDS" not in model.prompts[1]


@pytest.mark.parametrize(
    "exit_type", ["plain_text", "premature_complete", "thinking_limit"]
)
def test_coverage_profile_cannot_skip_all_retrieval(tmp_path, monkeypatch, exit_type):
    class ResearchTask:
        async def ainvoke(self, _state, _config):
            return {"compressed_research": "Retrieved evidence", "raw_notes": []}

    database = tmp_path / "runtime.db"
    store = RuntimeStore(database)
    monkeypatch.setattr(researcher, "researcher_subgraph", ResearchTask())
    message = AIMessage(content="Done")
    if exit_type == "premature_complete":
        message = AIMessage(
            content="",
            tool_calls=[{"id": "done", "name": "ResearchComplete", "args": {}}],
        )
    elif exit_type == "thinking_limit":
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "think",
                    "name": "think_tool",
                    "args": {"reflection": "Need research"},
                }
            ],
        )
    command = asyncio.run(
        researcher.supervisor_tools(
            {
                "supervisor_messages": [message],
                "research_brief": "Compare task commits and API side effects",
                "research_iterations": 2 if exit_type == "thinking_limit" else 1,
                "research_units_started": 0,
                "raw_notes": [],
            },
            {
                "configurable": {
                    "enable_question_coverage": True,
                    "experiment_profile": "coverage_v1",
                    "runtime_db_path": str(database),
                    "observability_enabled": False,
                    "search_api": "local",
                    "local_corpus_path": str(tmp_path),
                    "max_total_research_units": 2,
                    "max_initial_research_units": 1,
                    "max_researcher_iterations": 1,
                }
            },
        )
    )
    assert store.task_counts() == (1, 1)
    assert command.goto == "supervisor"


def test_feedback_repair_does_not_resurrect_rejected_plan_and_uses_new_evidence():
    bad = 'Memory guarantees success [EVIDENCE:ev-1].'
    draft = bad + '\n\n' + WEIGHTS
    check = complete_check(draft)
    check.blocks[0].verdict = 'UNSUPPORTED'
    check.blocks[0].reason = 'No guarantee is established.'
    stale = plan()
    stale.answers[0].answer = bad
    edits = coverage.CoverageEdits(edits=[coverage.BlockEdit(block_id='b1', replacement=FEEDBACK)])
    repaired = FEEDBACK + '\n\n' + WEIGHTS
    searches = []

    async def search(gaps):
        searches.append(gaps)
        return True

    model = Responses(stale, draft, check, edits, complete_check(repaired))
    result = run_pipeline(model, feedback=True, gap_search=search)
    assert len(searches) == 1
    assert 'No guarantee' in str(searches)
    assert 'guarantees success' not in result['final_report']
    assert result['report_quality_status'] == 'coverage_audited_pass'
    assert 'earlier planned answers' in model.prompts[3]


def test_feedback_repair_uses_second_round_without_relaxing_audit():
    bad = 'Memory guarantees success [EVIDENCE:ev-1].'
    draft = bad + '\n\n' + WEIGHTS
    check = complete_check(draft)
    check.blocks[0].verdict = 'PARTIAL'
    patch = coverage.CoverageEdits(edits=[coverage.BlockEdit(block_id='b1', replacement=FEEDBACK)])
    repaired = FEEDBACK + '\n\n' + WEIGHTS
    still_bad = complete_check(repaired)
    still_bad.blocks[0].verdict = 'PARTIAL'
    narrower = FEEDBACK.replace('Stored reflections', 'Stored verbal reflections')
    patch2 = coverage.CoverageEdits(edits=[coverage.BlockEdit(block_id='b1', replacement=narrower)])
    final_check = complete_check(narrower + '\n\n' + WEIGHTS)
    final_check.questions[0].status = 'ANSWERED'
    final_check.questions[0].answer_excerpt = narrower
    model = Responses(plan(), draft, check, patch, still_bad, patch2, final_check)
    result = run_pipeline(model, feedback=True)
    assert result['report_quality_status'] == 'coverage_audited_pass'
    assert 'Stored verbal reflections' in result['final_report']
    assert len(model.prompts) == 7


def test_feedback_omitted_or_invalid_edit_does_not_hide_failed_block():
    bad = 'Memory guarantees success [EVIDENCE:ev-1].'
    draft = bad + '\n\n' + WEIGHTS
    check = complete_check(draft)
    check.blocks[0].verdict = 'UNSUPPORTED'
    for replacement in ('', FEEDBACK.replace('ev-1', 'invented')):
        edits = coverage.CoverageEdits(edits=[coverage.BlockEdit(block_id='b1', replacement=replacement)])
        model = Responses(plan(), draft, check, edits)
        result = run_pipeline(model, feedback=True)
        assert result['report_quality_status'] == 'degraded_needs_review'
        assert 'guarantees success' not in result['final_report']
        assert 'without updating weights' in result['final_report']


def test_feedback_citation_only_repair_preserves_prose_and_gets_rechecked():
    draft = FEEDBACK + '\n\n' + WEIGHTS
    check = complete_check(draft)
    check.blocks[0].evidence_ids = ['ev-1', 'ev-3']
    repaired = coverage.apply_feedback_edits(draft, check, questions(), coverage.CoverageEdits(),
                                             {'ev-1', 'ev-2', 'ev-3'})
    verification = complete_check(repaired)
    verification.questions[0].status = 'ANSWERED'
    verification.questions[0].answer_excerpt = coverage.report_blocks(repaired)[0][1]
    model = Responses(plan(), draft, check, verification)
    result = run_pipeline(model, feedback=True)
    assert result['report_quality_status'] == 'coverage_audited_pass'
    assert '[3]' in result['final_report'] or 'local://extra' in result['final_report']
    assert len(model.prompts) == 4


def test_json_format_retry_is_bounded_and_budgeted():
    from open_deep_research.utils import ainvoke_structured_with_fallback
    budget = RunBudget(max_llm_calls=2)
    model = Responses('not JSON', plan())
    config = {'configurable': {'structured_json_repair_attempts': 1,
                               'run_budget': budget, 'observability_enabled': False}}
    result = asyncio.run(ainvoke_structured_with_fallback(
        model, coverage.CoveragePlan, [HumanMessage(content='Plan')],
        native_structured_output=False, config=config))
    assert result.title == 'Reflexion'
    assert budget.snapshot().llm_calls == 2
    model = Responses('not JSON', plan())
    config['configurable']['run_budget'] = RunBudget(max_llm_calls=1)
    with pytest.raises(BudgetExceeded):
        asyncio.run(ainvoke_structured_with_fallback(
            model, coverage.CoveragePlan, [HumanMessage(content='Plan')],
            native_structured_output=False, config=config))
    assert len(model.prompts) == 1
