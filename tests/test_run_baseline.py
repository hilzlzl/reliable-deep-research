import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "runners" / "run_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_baseline", MODULE_PATH)
run_baseline = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(run_baseline)


def test_load_cases_accepts_researchbench_schema(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"case_id":"cad_001","question":"Compare systems.","domain":"cad_generation"}\n',
        encoding="utf-8",
    )

    cases = run_baseline.load_cases(path, {"cad_001"})

    assert cases[0]["id"] == "cad_001"
    assert cases[0]["prompt"] == "Compare systems."
    assert cases[0]["category"] == "cad_generation"


def test_review_metrics_reads_last_review_event(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event_type":"report_review_finished","data":{"final_score":0.9,"stop_reason":"max_rounds","rounds":[{"round_number":1}]}}\n',
        encoding="utf-8",
    )

    result = run_baseline.review_metrics(path)

    assert result["final_score"] == 0.9
    assert result["stop_reason"] == "max_rounds"
    assert result["status"] == "succeeded"


def test_review_metrics_records_graceful_fallback(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event_type":"report_review_failed","data":{"error_type":"BudgetExceeded","fallback":"original_report"}}\n',
        encoding="utf-8",
    )

    result = run_baseline.review_metrics(path)

    assert result == {
        "status": "failed",
        "error_type": "BudgetExceeded",
        "fallback": "original_report",
    }


def test_report_grounding_metrics_combines_last_pipeline_events(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event_type":"report_plan_created","data":{"sections":2,"planned_claims":4,"remaining_gaps":[]}}',
                '{"event_type":"report_citations_rendered","data":{"cited_evidence_ids":["ev-1"],"unknown_evidence_ids":[],"cited_sources":1}}',
                '{"event_type":"report_source_audited","data":{"source_precision":1.0,"unsupported_sources":[]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_baseline.report_grounding_metrics(path)

    assert result["planned_claims"] == 4
    assert result["cited_evidence_ids"] == ["ev-1"]
    assert result["source_precision"] == 1.0
    assert result["accepted"] is True


def test_v2_grounding_error_rejects_missing_metrics():
    error = run_baseline.grounding_acceptance_error("reliable_v2", None)

    assert error["type"] == "GroundingAcceptanceError"


def test_v2_grounding_error_accepts_complete_contract():
    grounding = {
        "accepted": True,
        "claim_evidence_link_rate": 1.0,
        "citation_coverage": 1.0,
        "source_precision": 1.0,
        "unknown_evidence_ids": [],
        "unsupported_sources": [],
    }

    assert run_baseline.grounding_acceptance_error("reliable_v2", grounding) is None
    assert run_baseline.grounding_acceptance_error("adaptive_v3", grounding) is None


def test_adaptive_research_and_sentence_audit_metrics(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event_type":"adaptive_research_round_finished","data":{"round":1,"actions":[{"action":"SEARCH_LOCAL"}],"evidence_gain":2}}',
                '{"event_type":"adaptive_research_finished","data":{"stop_reason":"no_positive_utility_action","objectives":2,"decisions":1}}',
                '{"event_type":"report_sentence_audit_finished","data":{"total_assertions":4,"supported":3,"support_rate":0.75,"repairs_applied":1}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    adaptive = run_baseline.adaptive_research_metrics(path)
    audit = run_baseline.sentence_audit_metrics(path)

    assert adaptive["actions"] == 1
    assert adaptive["evidence_gain"] == 2
    assert adaptive["stop_reason"] == "no_positive_utility_action"
    assert audit["support_rate"] == 0.75
    assert audit["repairs_applied"] == 1


def test_final_report_budget_error_detects_hidden_fallback(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event_type":"run_budget_exhausted","node":"final_report_generation",'
        '"data":{"resource":"llm_calls","limit":22}}\n',
        encoding="utf-8",
    )

    assert run_baseline.final_report_budget_error(path) == {
        "type": "BudgetExceeded",
        "message": "Final report generation exhausted llm_calls",
    }


def test_adaptive_retrieval_metrics_preserves_decision_reasons(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event_type":"retrieval_route_selected","data":{"action":"accept_local","query_term_coverage":1.0,"reasons":[]}}',
                '{"event_type":"retrieval_route_selected","data":{"action":"escalate_web","query_term_coverage":0.5,"reasons":["source_diversity_below_minimum:1<2"]}}',
                '{"event_type":"retrieval_web_escalation_succeeded","data":{}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_baseline.adaptive_retrieval_metrics(path)

    assert result["decisions"] == 2
    assert result["local_accepts"] == 1
    assert result["web_escalation_requests"] == 1
    assert result["web_escalations_succeeded"] == 1
    assert result["mean_query_term_coverage"] == 0.75
    assert result["decision_trace"][1]["reasons"] == [
        "source_diversity_below_minimum:1<2"
    ]


def test_model_routing_metrics_counts_provider_neutral_tiers(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event_type":"model_route_selected","node":"researcher","data":{"tier":"local_deployment","model":"local","reason":"local_first_node"}}',
                '{"event_type":"model_route_selected","node":"final_report_generation","data":{"tier":"paid_api","model":"paid","reason":"high_risk_node"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_baseline.model_routing_metrics(path)

    assert result["selections"] == 2
    assert result["tier_counts"] == {"local_deployment": 1, "paid_api": 1}
