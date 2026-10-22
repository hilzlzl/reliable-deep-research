"""Check the public entry point's profile, input and spending boundaries offline."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from experiments.runners import run_baseline, run_research


def test_public_entry_delegates_to_coverage_v2_with_budgets(monkeypatch):
    seen = []

    async def capture(argv):
        seen.extend(argv)

    monkeypatch.setattr(run_baseline, "main", capture)
    run_research.main(["--question", "How do durable tasks recover?"])
    options = dict(zip(seen[::2], seen[1::2]))
    assert options["--profile"] == "coverage_v2"
    assert options["--question"] == "How do durable tasks recover?"
    assert options["--max-llm-calls"] == "40"
    assert options["--max-search-calls"] == "4"
    assert options["--max-task-attempts"] == "6"
    assert options["--final-report-max-tokens"] == "4096"


@pytest.mark.parametrize("argv", [
    [], ["--question", " "], ["--question", "q", "--case", "x"],
    ["--question", "q", "--max-llm-calls", "0"],
    ["--question", "q", "--max-total-seconds", "nan"],
    ["--question", "q", "--search-api", "local"],
    ["--question", "q", "--streaming"],
    ["--question", "q", "--base-url", "https://example.test/v1"],
    ["--question", "q", "--profile", "coverage_v2"],
])
def test_invalid_input_fails_before_dispatch(argv):
    with pytest.raises(SystemExit) as exc:
        run_research.parse_args(argv)
    assert exc.value.code == 2


def test_direct_question_reaches_existing_run_and_failure_exit(tmp_path, monkeypatch):
    captured = []

    async def fail_case(case, args, root):
        captured.append((case, args.profile))
        return {"case_id": case["id"], "status": "failed", "elapsed_seconds": 0}

    monkeypatch.setattr(run_baseline, "run_case", fail_case)
    with pytest.raises(SystemExit) as exc:
        asyncio.run(run_baseline.main([
            "--question", "A custom question", "--profile", "coverage_v2",
            "--output", str(tmp_path),
        ]))
    assert exc.value.code == 1
    assert captured == [({"id": "question", "category": "custom", "prompt": "A custom question"}, "coverage_v2")]
    assert len(list(tmp_path.glob("*/manifest.json"))) == 1


def test_release_entry_enables_repair_and_preserves_degraded_exit(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, state, config):
            return {"final_report": "Needs review", "report_quality_status": "degraded_needs_review"}

    monkeypatch.setattr(run_baseline, "deep_researcher_builder", SimpleNamespace(compile=lambda **kw: FakeGraph()))
    with pytest.raises(SystemExit) as exc:
        run_research.main(["--question", "A coverage question", "--output", str(tmp_path)])
    assert exc.value.code == 1
    summary = json.loads(next(tmp_path.glob("*/*/summary.json")).read_text())
    assert summary["status"] == "degraded"
    config = summary["configuration"]
    assert config["experiment_profile"] == "coverage_v2"
    assert config["enable_reliable_runtime"] is True
    assert config["enable_question_coverage"] is True
    assert config["enable_coverage_feedback_repair"] is True
    assert config["structured_json_repair_attempts"] == 1
    assert config["max_total_research_units"] == 4
    assert config["max_task_attempts"] == 6
    assert config["enable_claim_extraction"] is False
