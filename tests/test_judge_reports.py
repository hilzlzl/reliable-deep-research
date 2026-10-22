import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "evaluation" / "judge_reports.py"
SPEC = importlib.util.spec_from_file_location("judge_reports", MODULE_PATH)
judge_reports = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(judge_reports)


def test_truncate_report_preserves_beginning_and_sources():
    report = "BEGIN" + "x" * 200 + "SOURCES"

    truncated = judge_reports.truncate_report(report, 100)

    assert len(truncated) == 100
    assert truncated.startswith("BEGIN")
    assert truncated.endswith("SOURCES")
    assert "middle omitted" in truncated
