"""Tests for the local event log used by baseline and recovery experiments."""

import asyncio
import json

import pytest

from open_deep_research.observability import observed_node, record_event


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_event_uses_thread_id_as_run_id(tmp_path):
    path = tmp_path / "events.jsonl"
    config = {"configurable": {"thread_id": "run-123", "event_log_path": str(path)}}

    record_event(config, "run_started", "runtime", data={"case_id": "case-1"})

    event = read_events(path)[0]
    assert event["run_id"] == "run-123"
    assert event["event_type"] == "run_started"
    assert event["data"] == {"case_id": "case-1"}


def test_observed_node_records_success(tmp_path):
    path = tmp_path / "events.jsonl"
    config = {"configurable": {"run_id": "success", "event_log_path": str(path)}}

    async def node(state, config):
        return {"answer": state["question"]}

    result = asyncio.run(observed_node("demo", node)({"question": "why"}, config))
    events = read_events(path)

    assert result == {"answer": "why"}
    assert [event["event_type"] for event in events] == ["node_started", "node_succeeded"]
    assert events[1]["data"]["result_keys"] == ["answer"]


def test_observed_node_records_and_reraises_failure(tmp_path):
    path = tmp_path / "events.jsonl"
    config = {"configurable": {"run_id": "failure", "event_log_path": str(path)}}

    async def node(state, config):
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        asyncio.run(observed_node("demo", node)({}, config))

    events = read_events(path)
    assert events[-1]["event_type"] == "node_failed"
    assert events[-1]["data"]["error_type"] == "RuntimeError"


def test_observability_can_be_disabled(tmp_path):
    path = tmp_path / "events.jsonl"
    config = {
        "configurable": {
            "observability_enabled": False,
            "event_log_path": str(path),
        }
    }

    record_event(config, "run_started", "runtime")

    assert not path.exists()
