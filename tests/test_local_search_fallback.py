import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import open_deep_research.deep_researcher as researcher


def test_reliable_local_research_forces_one_search(monkeypatch):
    calls = []

    @tool
    async def local_corpus_search(queries: list[str]) -> str:
        """Search the fixed local corpus."""
        calls.append(queries)
        return "verified local evidence"

    async def fake_get_all_tools(_config):
        return [local_corpus_search]

    monkeypatch.setattr(researcher, "get_all_tools", fake_get_all_tools)
    config = {
        "configurable": {
            "enable_reliable_runtime": True,
            "search_api": "local",
            "observability_enabled": False,
        }
    }
    state = {
        "researcher_messages": [AIMessage(content="I can answer directly.")],
        "research_topic": "Compare DeepCAD and SketchGraphs",
        "tool_call_iterations": 1,
        "local_search_fallback_used": False,
    }

    command = asyncio.run(researcher.researcher_tools(state, config))

    assert command.goto == "compress_research"
    assert command.update["local_search_fallback_used"] is True
    assert command.update["search_performed"] is True
    assert command.update["fallback_evidence_context"] == "verified local evidence"
    assert "verified local evidence" in command.update["researcher_messages"][0].content
    assert calls == [["Compare DeepCAD and SketchGraphs"]]


def test_local_search_fallback_runs_only_once():
    config = {
        "configurable": {
            "enable_reliable_runtime": True,
            "search_api": "local",
            "observability_enabled": False,
        }
    }
    state = {
        "researcher_messages": [AIMessage(content="Research summary")],
        "research_topic": "topic",
        "tool_call_iterations": 2,
        "local_search_fallback_used": True,
        "search_performed": True,
    }

    command = asyncio.run(researcher.researcher_tools(state, config))

    assert command.goto == "compress_research"


def test_model_search_prevents_fallback(monkeypatch):
    async def unexpected_tools(_config):
        raise AssertionError("fallback search should not execute")

    monkeypatch.setattr(researcher, "get_all_tools", unexpected_tools)
    config = {
        "configurable": {
            "enable_reliable_runtime": True,
            "search_api": "local",
            "observability_enabled": False,
        }
    }
    state = {
        "researcher_messages": [AIMessage(content="Enough evidence")],
        "research_topic": "topic",
        "tool_call_iterations": 2,
        "local_search_fallback_used": False,
        "search_performed": True,
    }

    command = asyncio.run(researcher.researcher_tools(state, config))

    assert command.goto == "compress_research"
