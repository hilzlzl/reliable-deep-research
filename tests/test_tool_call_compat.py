"""Tests for native local-model tool-call compatibility parsing."""

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from open_deep_research.tool_call_compat import normalize_text_tool_calls


class LookupEvidence(BaseModel):
    """Look up evidence."""

    query: str = Field(min_length=1)


class SearchCorpus(BaseModel):
    """Search a corpus."""

    queries: list[str]


def test_parses_observed_gemma_native_call_text():
    message = AIMessage(content="call:LookupEvidence{query:CAD generation}")

    parsed = normalize_text_tool_calls(message, [LookupEvidence])

    assert parsed.content == ""
    assert parsed.tool_calls[0]["name"] == "LookupEvidence"
    assert parsed.tool_calls[0]["args"] == {"query": "CAD generation"}


def test_parses_array_arguments_and_rejects_unknown_tools():
    message = AIMessage(
        content=(
            "call:Unknown{query:ignored} "
            "call:SearchCorpus{queries:[DeepCAD, SketchGraphs]}"
        )
    )

    parsed = normalize_text_tool_calls(message, [SearchCorpus])

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["args"] == {
        "queries": ["DeepCAD", "SketchGraphs"]
    }


def test_preserves_provider_parsed_tool_calls():
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "LookupEvidence",
                "args": {"query": "CAD"},
                "id": "provider-1",
                "type": "tool_call",
            }
        ],
    )

    assert normalize_text_tool_calls(message, [LookupEvidence]) is message
