"""Tests for structured-output degradation on OpenAI-compatible gateways."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from open_deep_research.utils import ainvoke_structured_with_fallback


class Answer(BaseModel):
    answer: str


def test_structured_fallback_parses_json_text():
    calls = 0

    class FakeModel:
        def with_structured_output(self, _schema):
            return self

        def with_retry(self, **_kwargs):
            return self

        async def ainvoke(self, _messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("gateway rejected json schema")
            return AIMessage(content='```json\n{"answer":"OK"}\n```')

    model = FakeModel()
    result = asyncio.run(
        ainvoke_structured_with_fallback(model, Answer, [HumanMessage(content="reply")], retries=1)
    )

    assert result.answer == "OK"
    assert calls == 2


def test_plain_json_mode_skips_native_structured_output():
    class FakeModel:
        def with_structured_output(self, _schema):
            raise AssertionError("native structured output should be skipped")

        async def ainvoke(self, _messages):
            return AIMessage(content='{"answer":"OK"}')

    result = asyncio.run(
        ainvoke_structured_with_fallback(
            FakeModel(),
            Answer,
            [HumanMessage(content="reply")],
            native_structured_output=False,
        )
    )

    assert result.answer == "OK"
