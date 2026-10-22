import asyncio
from types import SimpleNamespace

import open_deep_research.utils as utils
from open_deep_research.adaptive_retrieval import RetrievalCandidate
from open_deep_research.run_budget import BudgetExceeded


def ranked(document_id: str, title: str, text: str, index: int):
    chunk = SimpleNamespace(
        document_id=document_id,
        title=title,
        text=text,
        chunk_id=f"chunk-{index}",
        source_url=f"local://{document_id}",
    )
    return SimpleNamespace(chunk=chunk, score=1.0)


class FakeWebSearch:
    def __init__(self):
        self.calls = []

    async def __call__(self, queries, *, max_results, topic, config=None):
        self.calls.append((queries, max_results, topic, config))
        return [
            RetrievalCandidate(
                document_id="web-deepcad",
                chunk_id="web-deepcad:0",
                source="web",
                sources=["web"],
                url="https://example.org/deepcad",
                title="DeepCAD",
                text="Verified web evidence about DeepCAD command representation.",
                rank=1,
            )
        ]


class ExhaustedWebSearch:
    async def __call__(self, queries, *, max_results, topic, config=None):
        raise BudgetExceeded("search_calls", 1)


def configuration(**overrides):
    values = {
        "search_api": "hybrid",
        "observability_enabled": False,
        "hybrid_local_min_results": 2,
        "hybrid_local_min_unique_documents": 2,
        "hybrid_local_min_query_term_coverage": 0.75,
    }
    values.update(overrides)
    return {"configurable": values}


def test_hybrid_search_keeps_sufficient_local_results(monkeypatch):
    async def local_results(*_args):
        return [
            ranked("deepcad", "DeepCAD", "DeepCAD command representation", 1),
            ranked(
                "sketchgraphs",
                "SketchGraphs",
                "SketchGraphs graph representation",
                2,
            ),
        ]

    web = FakeWebSearch()
    monkeypatch.setattr(utils, "_rank_local_chunks", local_results)
    monkeypatch.setattr(utils, "_retrieve_tavily_candidates", web)

    result = asyncio.run(
        utils.adaptive_hybrid_search.ainvoke(
            {"queries": ["DeepCAD SketchGraphs representation"]},
            config=configuration(),
        )
    )

    assert "LOCAL SOURCE" in result
    assert "ADAPTIVE WEB SUPPLEMENT" not in result
    assert web.calls == []


def test_hybrid_search_escalates_only_after_local_quality_gate(monkeypatch):
    async def local_results(*_args):
        return [
            ranked(
                "sketchgraphs",
                "SketchGraphs",
                "SketchGraphs geometric constraints",
                1,
            )
        ]

    web = FakeWebSearch()
    monkeypatch.setattr(utils, "_rank_local_chunks", local_results)
    monkeypatch.setattr(utils, "_retrieve_tavily_candidates", web)

    result = asyncio.run(
        utils.adaptive_hybrid_search.ainvoke(
            {"queries": ["DeepCAD SketchGraphs representation"]},
            config=configuration(),
        )
    )

    assert "HYBRID SOURCE" in result
    assert "Verified web evidence" in result
    assert len(web.calls) == 1


def test_hybrid_search_does_not_hide_budget_exhaustion(monkeypatch):
    async def local_results(*_args):
        return []

    monkeypatch.setattr(utils, "_rank_local_chunks", local_results)
    monkeypatch.setattr(utils, "_retrieve_tavily_candidates", ExhaustedWebSearch())

    try:
        asyncio.run(
            utils.adaptive_hybrid_search.ainvoke(
                {"queries": ["missing evidence"]},
                config=configuration(),
            )
        )
    except BudgetExceeded as error:
        assert error.resource == "search_calls"
    else:
        raise AssertionError("Budget exhaustion must propagate to the reliable runtime")


def test_hybrid_search_stops_after_repeated_low_gain_rounds(monkeypatch, tmp_path):
    async def local_results(*_args):
        return [ranked("local", "Local", "partial evidence", 1)]

    event_path = tmp_path / "events.jsonl"
    event_path.write_text(
        "\n".join(
            [
                '{"run_id":"run-1","event_type":"retrieval_marginal_gain","data":{"useful":false}}',
                '{"run_id":"run-1","event_type":"retrieval_marginal_gain","data":{"useful":false}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    web = FakeWebSearch()
    monkeypatch.setattr(utils, "_rank_local_chunks", local_results)
    monkeypatch.setattr(utils, "_retrieve_tavily_candidates", web)
    config = configuration(
        observability_enabled=True,
        event_log_path=str(event_path),
        run_id="run-1",
        hybrid_max_consecutive_low_gain=2,
    )

    result = asyncio.run(
        utils.adaptive_hybrid_search.ainvoke(
            {"queries": ["missing independent evidence"]}, config=config
        )
    )

    assert "WEB SEARCH STOPPED" in result
    assert web.calls == []
