"""Checks for the paired report grounding experiment's measurement contract."""

from experiments.evaluation.evaluate_report_grounding import (
    cited_claim_candidates,
    corpus_index,
    source_index,
    source_snippets,
)
from open_deep_research.local_benchmark import CorpusChunk


def test_numbered_references_are_resolved_without_sampling_bibliography():
    report = (
        "# Report\n\n"
        "The command sequence preserves construction order and parameters [1].\n\n"
        "The sketch graph records geometric constraints between primitives [2].\n\n"
        "## Sources\n\n"
        "[1] Commands: https://example.test/commands#chunk=doc:0\n"
        "2. Graphs: https://example.test/graphs\n"
    )

    claims, total = cited_claim_candidates(report, "case-1", 5)

    assert total == 2
    assert {url for claim in claims for url in claim["source_urls"]} == {
        "https://example.test/commands#chunk=doc:0",
        "https://example.test/graphs",
    }
    assert len(source_index(report)) == 2


def test_citation_snippet_uses_exact_chunk_and_exposes_missing_source():
    chunks = [
        CorpusChunk(
            chunk_id="doc:0",
            document_id="doc",
            title="Source",
            domain="test",
            source_url="https://example.test/commands",
            text="The command sequence preserves construction order.",
            start_character=0,
            end_character=50,
        ),
        CorpusChunk(
            chunk_id="doc:1",
            document_id="doc",
            title="Source",
            domain="test",
            source_url="https://example.test/commands",
            text="This chunk describes unrelated evaluation results.",
            start_character=50,
            end_character=100,
        ),
    ]
    by_id, by_url = corpus_index(chunks)
    claim = {
        "statement": "The sequence preserves construction order.",
        "source_urls": [
            "https://example.test/commands#chunk=doc:0",
            "https://example.test/unknown",
        ],
    }

    snippets = source_snippets(claim, by_id, by_url)

    assert [item["chunk_id"] for item in snippets[0]["chunks"]] == ["doc:0"]
    assert snippets[0]["in_fixed_corpus"] is True
    assert snippets[1]["in_fixed_corpus"] is False


def test_inline_markdown_source_links_are_sampled():
    report = (
        "The framework interleaves reasoning and actions within one trajectory "
        "[ReAct](https://example.test/react).\n\n"
        "## Sources\n\n"
        "[ReAct](https://example.test/react)\n"
    )

    claims, total = cited_claim_candidates(report, "case-2", 5)

    assert total == 1
    assert claims[0]["source_urls"] == ["https://example.test/react"]
