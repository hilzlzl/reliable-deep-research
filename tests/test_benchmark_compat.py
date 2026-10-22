import json

from open_deep_research.benchmark_compat import (
    ExpectedFact,
    FactBenchmarkCase,
    evaluate_fact_retrieval_ablation,
    evaluate_text_facts,
    import_fact_benchmark,
)
from open_deep_research.local_benchmark import read_jsonl, CorpusChunk


def test_import_normalizes_external_schema_and_corpus(tmp_path):
    dataset = tmp_path / "source.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "domain": "rag",
                "difficulty": "easy",
                "question": "What does hybrid retrieval combine?",
                "expected_facts": [
                    {"fact": "It combines dense and sparse retrieval", "keywords": ["dense", "sparse"]}
                ],
                "required_citations": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "rag_overview.md").write_text(
        "# RAG overview\n\nHybrid search combines dense and sparse retrieval.", encoding="utf-8"
    )

    manifest = import_fact_benchmark(dataset, corpus, tmp_path / "output", chunk_size=200, overlap=20)

    assert manifest["cases"] == 1
    assert manifest["documents"] == 1
    cases = read_jsonl(tmp_path / "output" / "cases.jsonl", FactBenchmarkCase)
    assert cases[0].case_id == "case-1"
    assert cases[0].expected_facts[0].keywords == ["dense", "sparse"]


def test_fact_retrieval_scores_english_and_chinese_queries(tmp_path):
    dataset = tmp_path / "source.jsonl"
    rows = [
        {
            "id": "en",
            "domain": "rag",
            "difficulty": "easy",
            "question": "What combines dense and sparse retrieval?",
            "expected_facts": [{"fact": "Hybrid search combines dense and sparse retrieval", "keywords": ["hybrid search"]}],
            "required_citations": 1,
        },
        {
            "id": "zh",
            "domain": "rag",
            "difficulty": "easy",
            "question": "混合检索结合了什么？",
            "expected_facts": [{"fact": "混合检索结合密集检索和稀疏检索", "keywords": ["混合检索"]}],
            "required_citations": 1,
            "question_lang": "zh",
        },
    ]
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "rag_en.md").write_text("# Hybrid search\nDense and sparse retrieval are combined.", encoding="utf-8")
    (corpus / "rag_zh.md").write_text("# 混合检索\n混合检索结合密集检索和稀疏检索。", encoding="utf-8")
    output = tmp_path / "output"
    import_fact_benchmark(dataset, corpus, output, chunk_size=200, overlap=20)
    cases = read_jsonl(output / "cases.jsonl", FactBenchmarkCase)
    chunks = read_jsonl(output / "chunks.jsonl", CorpusChunk)

    result = evaluate_fact_retrieval_ablation(cases, chunks, top_k=1, candidate_k=2)

    assert result["methods"]["bm25"]["aggregate"]["fact_recall_at_k"] == 1.0
    assert result["methods"]["bm25"]["by_language"]["zh"]["fact_recall_at_k"] == 1.0


def test_fact_case_accepts_legacy_string_fact_after_adapter():
    case = FactBenchmarkCase(
        case_id="case",
        question="Question",
        domain="test",
        difficulty="easy",
        expected_facts=[ExpectedFact(fact="A fact")],
        required_citations=0,
    )
    assert case.expected_facts[0].fact == "A fact"


def test_generated_report_fact_coverage_is_explicitly_weak_label():
    facts = [
        ExpectedFact(fact="Hybrid search combines dense and sparse retrieval", keywords=["hybrid search"]),
        ExpectedFact(fact="RRF combines ranked lists", aliases=["reciprocal rank fusion"]),
    ]

    result = evaluate_text_facts("Hybrid search is useful, but no fusion method is named.", facts)

    assert result == {"fact_count": 2, "fact_hits": 1, "fact_coverage": 0.5}
