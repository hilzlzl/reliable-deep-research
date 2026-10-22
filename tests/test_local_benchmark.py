import hashlib

import pytest

from open_deep_research.local_benchmark import (
    BM25Retriever,
    CorpusDocument,
    DenseVectorRetriever,
    GoldEvidence,
    LocalBenchmarkCase,
    SourceRecord,
    TfidfVectorRetriever,
    build_documents,
    chunk_documents,
    evaluate_retrieval,
    retrieve_with_strategy,
    run_retrieval_ablation,
    run_semantic_retrieval_ablation,
    retrieve_with_dense_strategy,
    validate_benchmark,
)


def make_document(document_id: str, title: str, content: str) -> CorpusDocument:
    return CorpusDocument(
        document_id=document_id,
        title=title,
        domain="test",
        source_type="paper",
        source_url=f"https://example.test/{document_id}",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def test_build_documents_reports_missing_sources_and_hashes_content(tmp_path):
    sources = [
        SourceRecord(
            source_id="present",
            title="Present",
            domain="test",
            source_type="paper",
            locator="https://example.test/present",
            acquisition_status="approved",
            redistribution="metadata_only",
        ),
        SourceRecord(
            source_id="missing",
            title="Missing",
            domain="test",
            source_type="paper",
            locator="https://example.test/missing",
            acquisition_status="approved",
            redistribution="metadata_only",
        ),
    ]
    (tmp_path / "present.md").write_text("Verified English content.", encoding="utf-8")

    documents, missing = build_documents(sources, tmp_path)

    assert [document.document_id for document in documents] == ["present"]
    assert missing == ["missing"]
    assert documents[0].content_sha256 == hashlib.sha256(
        b"Verified English content."
    ).hexdigest()


def test_build_rejects_content_that_has_not_passed_source_review(tmp_path):
    source = SourceRecord(
        source_id="candidate",
        title="Candidate",
        domain="test",
        source_type="paper",
        locator="https://example.test/candidate",
        acquisition_status="candidate_requires_resolution",
        redistribution="metadata_only",
    )
    (tmp_path / "candidate.md").write_text("Unreviewed content.", encoding="utf-8")

    with pytest.raises(ValueError, match="not approved"):
        build_documents([source], tmp_path)


def test_build_accepts_local_only_source_without_redistributing_it(tmp_path):
    source = SourceRecord(
        source_id="local-paper",
        title="Local paper",
        domain="test",
        source_type="paper",
        locator="https://example.test/local-paper",
        acquisition_status="approved_local_only",
        redistribution="metadata_only",
    )
    (tmp_path / "local-paper.txt").write_text("Local research use.", encoding="utf-8")

    documents, missing = build_documents([source], tmp_path)

    assert len(documents) == 1
    assert missing == []


def test_benchmark_validation_requires_verbatim_evidence_quote():
    document = make_document(
        "doc-a",
        "Reliable execution",
        "Durable task results allow completed work to be restored after interruption.",
    )
    valid_case = LocalBenchmarkCase(
        case_id="case-a",
        question="What can be restored after interruption?",
        domain="agent_systems",
        difficulty="single_document",
        gold_document_ids=["doc-a"],
        gold_evidence=[
            GoldEvidence(
                document_id="doc-a",
                quote="completed work to be restored after interruption",
            )
        ],
    )
    invalid_case = valid_case.model_copy(
        update={
            "case_id": "case-b",
            "gold_evidence": [GoldEvidence(document_id="doc-a", quote="invented quote")],
        }
    )

    assert validate_benchmark([document], [valid_case])["valid"]
    invalid = validate_benchmark([document], [invalid_case])
    assert not invalid["valid"]
    assert "evidence quote not found" in invalid["errors"][0]


def test_bm25_baseline_and_metrics_retrieve_gold_evidence():
    relevant = make_document(
        "cad-kernel",
        "Geometric kernel feedback",
        "A geometric kernel can detect failed boolean operations and invalid topology.",
    )
    distractor = make_document(
        "agent-memory",
        "Agent memory",
        "Agent memory stores observations across research steps.",
    )
    chunks = chunk_documents([relevant, distractor], chunk_size=200, overlap=20)
    case = LocalBenchmarkCase(
        case_id="cad-1",
        question="What detects failed boolean operations and invalid topology?",
        domain="cad_generation",
        difficulty="single_document",
        gold_document_ids=["cad-kernel"],
        gold_evidence=[
            GoldEvidence(
                document_id="cad-kernel",
                quote="detect failed boolean operations and invalid topology",
            )
        ],
    )

    ranked = BM25Retriever(chunks).retrieve(case.question, top_k=1)
    result = evaluate_retrieval([case], chunks, top_k=1)

    assert ranked[0].chunk.document_id == "cad-kernel"
    assert result["aggregate"] == {
        "document_recall_at_k": 1.0,
        "evidence_recall_at_k": 1.0,
        "reciprocal_rank": 1.0,
        "ndcg_at_k": 1.0,
        "source_diversity_at_k": 1.0,
    }
    assert 0.0 <= result["cases"][0]["ndcg_at_k"] <= 1.0


def test_unanswerable_cases_are_excluded_from_retrieval_average():
    document = make_document("doc-a", "Only source", "Some unrelated content.")
    chunks = chunk_documents([document], chunk_size=200, overlap=20)
    case = LocalBenchmarkCase(
        case_id="unanswerable",
        question="What universal guarantee exists?",
        domain="agent_systems",
        difficulty="unanswerable",
        answerable=False,
    )

    result = evaluate_retrieval([case], chunks, top_k=1)

    assert result["evaluated_cases"] == 0
    assert result["cases"][0]["answerable"] is False
    assert BM25Retriever(chunks).retrieve("universal guarantee", top_k=1) == []


def test_tfidf_vector_and_hybrid_ablation_are_deterministic():
    first = make_document(
        "doc-first",
        "Constraint graph",
        "A constraint graph connects geometric primitives using relational edges.",
    )
    second = make_document(
        "doc-second",
        "Command sequence",
        "A command sequence preserves ordered CAD construction operations.",
    )
    chunks = chunk_documents([first, second], chunk_size=200, overlap=20)
    case = LocalBenchmarkCase(
        case_id="comparison",
        question="Compare constraint graph primitives with ordered command sequence operations.",
        domain="cad_generation",
        difficulty="multi_document",
        gold_document_ids=["doc-first", "doc-second"],
        gold_evidence=[
            GoldEvidence(document_id="doc-first", quote="constraint graph connects geometric primitives"),
            GoldEvidence(document_id="doc-second", quote="command sequence preserves ordered CAD"),
        ],
    )

    vector = TfidfVectorRetriever(chunks).retrieve(case.question, top_k=2)
    first_run = run_retrieval_ablation([case], chunks, top_k=2, candidate_k=2)
    second_run = run_retrieval_ablation([case], chunks, top_k=2, candidate_k=2)

    assert {item.chunk.document_id for item in vector} == {"doc-first", "doc-second"}
    assert first_run == second_run
    assert set(first_run["methods"]) == {
        "bm25",
        "tfidf_vector",
        "rrf_hybrid",
        "rrf_mmr",
    }
    for result in first_run["methods"].values():
        assert 0.0 <= result["aggregate"]["ndcg_at_k"] <= 1.0
    for strategy in ("bm25", "tfidf_vector", "rrf_hybrid", "rrf_mmr"):
        ranked = retrieve_with_strategy(
            case.question,
            chunks,
            strategy=strategy,
            top_k=2,
            candidate_k=2,
        )
        assert len(ranked) == 2
        assert len({item.chunk.chunk_id for item in ranked}) == 2


def test_dense_semantic_ablation_uses_external_vectors():
    first = make_document("agent-doc", "Agent planning", "Planning and reflective memory.")
    second = make_document("cad-doc", "CAD kernel", "Geometry and topology validation.")
    chunks = chunk_documents([first, second], chunk_size=200, overlap=20)
    case = LocalBenchmarkCase(
        case_id="semantic-case",
        question="How is a solid checked for geometric validity?",
        domain="cad_generation",
        difficulty="single_document",
        gold_document_ids=["cad-doc"],
        gold_evidence=[GoldEvidence(document_id="cad-doc", quote="topology validation")],
    )
    chunk_vectors = [[1.0, 0.0], [0.0, 1.0]]
    query_vectors = {"semantic-case": [0.0, 1.0]}

    result = run_semantic_retrieval_ablation(
        [case],
        chunks,
        chunk_vectors,
        query_vectors,
        top_k=1,
        candidate_k=2,
    )

    assert result["methods"]["semantic_vector"]["aggregate"]["document_recall_at_k"] == 1.0
    assert result["methods"]["semantic_rrf_mmr"]["aggregate"]["ndcg_at_k"] == 1.0
    with pytest.raises(ValueError, match="counts must match"):
        DenseVectorRetriever(chunks, [[1.0, 0.0]])


def test_dense_strategies_are_available_to_production_retrieval():
    first = make_document("lexical-doc", "Sequence", "ordered command sequence")
    second = make_document("semantic-doc", "Geometry", "solid validity checks")
    chunks = chunk_documents([first, second], chunk_size=200, overlap=20)
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    for strategy in ("semantic_vector", "bm25_semantic_rrf", "semantic_rrf_mmr"):
        ranked = retrieve_with_dense_strategy(
            "geometric validation",
            chunks,
            vectors,
            [0.0, 1.0],
            strategy=strategy,
            top_k=1,
            candidate_k=2,
        )
        assert len(ranked) == 1
        assert ranked[0].rank == 1
