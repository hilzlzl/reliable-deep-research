from open_deep_research.adaptive_retrieval import (
    RetrievalAction,
    RetrievalCandidate,
    assess_retrieval_gain,
    assess_local_results,
    canonical_url,
    derive_research_facets,
    fuse_candidates,
    rerank_candidates,
    retrieval_terms,
    score_source_quality,
    select_minimal_facet_cover,
)


def candidate(document_id: str, text: str) -> RetrievalCandidate:
    return RetrievalCandidate(document_id=document_id, title=document_id, text=text)


def test_accepts_sufficient_diverse_local_results():
    result = assess_local_results(
        ["Compare DeepCAD and SketchGraphs representations"],
        [
            candidate("deepcad", "DeepCAD uses a command representation."),
            candidate("sketchgraphs", "SketchGraphs uses graph representations."),
            candidate("survey", "A comparison of both representations."),
        ],
        min_results=3,
        min_unique_documents=2,
        min_query_term_coverage=0.75,
    )

    assert result.action == RetrievalAction.ACCEPT_LOCAL
    assert result.query_term_coverage == 1.0
    assert result.reasons == []


def test_escalates_when_entity_coverage_and_diversity_are_low():
    result = assess_local_results(
        ["Compare DeepCAD and SketchGraphs representations"],
        [candidate("sketchgraphs", "SketchGraphs contains geometric constraint graphs.")],
        min_results=2,
        min_unique_documents=2,
        min_query_term_coverage=0.75,
    )

    assert result.action == RetrievalAction.ESCALATE_WEB
    assert result.query_term_coverage < 0.75
    assert len(result.reasons) == 4
    assert result.facet_coverage == 0.3333
    assert "facet_coverage_below_minimum" in result.reasons[-1]


def test_empty_query_still_requires_results():
    result = assess_local_results(
        ["the and of"],
        [],
        min_results=1,
        min_unique_documents=1,
        min_query_term_coverage=0.5,
    )

    assert result.action == RetrievalAction.ESCALATE_WEB
    assert result.query_term_coverage == 0.0


def test_chinese_queries_use_cjk_bigrams_instead_of_one_long_token():
    terms = retrieval_terms(["比较本地检索和网络检索"])

    assert "本地" in terms
    assert "检索" in terms
    assert "网络" in terms
    assert "比较本地检索和网络检索" not in terms


def test_fusion_deduplicates_same_page_across_sources_and_combines_ranks():
    local = RetrievalCandidate(
        document_id="local-paper",
        chunk_id="local-paper:0",
        source="local",
        url="https://example.org/paper#chunk=0",
        title="CAD paper",
        text="CAD command sequence representation.",
        rank=1,
    )
    web_duplicate = RetrievalCandidate(
        document_id="web-paper",
        source="web",
        url="https://example.org/paper",
        title="CAD paper",
        text="CAD command sequence representation with additional details.",
        rank=1,
    )
    web_new = RetrievalCandidate(
        document_id="web-survey",
        source="web",
        url="https://example.org/survey",
        title="CAD survey",
        text="A survey compares graph and command representations.",
        rank=2,
    )

    fused = fuse_candidates(
        ["CAD representation"], [[local], [web_duplicate, web_new]], limit=5
    )

    assert len(fused) == 2
    assert fused[0].sources == ["local", "web"]
    assert fused[0].score > fused[1].score
    assert canonical_url(fused[0].url) == "https://example.org/paper"


def test_retrieval_gain_requires_new_information():
    local = [
        RetrievalCandidate(
            document_id="local",
            url="https://example.org/paper",
            text="DeepCAD command representation",
        )
    ]
    duplicate = [
        RetrievalCandidate(
            document_id="web-copy",
            source="web",
            url="https://example.org/paper#abstract",
            text="DeepCAD command representation",
        )
    ]
    new_source = [
        RetrievalCandidate(
            document_id="web-new",
            source="web",
            url="https://example.org/sketchgraphs",
            text="SketchGraphs graph representation",
        )
    ]

    rejected = assess_retrieval_gain(
        ["DeepCAD SketchGraphs representation"],
        local,
        duplicate,
        min_new_documents=1,
        min_query_term_coverage_gain=0.1,
    )
    accepted = assess_retrieval_gain(
        ["DeepCAD SketchGraphs representation"],
        local,
        new_source,
        min_new_documents=1,
        min_query_term_coverage_gain=0.1,
    )

    assert rejected.useful is False
    assert "no_new_content" in rejected.reasons
    assert accepted.useful is True
    assert accepted.new_documents == 1
    assert accepted.new_query_terms >= 1


def test_comparison_question_builds_entity_dimension_facets():
    facets = derive_research_facets(
        [
            "How do the core representations in DeepCAD and SketchGraphs differ, "
            "and what design information does each preserve?"
        ]
    )

    assert {facet.facet_id for facet in facets} == {
        "deepcad:representation",
        "deepcad:preserved_information",
        "sketchgraphs:representation",
        "sketchgraphs:preserved_information",
        "comparison",
    }


def test_irrelevant_documents_do_not_inflate_facet_coverage():
    result = assess_local_results(
        [
            "How do the representations in DeepCAD and SketchGraphs differ, "
            "and what information does each preserve?"
        ],
        [
            candidate("pytorch", "PyTorch is a machine learning framework."),
            candidate("polygen", "PolyGen generates polygonal meshes."),
            candidate("deepcad-name", "DeepCAD project news and downloads."),
        ],
        min_results=3,
        min_unique_documents=2,
        min_query_term_coverage=0.2,
        min_facet_coverage=0.75,
    )

    assert result.action == RetrievalAction.ESCALATE_WEB
    assert result.facet_coverage == 0.0


def test_reranking_prefers_primary_relevant_sources_and_filters_social_pages():
    query = ["DeepCAD representation"]
    paper = RetrievalCandidate(
        document_id="paper",
        source="web",
        url="https://arxiv.org/abs/2105.09492",
        title="DeepCAD",
        text="DeepCAD uses a command sequence representation.",
        score=0.4,
    )
    social = RetrievalCandidate(
        document_id="social",
        source="web",
        url="https://www.linkedin.com/posts/example",
        title="DeepCAD",
        text="DeepCAD representation announcement.",
        score=0.9,
    )
    unrelated = RetrievalCandidate(
        document_id="unrelated",
        source="web",
        url="https://example.com/pytorch",
        title="PyTorch",
        text="A general tensor library.",
        score=1.0,
    )

    ranked = rerank_candidates(
        query,
        [social, unrelated, paper],
        min_relevance=0.35,
        min_source_quality=0.35,
    )

    assert [item.document_id for item in ranked] == ["paper"]
    assert score_source_quality(paper) == 1.0


def test_new_but_irrelevant_web_content_is_not_useful_gain():
    local = [candidate("deepcad", "DeepCAD uses a command representation.")]
    web = [candidate("pytorch", "PyTorch supports GPU tensor computation.")]

    gain = assess_retrieval_gain(
        ["Compare DeepCAD and SketchGraphs representations"],
        local,
        web,
        min_new_documents=1,
        min_query_term_coverage_gain=0.0,
    )

    assert gain.useful is False
    assert gain.facet_coverage_gain == 0.0
    assert "no_new_facet_coverage" in gain.reasons


def test_minimal_facet_cover_stops_before_redundant_related_work():
    query = ["Compare DeepCAD and SketchGraphs representations"]
    candidates = rerank_candidates(
        query,
        [
            candidate("sketch-1", "SketchGraphs graph representation"),
            candidate("sketch-2", "SketchGraphs constraint graph representation"),
            candidate("deepcad", "DeepCAD command sequence representation"),
            candidate("text2cad", "Text2CAD discusses DeepCAD representation"),
        ],
        min_relevance=0.35,
        min_source_quality=0.35,
    )

    selected = select_minimal_facet_cover(
        query, candidates, min_results=3, limit=5
    )

    assert len(selected) == 3
    assert "text2cad" not in {item.document_id for item in selected}
