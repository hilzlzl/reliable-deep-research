"""Deterministic control and fusion for adaptive multi-source retrieval."""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field


class RetrievalAction(str, Enum):
    """Action selected after inspecting results from one retrieval source."""

    ACCEPT_LOCAL = "accept_local"
    ESCALATE_WEB = "escalate_web"


class RetrievalCandidate(BaseModel):
    """Provider-neutral chunk returned by any retrieval backend."""

    document_id: str
    chunk_id: str = ""
    source: str = "local"
    sources: list[str] = Field(default_factory=list)
    url: str = ""
    title: str = ""
    text: str
    score: float = 0.0
    rank: int = Field(default=1, ge=1)
    query: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    covered_facets: list[str] = Field(default_factory=list)


class ResearchFacet(BaseModel):
    """One independently answerable part of the research question."""

    facet_id: str
    label: str
    required_terms: list[str] = Field(default_factory=list)
    context_terms: list[str] = Field(default_factory=list)


class FacetCoverage(BaseModel):
    """Evidence coverage for one research facet."""

    facet_id: str
    label: str
    covered: bool
    best_score: float = Field(ge=0.0, le=1.0)
    document_ids: list[str] = Field(default_factory=list)


class RetrievalGain(BaseModel):
    """New information contributed by a supplemental retrieval source."""

    useful: bool
    new_documents: int = Field(ge=0)
    new_content_items: int = Field(ge=0)
    new_query_terms: int = Field(ge=0)
    query_term_coverage_before: float = Field(ge=0.0, le=1.0)
    query_term_coverage_after: float = Field(ge=0.0, le=1.0)
    query_term_coverage_gain: float = Field(ge=0.0, le=1.0)
    facet_coverage_before: float = Field(default=0.0, ge=0.0, le=1.0)
    facet_coverage_after: float = Field(default=0.0, ge=0.0, le=1.0)
    facet_coverage_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class RetrievalAssessment(BaseModel):
    """Auditable measurements and the resulting retrieval action."""

    action: RetrievalAction
    result_count: int
    unique_documents: int
    query_term_coverage: float = Field(ge=0.0, le=1.0)
    facet_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    facets: list[FacetCoverage] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


_LATIN_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
_CJK_SEQUENCE_PATTERN = re.compile(r"[\u3400-\u9fff]+")
_STOP_WORDS = {
    "about",
    "and",
    "are",
    "compare",
    "explain",
    "for",
    "from",
    "how",
    "in",
    "of",
    "on",
    "the",
    "their",
    "to",
    "versus",
    "what",
    "with",
}

_GENERIC_ENTITY_WORDS = {
    "a", "an", "compare", "describe", "explain", "how", "the", "what", "which",
}
_DIMENSION_TERMS = {
    "representation": {"representation", "representations", "encode", "encoding", "format", "schema", "structure", "command", "sequence", "graph", "primitive", "constraint"},
    "preserved_information": {"preserve", "preserved", "information", "retain", "retained", "capture", "captured", "geometry", "topology", "constraint", "constraints", "design", "operation", "parameter", "parameters"},
    "comparison": {"compare", "comparison", "differ", "difference", "differences", "versus", "tradeoff", "tradeoffs"},
    "method": {"method", "methods", "approach", "approaches", "architecture", "pipeline"},
    "evaluation": {"evaluate", "evaluation", "metric", "metrics", "benchmark", "performance", "result", "results"},
    "limitation": {"limitation", "limitations", "failure", "failures", "challenge", "challenges"},
}


def derive_research_facets(queries: list[str]) -> list[ResearchFacet]:
    """Derive a deterministic coverage matrix without spending an LLM call."""
    joined = " ".join(queries)
    tokens = _LATIN_TOKEN_PATTERN.findall(joined)
    entities: list[str] = []
    for token in tokens:
        lower = token.lower()
        looks_named = (
            any(character.isupper() for character in token[1:])
            or (token.isupper() and len(token) > 1)
            or (token[:1].isupper() and token not in tokens[:1])
        )
        if looks_named and lower not in _GENERIC_ENTITY_WORDS and lower not in entities:
            entities.append(lower)

    query_terms = retrieval_terms(queries)
    all_query_tokens = {token.lower() for token in tokens}
    dimensions = [
        (name, sorted(terms))
        for name, terms in _DIMENSION_TERMS.items()
        if terms & all_query_tokens
    ]
    non_comparison = [(name, terms) for name, terms in dimensions if name != "comparison"]
    facets: list[ResearchFacet] = []
    if entities and non_comparison:
        for entity in entities[:4]:
            for dimension, terms in non_comparison[:4]:
                facets.append(
                    ResearchFacet(
                        facet_id=f"{entity}:{dimension}",
                        label=f"{entity} x {dimension}",
                        required_terms=[entity],
                        context_terms=terms,
                    )
                )
        if len(entities) >= 2 and any(name == "comparison" for name, _ in dimensions):
            facets.append(
                ResearchFacet(
                    facet_id="comparison",
                    label="comparison and trade-offs",
                    required_terms=entities[:4],
                    context_terms=next(terms for name, terms in dimensions if name == "comparison"),
                )
            )
    else:
        for index, query in enumerate(queries, 1):
            terms = sorted(retrieval_terms([query]))
            if terms:
                facets.append(
                    ResearchFacet(
                        facet_id=f"query:{index}",
                        label=query,
                        context_terms=terms,
                    )
                )
    return facets


def score_source_quality(candidate: RetrievalCandidate) -> float:
    """Estimate source authority from stable URL and corpus provenance signals."""
    if candidate.source == "local" or candidate.url.startswith("local://"):
        return 0.9
    parts = urlsplit(candidate.url.lower())
    host = parts.netloc.removeprefix("www.")
    path = parts.path
    if host in {"arxiv.org", "openreview.net"} or path.endswith(".pdf"):
        return 1.0
    if any(marker in host for marker in ("acm.org", "ieee.org", "springer.com", "sciencedirect.com")):
        return 0.95
    if host.endswith(".edu") or host.endswith(".ac.uk") or host.endswith(".gov"):
        return 0.9
    if host in {"github.com", "docs.python.org"} or host.startswith("docs."):
        return 0.8
    if any(marker in host for marker in ("linkedin.com", "youtube.com", "facebook.com", "medium.com")):
        return 0.2
    if any(marker in host for marker in ("researchgate.net", "semanticscholar.org", "paperswithcode.com")):
        return 0.55
    return 0.6


def _facet_score(facet: ResearchFacet, candidate: RetrievalCandidate) -> float:
    text = f"{candidate.title} {candidate.text}".lower()
    required_hits = sum(term in text for term in facet.required_terms)
    if facet.required_terms and required_hits < len(facet.required_terms):
        return 0.0
    context_hits = sum(term in text for term in facet.context_terms)
    context_ratio = context_hits / len(facet.context_terms) if facet.context_terms else 1.0
    if not facet.required_terms:
        return context_ratio
    if facet.context_terms and context_hits == 0:
        return 0.3
    required_ratio = required_hits / len(facet.required_terms) if facet.required_terms else 1.0
    return min(1.0, 0.6 * required_ratio + 0.4 * context_ratio)


def facet_coverage_matrix(
    queries: list[str],
    candidates: list[RetrievalCandidate],
    *,
    min_relevance: float,
) -> list[FacetCoverage]:
    facets = derive_research_facets(queries)
    matrix = []
    for facet in facets:
        scored = [(candidate, _facet_score(facet, candidate)) for candidate in candidates]
        matching = [candidate.document_id for candidate, score in scored if score >= min_relevance]
        best = max((score for _, score in scored), default=0.0)
        if facet.facet_id == "comparison" and candidates:
            corpus = " ".join(f"{item.title} {item.text}" for item in candidates).lower()
            entity_coverage = all(term in corpus for term in facet.required_terms)
            # A comparison can be synthesized from primary evidence for every
            # named entity; it does not require a lower-quality comparison page.
            if entity_coverage:
                matching = list(dict.fromkeys(item.document_id for item in candidates))
                best = max(best, 1.0)
        matrix.append(
            FacetCoverage(
                facet_id=facet.facet_id,
                label=facet.label,
                covered=bool(matching),
                best_score=round(best, 4),
                document_ids=matching,
            )
        )
    return matrix


def rerank_candidates(
    queries: list[str],
    candidates: list[RetrievalCandidate],
    *,
    min_relevance: float,
    min_source_quality: float,
    max_per_document: int | None = None,
) -> list[RetrievalCandidate]:
    """Apply the same relevance and authority ranking to every retrieval source."""
    facets = derive_research_facets(queries)
    terms = retrieval_terms(queries)
    output: list[RetrievalCandidate] = []
    for candidate in candidates:
        text = f"{candidate.title} {candidate.text}".lower()
        lexical = sum(term in text for term in terms) / len(terms) if terms else 0.0
        facet_scores = {facet.facet_id: _facet_score(facet, candidate) for facet in facets}
        facet_relevance = max(facet_scores.values(), default=lexical)
        relevance = min(1.0, 0.65 * facet_relevance + 0.35 * lexical)
        quality = score_source_quality(candidate)
        candidate = candidate.model_copy(deep=True)
        candidate.relevance_score = round(relevance, 4)
        candidate.source_quality = round(quality, 4)
        candidate.covered_facets = sorted(
            facet_id for facet_id, score in facet_scores.items() if score >= min_relevance
        )
        if relevance >= min_relevance and quality >= min_source_quality:
            output.append(candidate)
    output.sort(
        key=lambda item: (
            -(
                0.55 * item.relevance_score
                + 0.3 * item.source_quality
                + 0.15 * (1.0 / max(item.rank, 1))
            ),
            item.rank,
            item.document_id,
        )
    )
    if max_per_document is not None:
        diversified: list[RetrievalCandidate] = []
        document_counts: dict[str, int] = {}
        for candidate in output:
            count = document_counts.get(candidate.document_id, 0)
            if count >= max_per_document:
                continue
            document_counts[candidate.document_id] = count + 1
            diversified.append(candidate)
        output = diversified
    for rank, candidate in enumerate(output, 1):
        candidate.rank = rank
    return output


def select_minimal_facet_cover(
    queries: list[str],
    candidates: list[RetrievalCandidate],
    *,
    min_results: int,
    limit: int,
) -> list[RetrievalCandidate]:
    """Keep the smallest ranked prefix that covers every evidence facet."""
    required = {
        facet.facet_id
        for facet in derive_research_facets(queries)
        if facet.facet_id != "comparison"
    }
    selected: list[RetrievalCandidate] = []
    covered: set[str] = set()
    for candidate in candidates:
        selected.append(candidate)
        covered.update(candidate.covered_facets)
        if len(selected) >= min_results and required.issubset(covered):
            break
        if len(selected) >= limit:
            break
    return selected[:limit]


def retrieval_terms(queries: list[str]) -> set[str]:
    """Extract stable lexical anchors without requiring another model call."""
    latin_terms = {
        token.lower()
        for query in queries
        for token in _LATIN_TOKEN_PATTERN.findall(query)
        if token.lower() not in _STOP_WORDS
    }
    cjk_terms = {
        sequence[index : index + 2]
        for query in queries
        for sequence in _CJK_SEQUENCE_PATTERN.findall(query)
        for index in range(max(0, len(sequence) - 1))
    }
    return latin_terms | cjk_terms


def canonical_url(url: str) -> str:
    """Normalize URL identity while ignoring fragments such as local chunk IDs."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def content_fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _covered_terms(queries: list[str], candidates: list[RetrievalCandidate]) -> set[str]:
    terms = retrieval_terms(queries)
    corpus_text = " ".join(
        f"{candidate.title} {candidate.text}" for candidate in candidates
    ).lower()
    return {term for term in terms if term in corpus_text}


def fuse_candidates(
    queries: list[str],
    candidate_groups: list[list[RetrievalCandidate]],
    *,
    limit: int,
    rrf_k: int = 60,
) -> list[RetrievalCandidate]:
    """Deduplicate and rank local/web chunks with reciprocal-rank fusion.

    Exact content is always deduplicated. Matching canonical URLs are merged
    across providers, while multiple chunks from one provider remain available.
    This prevents the same page returned by local and web search from receiving
    two output slots without discarding useful within-document chunks.
    """
    terms = retrieval_terms(queries)
    merged: list[dict[str, object]] = []
    content_index: dict[str, int] = {}
    url_source_index: dict[tuple[str, str], int] = {}

    for group in candidate_groups:
        for position, candidate in enumerate(group, 1):
            source = candidate.source or "unknown"
            fingerprint = content_fingerprint(candidate.text)
            normalized_url = canonical_url(candidate.url)
            duplicate_index = content_index.get(fingerprint)
            if duplicate_index is None and normalized_url:
                for (known_url, known_source), index in url_source_index.items():
                    if known_url == normalized_url and known_source != source:
                        duplicate_index = index
                        break

            contribution = 1.0 / (rrf_k + (candidate.rank or position))
            lexical_hits = sum(
                term in f"{candidate.title} {candidate.text}".lower() for term in terms
            )
            if duplicate_index is None:
                item = candidate.model_copy(deep=True)
                item.sources = sorted(set(item.sources or [source]))
                merged.append(
                    {
                        "candidate": item,
                        "rrf": contribution,
                        "lexical_hits": lexical_hits,
                    }
                )
                duplicate_index = len(merged) - 1
                content_index[fingerprint] = duplicate_index
            else:
                entry = merged[duplicate_index]
                existing = entry["candidate"]
                assert isinstance(existing, RetrievalCandidate)
                existing.sources = sorted(set(existing.sources + [source]))
                entry["rrf"] = float(entry["rrf"]) + contribution
                entry["lexical_hits"] = max(int(entry["lexical_hits"]), lexical_hits)
                if len(candidate.text) > len(existing.text):
                    existing.text = candidate.text
                    existing.title = candidate.title or existing.title
            if normalized_url:
                url_source_index[(normalized_url, source)] = duplicate_index

    merged.sort(
        key=lambda item: (
            -float(item["rrf"]),
            -int(item["lexical_hits"]),
            str(getattr(item["candidate"], "document_id")),
        )
    )
    output: list[RetrievalCandidate] = []
    for rank, entry in enumerate(merged[:limit], 1):
        candidate = entry["candidate"]
        assert isinstance(candidate, RetrievalCandidate)
        candidate.rank = rank
        candidate.score = round(float(entry["rrf"]), 8)
        output.append(candidate)
    return output


def assess_retrieval_gain(
    queries: list[str],
    existing: list[RetrievalCandidate],
    supplemental: list[RetrievalCandidate],
    *,
    min_new_documents: int,
    min_query_term_coverage_gain: float,
    min_facet_relevance: float = 0.55,
) -> RetrievalGain:
    """Decide whether a supplemental search contributed enough new evidence."""
    terms = retrieval_terms(queries)
    before_terms = _covered_terms(queries, existing)
    after_terms = _covered_terms(queries, existing + supplemental)
    before = len(before_terms) / len(terms) if terms else (1.0 if existing else 0.0)
    after = len(after_terms) / len(terms) if terms else (1.0 if existing or supplemental else 0.0)
    existing_documents = {
        canonical_url(item.url) or item.document_id for item in existing
    }
    supplemental_documents = {
        canonical_url(item.url) or item.document_id for item in supplemental
    }
    new_documents = len(supplemental_documents - existing_documents)
    existing_content = {content_fingerprint(item.text) for item in existing}
    new_content_items = len(
        {content_fingerprint(item.text) for item in supplemental} - existing_content
    )
    coverage_gain = max(0.0, after - before)
    before_matrix = facet_coverage_matrix(
        queries, existing, min_relevance=min_facet_relevance
    )
    after_matrix = facet_coverage_matrix(
        queries, existing + supplemental, min_relevance=min_facet_relevance
    )
    facet_before = (
        sum(item.covered for item in before_matrix) / len(before_matrix)
        if before_matrix
        else 0.0
    )
    facet_after = (
        sum(item.covered for item in after_matrix) / len(after_matrix)
        if after_matrix
        else 0.0
    )
    facet_gain = max(0.0, facet_after - facet_before)
    document_signal = new_documents >= min_new_documents
    coverage_signal = coverage_gain >= min_query_term_coverage_gain
    facet_signal = facet_gain > 0.0
    useful = new_content_items > 0 and facet_signal and (document_signal or coverage_signal)
    reasons: list[str] = []
    if not useful and not document_signal:
        reasons.append(
            f"new_documents_below_minimum:{new_documents}<{min_new_documents}"
        )
    if not useful and not coverage_signal:
        reasons.append(
            "query_term_coverage_gain_below_minimum:"
            f"{coverage_gain:.4f}<{min_query_term_coverage_gain:.4f}"
        )
    if new_content_items == 0:
        reasons.append("no_new_content")
    if not facet_signal:
        reasons.append("no_new_facet_coverage")
    return RetrievalGain(
        useful=useful,
        new_documents=new_documents,
        new_content_items=new_content_items,
        new_query_terms=len(after_terms - before_terms),
        query_term_coverage_before=round(before, 4),
        query_term_coverage_after=round(after, 4),
        query_term_coverage_gain=round(coverage_gain, 4),
        facet_coverage_before=round(facet_before, 4),
        facet_coverage_after=round(facet_after, 4),
        facet_coverage_gain=round(facet_gain, 4),
        reasons=reasons,
    )


def assess_local_results(
    queries: list[str],
    candidates: list[RetrievalCandidate],
    *,
    min_results: int,
    min_unique_documents: int,
    min_query_term_coverage: float,
    min_facet_coverage: float = 1.0,
    min_facet_relevance: float = 0.55,
) -> RetrievalAssessment:
    """Choose whether local evidence is sufficient or web search is warranted."""
    terms = retrieval_terms(queries)
    matched_terms = _covered_terms(queries, candidates)
    coverage = len(matched_terms) / len(terms) if terms else (1.0 if candidates else 0.0)
    unique_documents = len({item.document_id for item in candidates})
    facet_matrix = facet_coverage_matrix(
        queries, candidates, min_relevance=min_facet_relevance
    )
    facet_coverage = (
        sum(item.covered for item in facet_matrix) / len(facet_matrix)
        if facet_matrix
        else (1.0 if candidates else 0.0)
    )

    reasons: list[str] = []
    if len(candidates) < min_results:
        reasons.append(f"results_below_minimum:{len(candidates)}<{min_results}")
    if unique_documents < min_unique_documents:
        reasons.append(
            f"source_diversity_below_minimum:{unique_documents}<{min_unique_documents}"
        )
    if coverage < min_query_term_coverage and facet_coverage < min_facet_coverage:
        reasons.append(
            "query_term_coverage_below_minimum:"
            f"{coverage:.4f}<{min_query_term_coverage:.4f}"
        )
    if facet_coverage < min_facet_coverage:
        missing = ",".join(item.facet_id for item in facet_matrix if not item.covered)
        reasons.append(
            "facet_coverage_below_minimum:"
            f"{facet_coverage:.4f}<{min_facet_coverage:.4f};missing={missing}"
        )

    return RetrievalAssessment(
        action=(
            RetrievalAction.ESCALATE_WEB
            if reasons
            else RetrievalAction.ACCEPT_LOCAL
        ),
        result_count=len(candidates),
        unique_documents=unique_documents,
        query_term_coverage=round(coverage, 4),
        facet_coverage=round(facet_coverage, 4),
        facets=facet_matrix,
        reasons=reasons,
    )
