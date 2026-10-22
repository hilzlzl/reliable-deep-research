"""Compatibility helpers for fact-annotated Deep Research benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field, model_validator

from open_deep_research.local_benchmark import (
    BM25Retriever,
    CorpusChunk,
    CorpusDocument,
    DenseVectorRetriever,
    RankedChunk,
    TfidfVectorRetriever,
    chunk_documents,
    dense_mmr_select,
    mmr_select,
    reciprocal_rank_fusion,
    write_jsonl,
)


class ExpectedFact(BaseModel):
    """One answer fact and the lexical alternatives used to detect it."""

    fact: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class FactBenchmarkCase(BaseModel):
    """Portable subset of the benchmark schema used by deepresearch."""

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    difficulty: str
    expected_facts: list[ExpectedFact]
    required_citations: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)
    question_lang: str = "en"
    evidence_lang: str = "en"
    source_dataset: str = "deepresearch"
    evaluation_focus: str = "factual_accuracy"

    @model_validator(mode="after")
    def require_facts(self) -> "FactBenchmarkCase":
        if not self.expected_facts:
            raise ValueError("Fact benchmark cases require expected facts")
        return self


def import_fact_benchmark(
    dataset_path: Path,
    corpus_directory: Path,
    output_directory: Path,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> dict[str, Any]:
    """Normalize another project's benchmark without modifying its source files."""
    raw_cases = _read_raw_jsonl(dataset_path)
    cases = [_normalize_case(value) for value in raw_cases]
    documents = [_document_from_markdown(path) for path in sorted(corpus_directory.glob("*.md"))]
    if not documents:
        raise ValueError(f"No Markdown documents found in {corpus_directory}")
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    output_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_directory / "cases.jsonl", cases)
    write_jsonl(output_directory / "documents.jsonl", documents)
    write_jsonl(output_directory / "chunks.jsonl", chunks)
    manifest = {
        "source_dataset": str(dataset_path.resolve()),
        "source_corpus": str(corpus_directory.resolve()),
        "source_dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "cases": len(cases),
        "documents": len(documents),
        "chunks": len(chunks),
        "question_languages": _counts(case.question_lang for case in cases),
        "domains": _counts(case.domain for case in cases),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "metric_contract": "expected-fact lexical coverage; no gold document or quote labels",
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def evaluate_fact_retrieval_ablation(
    cases: list[FactBenchmarkCase],
    chunks: list[CorpusChunk],
    *,
    top_k: int = 5,
    candidate_k: int = 40,
    rrf_k: int = 60,
    mmr_lambda: float = 0.7,
) -> dict[str, Any]:
    """Evaluate dependency-free retrieval methods using expected-fact coverage."""
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    bm25 = BM25Retriever(chunks)
    vector = TfidfVectorRetriever(chunks)
    methods: dict[str, dict[str, list[RankedChunk]]] = {
        "bm25": {},
        "tfidf_vector": {},
        "rrf_hybrid": {},
        "rrf_mmr": {},
    }
    for case in cases:
        lexical = bm25.retrieve(case.question, top_k=candidate_k)
        vector_ranking = vector.retrieve(case.question, top_k=candidate_k)
        fused = reciprocal_rank_fusion(
            [lexical, vector_ranking], top_k=candidate_k, rrf_k=rrf_k
        )
        methods["bm25"][case.case_id] = lexical
        methods["tfidf_vector"][case.case_id] = vector_ranking
        methods["rrf_hybrid"][case.case_id] = fused
        methods["rrf_mmr"][case.case_id] = mmr_select(
            fused, vector, top_k=top_k, lambda_weight=mmr_lambda
        )
    return _render_ablation(
        cases,
        methods,
        top_k=top_k,
        configuration={
            "top_k": top_k,
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "mmr_lambda": mmr_lambda,
            "vector_method": "word/CJK-ngram TF-IDF cosine; not a semantic model",
        },
    )


def evaluate_semantic_fact_retrieval_ablation(
    cases: list[FactBenchmarkCase],
    chunks: list[CorpusChunk],
    chunk_vectors: Sequence[Sequence[float]],
    query_vectors: dict[str, Sequence[float]],
    *,
    top_k: int = 5,
    candidate_k: int = 40,
    rrf_k: int = 60,
    mmr_lambda: float = 0.7,
) -> dict[str, Any]:
    """Evaluate dense and hybrid retrieval using expected-fact coverage."""
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    missing = {case.case_id for case in cases} - query_vectors.keys()
    if missing:
        raise ValueError(f"Missing query vectors: {', '.join(sorted(missing))}")
    bm25 = BM25Retriever(chunks)
    dense = DenseVectorRetriever(chunks, chunk_vectors)
    methods: dict[str, dict[str, list[RankedChunk]]] = {
        "bm25": {},
        "semantic_vector": {},
        "bm25_semantic_rrf": {},
        "semantic_rrf_mmr": {},
    }
    for case in cases:
        lexical = bm25.retrieve(case.question, top_k=candidate_k)
        semantic = dense.retrieve(query_vectors[case.case_id], top_k=candidate_k)
        fused = reciprocal_rank_fusion(
            [lexical, semantic], top_k=candidate_k, rrf_k=rrf_k
        )
        methods["bm25"][case.case_id] = lexical
        methods["semantic_vector"][case.case_id] = semantic
        methods["bm25_semantic_rrf"][case.case_id] = fused
        methods["semantic_rrf_mmr"][case.case_id] = dense_mmr_select(
            fused, dense, top_k=top_k, lambda_weight=mmr_lambda
        )
    return _render_ablation(
        cases,
        methods,
        top_k=top_k,
        configuration={
            "top_k": top_k,
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "mmr_lambda": mmr_lambda,
        },
    )


def evaluate_fact_rankings(
    cases: list[FactBenchmarkCase],
    rankings: dict[str, list[RankedChunk]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Score how many annotated facts have a lexical witness in retrieved text."""
    results = []
    for case in cases:
        ranked = rankings[case.case_id][:top_k]
        text = "\n".join(f"{item.chunk.title}\n{item.chunk.text}" for item in ranked)
        fact_hits = [_fact_matches(fact, text) for fact in case.expected_facts]
        results.append(
            {
                "case_id": case.case_id,
                "domain": case.domain,
                "difficulty": case.difficulty,
                "question_lang": case.question_lang,
                "fact_count": len(fact_hits),
                "fact_hits": sum(fact_hits),
                "fact_recall_at_k": round(sum(fact_hits) / len(fact_hits), 4),
                "source_diversity_at_k": round(
                    len({item.chunk.document_id for item in ranked}) / len(ranked), 4
                )
                if ranked
                else 0.0,
                "retrieved_chunk_ids": [item.chunk.chunk_id for item in ranked],
            }
        )
    return {
        "top_k": top_k,
        "evaluated_cases": len(results),
        "aggregate": _aggregate_results(results),
        "by_language": _group_results(results, "question_lang"),
        "by_difficulty": _group_results(results, "difficulty"),
        "cases": results,
    }


def evaluate_text_facts(text: str, facts: list[ExpectedFact]) -> dict[str, Any]:
    """Measure weak-label expected-fact coverage in a generated report."""
    hits = [_fact_matches(fact, text) for fact in facts]
    return {
        "fact_count": len(hits),
        "fact_hits": sum(hits),
        "fact_coverage": round(sum(hits) / len(hits), 4) if hits else 0.0,
    }


def _render_ablation(
    cases: list[FactBenchmarkCase],
    methods: dict[str, dict[str, list[RankedChunk]]],
    *,
    top_k: int,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "metric_note": (
            "A fact is covered when at least one annotated keyword, alias, or fact phrase "
            "has an exact or >=60% token-overlap witness in the top-k chunks. This is a "
            "weak-label retrieval metric, not answer correctness or gold-document recall."
        ),
        "methods": {
            name: evaluate_fact_rankings(cases, rankings, top_k=top_k)
            for name, rankings in methods.items()
        },
    }


def _normalize_case(value: dict[str, Any]) -> FactBenchmarkCase:
    facts = []
    for raw_fact in value.get("expected_facts", []):
        if isinstance(raw_fact, str):
            facts.append(ExpectedFact(fact=raw_fact))
        else:
            facts.append(ExpectedFact.model_validate(raw_fact))
    return FactBenchmarkCase(
        case_id=value["id"],
        question=value["question"],
        domain=value["domain"],
        difficulty=value.get("difficulty", "unknown"),
        expected_facts=facts,
        required_citations=value.get("required_citations", 0),
        tags=value.get("tags", []),
        question_lang=value.get("question_lang", "en"),
        evidence_lang=value.get("evidence_lang", "en"),
        source_dataset=value.get("source_dataset", "deepresearch_import"),
        evaluation_focus=value.get("evaluation_focus", "factual_accuracy"),
    )


def _document_from_markdown(path: Path) -> CorpusDocument:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Empty corpus document: {path}")
    heading = next(
        (line.lstrip("# ").strip() for line in content.splitlines() if line.startswith("#")),
        path.stem.replace("_", " ").title(),
    )
    return CorpusDocument(
        document_id=path.stem,
        title=heading,
        domain=_infer_domain(path.stem),
        source_type="imported_local_corpus",
        source_url=f"local://deepresearch/{path.name}",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _infer_domain(stem: str) -> str:
    domains = (
        "agent_orchestration",
        "data_engineering",
        "fine_tuning",
        "llm_agents",
        "memory_systems",
    )
    match = next((domain for domain in domains if stem.startswith(domain)), None)
    if match:
        return match
    aliases = {"agent": "agent_orchestration", "memory": "memory_systems"}
    prefix = stem.split("_", 1)[0]
    return aliases.get(prefix, prefix)


def _read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid {path}:{line_number}: {error}") from error
    return values


def _fact_matches(fact: ExpectedFact, content: str) -> bool:
    return any(_term_matches(term, content) for term in [*fact.keywords, *fact.aliases, fact.fact])


def _term_matches(term: str, content: str) -> bool:
    normalized_term = term.casefold().strip()
    normalized_content = content.casefold()
    if not normalized_term:
        return False
    if normalized_term in normalized_content:
        return True
    term_tokens = set(_multilingual_tokens(normalized_term))
    content_tokens = set(_multilingual_tokens(normalized_content))
    return bool(term_tokens) and len(term_tokens & content_tokens) / len(term_tokens) >= 0.6


def _multilingual_tokens(text: str) -> list[str]:
    latin = re.findall(r"[a-z0-9]+", text.casefold())
    cjk = []
    for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        cjk.extend(sequence if len(sequence) == 1 else (sequence[index : index + 2] for index in range(len(sequence) - 1)))
    return latin + cjk


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"fact_recall_at_k": 0.0, "source_diversity_at_k": 0.0}
    return {
        metric: round(sum(result[metric] for result in results) / len(results), 4)
        for metric in ("fact_recall_at_k", "source_diversity_at_k")
    }


def _group_results(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(result[key], []).append(result)
    return {
        name: {"cases": len(group), **_aggregate_results(group)}
        for name, group in sorted(groups.items())
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
