"""Deterministic local-corpus construction and retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field, model_validator


class SourceRecord(BaseModel):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    acquisition_status: str
    redistribution: str
    notes: str = ""


class CorpusDocument(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    content: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self) -> "CorpusDocument":
        actual = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if actual != self.content_sha256:
            raise ValueError(f"Content hash mismatch for {self.document_id}")
        return self


class CorpusChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    domain: str
    source_url: str
    text: str
    start_character: int
    end_character: int


class GoldEvidence(BaseModel):
    document_id: str
    quote: str = Field(min_length=1)


class LocalBenchmarkCase(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    difficulty: str
    answerable: bool = True
    gold_document_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    reference_answer_points: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answerability(self) -> "LocalBenchmarkCase":
        if self.answerable and (not self.gold_document_ids or not self.gold_evidence):
            raise ValueError("Answerable cases require gold documents and evidence")
        if not self.answerable and (self.gold_document_ids or self.gold_evidence):
            raise ValueError("Unanswerable cases cannot contain gold evidence")
        return self


class RankedChunk(BaseModel):
    chunk: CorpusChunk
    score: float
    rank: int


def read_jsonl(path: Path, schema: type[BaseModel]) -> list[Any]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(schema.model_validate_json(line))
        except Exception as error:
            raise ValueError(f"Invalid {path}:{line_number}: {error}") from error
    return values


def write_jsonl(path: Path, values: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(value.model_dump_json() + "\n")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for element in soup(["script", "style", "nav"]):
            element.decompose()
        return soup.get_text("\n", strip=True)
    if suffix == ".pdf":
        import fitz

        with fitz.open(path) as document:
            return "\n\n".join(page.get_text("text") for page in document)
    raise ValueError(f"Unsupported source format: {path}")


def build_documents(
    sources: list[SourceRecord], raw_directory: Path
) -> tuple[list[CorpusDocument], list[str]]:
    documents = []
    missing = []
    for source in sources:
        matches = sorted(raw_directory.glob(f"{source.source_id}.*"))
        matches = [path for path in matches if path.suffix.lower() in {".md", ".txt", ".html", ".htm", ".pdf"}]
        if not matches:
            missing.append(source.source_id)
            continue
        if source.acquisition_status not in {"approved", "approved_local_only"}:
            raise ValueError(
                f"Source {source.source_id} has local content but is not approved for corpus admission"
            )
        if len(matches) > 1:
            raise ValueError(f"Multiple raw files found for {source.source_id}: {matches}")
        content = extract_text(matches[0]).strip()
        if not content:
            raise ValueError(f"Source {source.source_id} produced empty text")
        documents.append(
            CorpusDocument(
                document_id=source.source_id,
                title=source.title,
                domain=source.domain,
                source_type=source.source_type,
                source_url=source.locator,
                content=content,
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
    return documents, missing


def chunk_documents(
    documents: list[CorpusDocument], *, chunk_size: int = 1200, overlap: int = 200
) -> list[CorpusChunk]:
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size >= 100 and 0 <= overlap < chunk_size")
    chunks = []
    for document in documents:
        start = 0
        index = 0
        while start < len(document.content):
            hard_end = min(len(document.content), start + chunk_size)
            end = hard_end
            if hard_end < len(document.content):
                boundary = document.content.rfind(" ", start, hard_end)
                if boundary > start + chunk_size // 2:
                    end = boundary
            text = document.content[start:end].strip()
            if text:
                chunks.append(
                    CorpusChunk(
                        chunk_id=f"{document.document_id}:{index}",
                        document_id=document.document_id,
                        title=document.title,
                        domain=document.domain,
                        source_url=document.source_url,
                        text=text,
                        start_character=start,
                        end_character=end,
                    )
                )
                index += 1
            if end >= len(document.content):
                break
            start = max(start + 1, end - overlap)
    return chunks


def validate_benchmark(
    documents: list[CorpusDocument], cases: list[LocalBenchmarkCase]
) -> dict[str, Any]:
    by_id = {document.document_id: document for document in documents}
    if len(by_id) != len(documents):
        raise ValueError("Document IDs must be unique")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique")
    errors = []
    for case in cases:
        for document_id in case.gold_document_ids:
            if document_id not in by_id:
                errors.append(f"{case.case_id}: unknown document {document_id}")
        for evidence in case.gold_evidence:
            document = by_id.get(evidence.document_id)
            if document is None:
                errors.append(f"{case.case_id}: unknown evidence document {evidence.document_id}")
            elif _normalize(evidence.quote) not in _normalize(document.content):
                errors.append(f"{case.case_id}: evidence quote not found in {evidence.document_id}")
    return {
        "documents": len(documents),
        "cases": len(cases),
        "answerable_cases": sum(case.answerable for case in cases),
        "unanswerable_cases": sum(not case.answerable for case in cases),
        "valid": not errors,
        "errors": errors,
    }


class BM25Retriever:
    def __init__(self, chunks: list[CorpusChunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [_tokens(f"{chunk.title} {chunk.text}") for chunk in chunks]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.average_length = (
            sum(len(tokens) for tokens in self.tokens) / len(self.tokens) if self.tokens else 0.0
        )
        self.document_frequency = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RankedChunk]:
        query_terms = _tokens(query)
        scores = []
        total = len(self.chunks)
        for chunk, frequencies, tokens in zip(
            self.chunks, self.term_frequencies, self.tokens, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse_frequency = math.log(1 + (total - df + 0.5) / (df + 0.5))
                length_ratio = len(tokens) / self.average_length if self.average_length else 0.0
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            scores.append((score, chunk))
        positive_scores = [item for item in scores if item[0] > 0]
        ranked = sorted(
            positive_scores, key=lambda item: (-item[0], item[1].chunk_id)
        )[:top_k]
        return [
            RankedChunk(chunk=chunk, score=round(score, 8), rank=index)
            for index, (score, chunk) in enumerate(ranked, 1)
        ]


class TfidfVectorRetriever:
    """Dependency-free TF-IDF cosine baseline; this is not a semantic encoder."""

    def __init__(self, chunks: list[CorpusChunk]):
        self.chunks = chunks
        tokenized = [_tokens(f"{chunk.title} {chunk.text}") for chunk in chunks]
        document_frequency = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        total = len(chunks)
        self.idf = {
            term: math.log((total + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self.vectorize_tokens(tokens) for tokens in tokenized]

    def vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        frequencies = Counter(tokens)
        return {
            term: frequency * self.idf[term]
            for term, frequency in frequencies.items()
            if term in self.idf
        }

    def vectorize(self, text: str) -> dict[str, float]:
        return self.vectorize_tokens(_tokens(text))

    def similarity(self, left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        shared = left.keys() & right.keys()
        numerator = sum(left[term] * right[term] for term in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RankedChunk]:
        query_vector = self.vectorize(query)
        scored = [
            (self.similarity(query_vector, vector), chunk)
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
        ]
        ranked = sorted(
            (item for item in scored if item[0] > 0),
            key=lambda item: (-item[0], item[1].chunk_id),
        )[:top_k]
        return [
            RankedChunk(chunk=chunk, score=round(score, 8), rank=index)
            for index, (score, chunk) in enumerate(ranked, 1)
        ]


class DenseVectorRetriever:
    """Cosine retrieval over externally generated dense vectors."""

    def __init__(self, chunks: list[CorpusChunk], vectors: Sequence[Sequence[float]]):
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and dense-vector counts must match")
        dimensions = {len(vector) for vector in vectors}
        if chunks and (not dimensions or 0 in dimensions or len(dimensions) != 1):
            raise ValueError("Dense vectors must have one non-zero shared dimension")
        self.chunks = chunks
        self.vectors = vectors

    @staticmethod
    def similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Dense vector dimensions must match")
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def retrieve(
        self, query_vector: Sequence[float], *, top_k: int = 5
    ) -> list[RankedChunk]:
        scored = [
            (self.similarity(query_vector, vector), chunk)
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
        ]
        ranked = sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:top_k]
        return [
            RankedChunk(chunk=chunk, score=round(score, 8), rank=index)
            for index, (score, chunk) in enumerate(ranked, 1)
        ]


def reciprocal_rank_fusion(
    rankings: list[list[RankedChunk]], *, top_k: int, rrf_k: int = 60
) -> list[RankedChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, CorpusChunk] = {}
    for ranking in rankings:
        for item in ranking:
            chunk_id = item.chunk.chunk_id
            chunks[chunk_id] = item.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + item.rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        RankedChunk(
            chunk=chunks[chunk_id],
            score=round(scores[chunk_id], 8),
            rank=rank,
        )
        for rank, chunk_id in enumerate(ordered, 1)
    ]


def mmr_select(
    candidates: list[RankedChunk],
    vector_retriever: TfidfVectorRetriever,
    *,
    top_k: int,
    lambda_weight: float = 0.7,
) -> list[RankedChunk]:
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError("lambda_weight must be between 0 and 1")
    if not candidates:
        return []
    vectors_by_id = {
        chunk.chunk_id: vector
        for chunk, vector in zip(
            vector_retriever.chunks, vector_retriever.vectors, strict=True
        )
    }
    maximum = max(item.score for item in candidates)
    minimum = min(item.score for item in candidates)

    def relevance(item: RankedChunk) -> float:
        if maximum == minimum:
            return 1.0
        return (item.score - minimum) / (maximum - minimum)

    remaining = list(candidates)
    selected: list[RankedChunk] = []
    while remaining and len(selected) < top_k:
        scored = []
        for item in remaining:
            redundancy = max(
                (
                    vector_retriever.similarity(
                        vectors_by_id[item.chunk.chunk_id],
                        vectors_by_id[chosen.chunk.chunk_id],
                    )
                    for chosen in selected
                ),
                default=0.0,
            )
            mmr_score = lambda_weight * relevance(item) - (1 - lambda_weight) * redundancy
            scored.append((mmr_score, item))
        _, chosen = min(
            scored,
            key=lambda pair: (-pair[0], pair[1].rank, pair[1].chunk.chunk_id),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return [
        RankedChunk(chunk=item.chunk, score=item.score, rank=rank)
        for rank, item in enumerate(selected, 1)
    ]


def dense_mmr_select(
    candidates: list[RankedChunk],
    dense_retriever: DenseVectorRetriever,
    *,
    top_k: int,
    lambda_weight: float = 0.7,
) -> list[RankedChunk]:
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError("lambda_weight must be between 0 and 1")
    if not candidates:
        return []
    vectors_by_id = {
        chunk.chunk_id: vector
        for chunk, vector in zip(
            dense_retriever.chunks, dense_retriever.vectors, strict=True
        )
    }
    maximum = max(item.score for item in candidates)
    minimum = min(item.score for item in candidates)

    def relevance(item: RankedChunk) -> float:
        return 1.0 if maximum == minimum else (item.score - minimum) / (maximum - minimum)

    remaining = list(candidates)
    selected: list[RankedChunk] = []
    while remaining and len(selected) < top_k:
        scored = []
        for item in remaining:
            redundancy = max(
                (
                    dense_retriever.similarity(
                        vectors_by_id[item.chunk.chunk_id],
                        vectors_by_id[chosen.chunk.chunk_id],
                    )
                    for chosen in selected
                ),
                default=0.0,
            )
            score = lambda_weight * relevance(item) - (1 - lambda_weight) * redundancy
            scored.append((score, item))
        _, chosen = min(
            scored,
            key=lambda pair: (-pair[0], pair[1].rank, pair[1].chunk.chunk_id),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return [
        RankedChunk(chunk=item.chunk, score=item.score, rank=rank)
        for rank, item in enumerate(selected, 1)
    ]


def evaluate_retrieval(
    cases: list[LocalBenchmarkCase], chunks: list[CorpusChunk], *, top_k: int = 5
) -> dict[str, Any]:
    retriever = BM25Retriever(chunks)
    rankings = {
        case.case_id: retriever.retrieve(case.question, top_k=top_k) for case in cases
    }
    return evaluate_rankings(cases, rankings, top_k=top_k)


def evaluate_rankings(
    cases: list[LocalBenchmarkCase],
    rankings: dict[str, list[RankedChunk]],
    *,
    top_k: int,
) -> dict[str, Any]:
    results = []
    for case in cases:
        ranked = rankings[case.case_id][:top_k]
        retrieved_documents = [item.chunk.document_id for item in ranked]
        if not case.answerable:
            results.append({
                "case_id": case.case_id,
                "answerable": False,
                "retrieved_chunk_ids": [item.chunk.chunk_id for item in ranked],
            })
            continue
        gold_documents = set(case.gold_document_ids)
        document_recall = len(gold_documents.intersection(retrieved_documents)) / len(gold_documents)
        first_relevant = next(
            (index for index, document_id in enumerate(retrieved_documents, 1) if document_id in gold_documents),
            None,
        )
        seen_relevant_documents: set[str] = set()
        gains = []
        for document_id in retrieved_documents:
            gain = int(
                document_id in gold_documents
                and document_id not in seen_relevant_documents
            )
            gains.append(gain)
            if gain:
                seen_relevant_documents.add(document_id)
        dcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(gains, 1))
        ideal_gains = [1] * min(len(gold_documents), top_k)
        ideal_dcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(ideal_gains, 1))
        evidence_hits = sum(
            any(
                item.chunk.document_id == evidence.document_id
                and _normalize(evidence.quote) in _normalize(item.chunk.text)
                for item in ranked
            )
            for evidence in case.gold_evidence
        )
        results.append({
            "case_id": case.case_id,
            "answerable": True,
            "document_recall_at_k": round(document_recall, 4),
            "evidence_recall_at_k": round(evidence_hits / len(case.gold_evidence), 4),
            "reciprocal_rank": round(1 / first_relevant, 4) if first_relevant else 0.0,
            "ndcg_at_k": round(dcg / ideal_dcg, 4) if ideal_dcg else 0.0,
            "source_diversity_at_k": round(
                len(set(retrieved_documents)) / len(retrieved_documents), 4
            ) if retrieved_documents else 0.0,
            "retrieved_chunk_ids": [item.chunk.chunk_id for item in ranked],
        })
    scored = [result for result in results if result["answerable"]]
    metric_names = (
        "document_recall_at_k",
        "evidence_recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "source_diversity_at_k",
    )
    aggregate = {
        metric: round(sum(result[metric] for result in scored) / len(scored), 4)
        if scored else 0.0
        for metric in metric_names
    }
    return {"top_k": top_k, "evaluated_cases": len(scored), "aggregate": aggregate, "cases": results}


def run_semantic_retrieval_ablation(
    cases: list[LocalBenchmarkCase],
    chunks: list[CorpusChunk],
    chunk_vectors: Sequence[Sequence[float]],
    query_vectors: dict[str, Sequence[float]],
    *,
    top_k: int = 5,
    candidate_k: int = 40,
    rrf_k: int = 60,
    mmr_lambda: float = 0.7,
) -> dict[str, Any]:
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    missing_queries = {case.case_id for case in cases} - query_vectors.keys()
    if missing_queries:
        raise ValueError(f"Missing query vectors: {', '.join(sorted(missing_queries))}")
    bm25 = BM25Retriever(chunks)
    dense = DenseVectorRetriever(chunks, chunk_vectors)
    methods: dict[str, dict[str, list[RankedChunk]]] = {
        "bm25": {},
        "semantic_vector": {},
        "bm25_semantic_rrf": {},
        "semantic_rrf_mmr": {},
    }
    for case in cases:
        bm25_ranking = bm25.retrieve(case.question, top_k=candidate_k)
        dense_ranking = dense.retrieve(query_vectors[case.case_id], top_k=candidate_k)
        fused = reciprocal_rank_fusion(
            [bm25_ranking, dense_ranking], top_k=candidate_k, rrf_k=rrf_k
        )
        methods["bm25"][case.case_id] = bm25_ranking
        methods["semantic_vector"][case.case_id] = dense_ranking
        methods["bm25_semantic_rrf"][case.case_id] = fused
        methods["semantic_rrf_mmr"][case.case_id] = dense_mmr_select(
            fused, dense, top_k=top_k, lambda_weight=mmr_lambda
        )
    return {
        "configuration": {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "mmr_lambda": mmr_lambda,
        },
        "methods": {
            name: evaluate_rankings(cases, rankings, top_k=top_k)
            for name, rankings in methods.items()
        },
    }


def retrieve_with_dense_strategy(
    query: str,
    chunks: list[CorpusChunk],
    chunk_vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    *,
    strategy: str,
    top_k: int,
    candidate_k: int = 40,
    rrf_k: int = 60,
    mmr_lambda: float = 0.7,
) -> list[RankedChunk]:
    """Retrieve with cached semantic vectors using the production strategies."""
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    dense = DenseVectorRetriever(chunks, chunk_vectors)
    dense_ranking = dense.retrieve(query_vector, top_k=candidate_k)
    if strategy == "semantic_vector":
        return [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(dense_ranking[:top_k], 1)
        ]
    if strategy not in {"bm25_semantic_rrf", "semantic_rrf_mmr"}:
        raise ValueError(f"Unknown semantic retrieval strategy: {strategy}")
    bm25_ranking = BM25Retriever(chunks).retrieve(query, top_k=candidate_k)
    fused = reciprocal_rank_fusion(
        [bm25_ranking, dense_ranking], top_k=candidate_k, rrf_k=rrf_k
    )
    if strategy == "bm25_semantic_rrf":
        return [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(fused[:top_k], 1)
        ]
    return dense_mmr_select(
        fused, dense, top_k=top_k, lambda_weight=mmr_lambda
    )


class LocalSemanticRetriever:
    """Production semantic retriever backed by a validated embedding cache."""

    def __init__(
        self,
        *,
        corpus_path: Path,
        model_name: str,
        model_cache_path: Path,
        embedding_cache_path: Path,
        threads: int = 8,
    ):
        try:
            import numpy as np
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "Semantic local search requires the benchmark dependencies; "
                "install the project with the 'benchmark' extra"
            ) from error

        chunks_path = corpus_path / "chunks.jsonl"
        metadata_path = embedding_cache_path.with_suffix(".json")
        if not embedding_cache_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "Semantic embedding cache is missing; run "
                "experiments/benchmarks/evaluate_semantic_retrieval.py first"
            )
        self.chunks = read_jsonl(chunks_path, CorpusChunk)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        expected_ids = [chunk.chunk_id for chunk in self.chunks]
        if metadata.get("model") != model_name:
            raise ValueError("Semantic cache model does not match configured model")
        if metadata.get("corpus_fingerprint") != fingerprint:
            raise ValueError("Semantic cache does not match the configured corpus")
        if metadata.get("chunk_ids") != expected_ids:
            raise ValueError("Semantic cache chunk order does not match the corpus")
        with np.load(embedding_cache_path, allow_pickle=False) as stored:
            self.chunk_vectors = stored["vectors"]
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(model_cache_path),
            threads=threads,
            local_files_only=True,
        )

    def retrieve(
        self,
        query: str,
        *,
        strategy: str,
        top_k: int,
        candidate_k: int,
        rrf_k: int,
        mmr_lambda: float,
    ) -> list[RankedChunk]:
        query_vector = next(self.model.query_embed([query]))
        return retrieve_with_dense_strategy(
            query,
            self.chunks,
            self.chunk_vectors,
            query_vector,
            strategy=strategy,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            mmr_lambda=mmr_lambda,
        )


def run_retrieval_ablation(
    cases: list[LocalBenchmarkCase],
    chunks: list[CorpusChunk],
    *,
    top_k: int = 5,
    candidate_k: int = 40,
    rrf_k: int = 60,
    mmr_lambda: float = 0.7,
) -> dict[str, Any]:
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
        bm25_ranking = bm25.retrieve(case.question, top_k=candidate_k)
        vector_ranking = vector.retrieve(case.question, top_k=candidate_k)
        fused = reciprocal_rank_fusion(
            [bm25_ranking, vector_ranking], top_k=candidate_k, rrf_k=rrf_k
        )
        methods["bm25"][case.case_id] = bm25_ranking
        methods["tfidf_vector"][case.case_id] = vector_ranking
        methods["rrf_hybrid"][case.case_id] = fused
        methods["rrf_mmr"][case.case_id] = mmr_select(
            fused, vector, top_k=top_k, lambda_weight=mmr_lambda
        )
    return {
        "configuration": {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "mmr_lambda": mmr_lambda,
            "vector_method": "word-level TF-IDF cosine; not a semantic embedding model",
        },
        "methods": {
            name: evaluate_rankings(cases, rankings, top_k=top_k)
            for name, rankings in methods.items()
        },
    }


def retrieve_with_strategy(
    query: str,
    chunks: list[CorpusChunk],
    *,
    strategy: str,
    top_k: int,
    candidate_k: int = 40,
    rrf_k: int = 60,
    mmr_lambda: float = 0.7,
) -> list[RankedChunk]:
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    bm25 = BM25Retriever(chunks)
    if strategy == "bm25":
        return bm25.retrieve(query, top_k=top_k)
    vector = TfidfVectorRetriever(chunks)
    if strategy == "tfidf_vector":
        return vector.retrieve(query, top_k=top_k)
    bm25_ranking = bm25.retrieve(query, top_k=candidate_k)
    vector_ranking = vector.retrieve(query, top_k=candidate_k)
    fused = reciprocal_rank_fusion(
        [bm25_ranking, vector_ranking], top_k=candidate_k, rrf_k=rrf_k
    )
    if strategy == "rrf_hybrid":
        return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(fused[:top_k], 1)]
    if strategy == "rrf_mmr":
        return mmr_select(fused, vector, top_k=top_k, lambda_weight=mmr_lambda)
    raise ValueError(f"Unknown local retrieval strategy: {strategy}")


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))
