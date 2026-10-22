"""Run local semantic retrieval and hybrid ablations with cached embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from open_deep_research.local_benchmark import (
    CorpusChunk,
    CorpusDocument,
    LocalBenchmarkCase,
    read_jsonl,
    run_semantic_retrieval_ablation,
    validate_benchmark,
)


def load_or_build_chunk_vectors(
    model: TextEmbedding,
    *,
    model_name: str,
    chunks: list[CorpusChunk],
    chunks_path: Path,
    cache_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, bool, str]:
    corpus_fingerprint = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("model") == model_name
            and metadata.get("corpus_fingerprint") == corpus_fingerprint
            and metadata.get("chunk_ids") == [chunk.chunk_id for chunk in chunks]
        ):
            with np.load(cache_path, allow_pickle=False) as stored:
                return stored["vectors"], True, corpus_fingerprint

    texts = [f"{chunk.title}\n\n{chunk.text}" for chunk in chunks]
    vectors = np.asarray(
        list(model.passage_embed(texts, batch_size=batch_size)), dtype=np.float32
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, vectors=vectors)
    metadata_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "corpus_fingerprint": corpus_fingerprint,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "dimensions": int(vectors.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return vectors, False, corpus_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--model-cache", type=Path, default=Path("benchmark_data/models"))
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("benchmark_data/researchbench_en/embeddings/bge-small-en-v1.5.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    chunks_path = args.corpus_dir / "chunks.jsonl"
    documents = read_jsonl(args.corpus_dir / "documents.jsonl", CorpusDocument)
    chunks = read_jsonl(chunks_path, CorpusChunk)
    cases = read_jsonl(args.questions, LocalBenchmarkCase)
    validation = validate_benchmark(documents, cases)
    if not validation["valid"]:
        raise ValueError("Benchmark validation failed: " + "; ".join(validation["errors"]))

    started = time.perf_counter()
    model = TextEmbedding(
        model_name=args.model,
        cache_dir=str(args.model_cache),
        threads=args.threads,
    )
    chunk_vectors, cache_hit, corpus_fingerprint = load_or_build_chunk_vectors(
        model,
        model_name=args.model,
        chunks=chunks,
        chunks_path=chunks_path,
        cache_path=args.embedding_cache,
        batch_size=args.batch_size,
    )
    query_matrix = np.asarray(
        list(
            model.query_embed(
                [case.question for case in cases], batch_size=args.batch_size
            )
        ),
        dtype=np.float32,
    )
    query_vectors = {
        case.case_id: query_matrix[index] for index, case in enumerate(cases)
    }
    retrieval = run_semantic_retrieval_ablation(
        cases,
        chunks,
        chunk_vectors,
        query_vectors,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        mmr_lambda=args.mmr_lambda,
    )
    retrieval["configuration"].update(
        {
            "semantic_model": args.model,
            "embedding_dimensions": int(chunk_vectors.shape[1]),
            "corpus_fingerprint": corpus_fingerprint,
            "chunk_embedding_cache_hit": cache_hit,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    result = {"validation": validation, "retrieval": retrieval}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
