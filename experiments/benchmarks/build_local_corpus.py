"""Build deterministic English corpus documents and chunks from local raw files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_deep_research.local_benchmark import (
    SourceRecord,
    build_documents,
    chunk_documents,
    read_jsonl,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    sources = read_jsonl(args.manifest, SourceRecord)
    documents, missing = build_documents(sources, args.raw_dir)
    chunks = chunk_documents(
        documents, chunk_size=args.chunk_size, overlap=args.chunk_overlap
    )
    write_jsonl(args.output_dir / "documents.jsonl", documents)
    write_jsonl(args.output_dir / "chunks.jsonl", chunks)
    report = {
        "manifest_sources": len(sources),
        "imported_documents": len(documents),
        "chunks": len(chunks),
        "missing_source_ids": missing,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
