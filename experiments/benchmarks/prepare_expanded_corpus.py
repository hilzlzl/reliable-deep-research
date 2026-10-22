"""Acquire the 24-document expanded experiment corpus (raw text stays local)."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from open_deep_research.local_benchmark import (
    CorpusDocument, chunk_documents, extract_text, read_jsonl, write_jsonl,
)

SOURCES = [
    ("rag_original", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "ai_applications", "https://arxiv.org/abs/2005.11401", "https://arxiv.org/pdf/2005.11401"),
    ("rag_dpr", "Dense Passage Retrieval for Open-Domain Question Answering", "ai_applications", "https://arxiv.org/abs/2004.04906", "https://arxiv.org/pdf/2004.04906"),
    ("rag_lost_middle", "Lost in the Middle: How Language Models Use Long Contexts", "ai_applications", "https://arxiv.org/abs/2307.03172", "https://arxiv.org/pdf/2307.03172"),
    ("adaptation_lora", "LoRA: Low-Rank Adaptation of Large Language Models", "ai_applications", "https://arxiv.org/abs/2106.09685", "https://arxiv.org/pdf/2106.09685"),
    ("graph_persistence", "LangGraph persistence", "agent_systems", "https://docs.langchain.com/oss/python/langgraph/persistence", "https://docs.langchain.com/oss/python/langgraph/persistence.md"),
    ("graph_checkpointers", "LangGraph checkpointers", "agent_systems", "https://docs.langchain.com/oss/python/langgraph/checkpointers", "https://docs.langchain.com/oss/python/langgraph/checkpointers.md"),
    ("graph_interrupts", "LangGraph interrupts", "agent_systems", "https://docs.langchain.com/oss/python/langgraph/interrupts", "https://docs.langchain.com/oss/python/langgraph/interrupts.md"),
    ("graph_time_travel", "LangGraph time travel", "agent_systems", "https://docs.langchain.com/oss/python/langgraph/use-time-travel", "https://docs.langchain.com/oss/python/langgraph/use-time-travel.md"),
    ("mcp_tools", "MCP 2025-06-18 tools", "agent_systems", "https://modelcontextprotocol.io/specification/2025-06-18/server/tools", "https://modelcontextprotocol.io/specification/2025-06-18/server/tools.md"),
    ("mcp_transports", "MCP 2025-06-18 transports", "software_engineering", "https://modelcontextprotocol.io/specification/2025-06-18/basic/transports", "https://modelcontextprotocol.io/specification/2025-06-18/basic/transports.md"),
    ("mcp_cancellation", "MCP 2025-06-18 cancellation", "software_engineering", "https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/cancellation", "https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/cancellation.md"),
    ("sqlite_atomic", "SQLite atomic commit", "software_engineering", "https://sqlite.org/atomiccommit.html", "https://sqlite.org/atomiccommit.html"),
    ("sqlite_wal", "SQLite write-ahead logging", "software_engineering", "https://sqlite.org/wal.html", "https://sqlite.org/wal.html"),
    ("sqlite_transactions", "SQLite transactions", "software_engineering", "https://sqlite.org/lang_transaction.html", "https://sqlite.org/lang_transaction.html"),
    ("python_asyncio", "Python 3.11 asyncio tasks", "software_engineering", "https://docs.python.org/3.11/library/asyncio-task.html", "https://docs.python.org/3.11/library/asyncio-task.html"),
    ("pytest_practices", "pytest good integration practices", "software_engineering", "https://docs.pytest.org/en/stable/explanation/goodpractices.html", "https://docs.pytest.org/en/stable/explanation/goodpractices.html"),
    ("github_cache", "GitHub Actions dependency caching", "software_engineering", "https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows", "https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows"),
]


def acquire(source: tuple, output: Path) -> tuple[CorpusDocument, dict]:
    source_id, title, domain, locator, url = source
    suffix = ".pdf" if "/pdf/" in url else ".md" if url.endswith(".md") else ".html"
    raw = output / "raw" / (source_id + suffix)
    if not raw.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "ReliableResearchBenchmark/0.2"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if suffix == ".pdf" and not data.startswith(b"%PDF"):
            raise ValueError(f"Not a PDF: {source_id}")
        raw.write_bytes(data)
    text = extract_text(raw)
    if len(text) < 500:
        raise ValueError(f"Unexpectedly short document: {source_id}")
    document = CorpusDocument(
        document_id=source_id, title=title, domain=domain,
        source_type="paper" if suffix == ".pdf" else "official_documentation_snapshot",
        source_url=locator, content=text,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    return document, {
        "source_id": source_id, "title": title, "domain": domain,
        "locator": locator, "acquisition_url": url,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "content_sha256": document.content_sha256, "characters": len(text),
        "redistribution": "metadata_only", "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark_data/expanded_20261021"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/expanded_20261021/sources.jsonl"))
    args = parser.parse_args()
    (args.output / "raw").mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.existing_corpus / "documents.jsonl", CorpusDocument)
    if len(existing) != 7:
        raise ValueError("Expected the original seven-paper corpus")
    records = [{
        "source_id": d.document_id, "title": d.title, "domain": d.domain,
        "locator": d.source_url, "content_sha256": d.content_sha256,
        "characters": len(d.content), "redistribution": "metadata_only",
        "provenance": "Original seven-paper local snapshot; unchanged content hash",
    } for d in existing]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(acquire, source, args.output) for source in SOURCES]
        for source, future in zip(SOURCES, futures):
            try:
                document, record = future.result()
            except Exception as exc:
                print(source[0], type(exc).__name__, str(exc), flush=True)
                raise
            existing.append(document)
            records.append(record)
            print(source[0], len(document.content), flush=True)
    write_jsonl(args.output / "documents.jsonl", existing)
    write_jsonl(args.output / "chunks.jsonl", chunk_documents(existing, chunk_size=1200, overlap=200))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    print(f"Prepared {len(existing)} documents")


if __name__ == "__main__":
    main()
