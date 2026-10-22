"""Tests for automatic source and evidence capture."""

import asyncio
import hashlib

from open_deep_research.local_benchmark import CorpusDocument, chunk_documents, write_jsonl
from open_deep_research.run_budget import RunBudget
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.utils import local_corpus_search, persist_search_evidence


def test_search_evidence_is_deduplicated(tmp_path):
    db_path = tmp_path / "runtime.db"
    config = {
        "configurable": {
            "runtime_db_path": str(db_path),
            "current_task_id": "research:abc",
        }
    }

    first = persist_search_evidence(
        config,
        url="https://example.com/paper",
        title="Paper",
        content="The method uses a JSON-RPC transport.",
    )
    second = persist_search_evidence(
        config,
        url="https://example.com/paper",
        title="Paper duplicate",
        content="The method uses a JSON-RPC transport.",
    )

    assert first == second
    assert RuntimeStore(db_path).evidence_counts() == {
        "documents": 1,
        "evidence": 1,
        "claims": 0,
        "conflicts": 0,
    }


def test_duplicate_content_reuses_existing_document_id(tmp_path):
    db_path = tmp_path / "runtime.db"
    config = {
        "configurable": {
            "runtime_db_path": str(db_path),
            "current_task_id": "research:abc",
        }
    }
    first = persist_search_evidence(
        config,
        url="https://example.com/original",
        title="Original",
        content="identical content",
    )
    second = persist_search_evidence(
        config,
        url="https://mirror.example.com/copy",
        title="Copy",
        content="identical content",
    )

    assert first == second
    assert RuntimeStore(db_path).evidence_counts()["documents"] == 1


def test_search_evidence_requires_task_context(tmp_path):
    result = persist_search_evidence(
        {"configurable": {"runtime_db_path": str(tmp_path / "runtime.db")}},
        url="https://example.com",
        title="Example",
        content="content",
    )

    assert result is None


def test_local_corpus_search_is_deterministic_and_persists_evidence(tmp_path):
    corpus_dir = tmp_path / "corpus"
    content = "Geometric kernel execution detects invalid topology and failed boolean operations."
    document = CorpusDocument(
        document_id="cad-kernel",
        title="Kernel validation",
        domain="cad_generation",
        source_type="official_documentation",
        source_url="https://example.test/kernel",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    write_jsonl(corpus_dir / "chunks.jsonl", chunk_documents([document], chunk_size=200, overlap=20))
    db_path = tmp_path / "runtime.db"
    budget = RunBudget(max_search_calls=1)
    config = {
        "configurable": {
            "local_corpus_path": str(corpus_dir),
            "runtime_db_path": str(db_path),
            "current_task_id": "research:local",
            "max_search_queries": 1,
            "max_search_results": 3,
            "local_retrieval_strategy": "rrf_mmr",
            "local_retrieval_candidate_k": 3,
            "run_budget": budget,
        }
    }

    output = asyncio.run(
        local_corpus_search.ainvoke(
            {"queries": ["invalid topology boolean operations"], "max_results": 3},
            config,
        )
    )

    assert "Document ID: cad-kernel" in output
    assert "Chunk ID: cad-kernel:0" in output
    assert "Retrieval strategy: rrf_mmr" in output
    assert RuntimeStore(db_path).evidence_counts()["evidence"] == 1
    assert budget.snapshot().search_calls == 1
