"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import hashlib
import json
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Type, TypeVar

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.adaptive_retrieval import (
    RetrievalAction,
    RetrievalCandidate,
    assess_retrieval_gain,
    assess_local_results,
    fuse_candidates,
    rerank_candidates,
    select_minimal_facet_cover,
)
from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.evidence import Document, Evidence
from open_deep_research.local_benchmark import (
    CorpusChunk,
    LocalSemanticRetriever,
    read_jsonl,
    retrieve_with_strategy,
)
from open_deep_research.model_routing import (
    ModelRoute,
    select_model_route,
    select_paid_fallback_route,
)
from open_deep_research.observability import record_event
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.runtime_store import RuntimeStore
from open_deep_research.run_budget import (
    BudgetExceeded,
    budget_from_config,
    budgeted_ainvoke,
)
from open_deep_research.state import ResearchComplete, Summary

StructuredT = TypeVar("StructuredT")


def model_config_for_node(
    configurable: Configuration,
    config: RunnableConfig | None,
    *,
    node: str,
    default_model: str,
    default_max_tokens: int,
) -> tuple[dict[str, Any], ModelRoute]:
    """Build a LangChain config and record the provider-neutral route."""
    route = select_model_route(
        configurable,
        node=node,
        default_model=default_model,
        default_max_tokens=default_max_tokens,
    )
    api_key = os.getenv(route.api_key_env) if route.api_key_env else None
    if route.tier.value == "legacy":
        api_key = get_api_key_for_model(route.model, config or {})
    record_event(
        config,
        "model_route_selected",
        node,
        data={
            "tier": route.tier.value,
            "model": route.model,
            "reason": route.reason,
        },
    )
    return (
        {
            "model": route.model,
            "max_tokens": route.max_tokens,
            "api_key": api_key,
            "base_url": route.base_url,
            "tags": ["langsmith:nostream"],
        },
        route,
    )


def paid_fallback_config_for_node(
    configurable: Configuration,
    config: RunnableConfig | None,
    *,
    node: str,
    primary_route: ModelRoute,
    default_max_tokens: int,
    error: Exception,
) -> dict[str, Any] | None:
    """Build and trace a paid API retry for one failed local node."""
    route = select_paid_fallback_route(
        configurable,
        primary=primary_route,
        default_max_tokens=default_max_tokens,
    )
    if route is None:
        return None
    record_event(
        config,
        "model_route_escalated",
        node,
        data={
            "from_tier": primary_route.tier.value,
            "to_tier": route.tier.value,
            "reason": route.reason,
            "error_type": type(error).__name__,
        },
    )
    return {
        "model": route.model,
        "max_tokens": route.max_tokens,
        "api_key": os.getenv(route.api_key_env) if route.api_key_env else None,
        "base_url": route.base_url,
        "tags": ["langsmith:nostream"],
    }


def persist_search_evidence(
    config: RunnableConfig | None,
    *,
    url: str,
    title: str,
    content: str,
) -> tuple[str, str] | None:
    """Persist one searched source and a coarse evidence record.

    This intentionally reuses existing search output and makes no additional
    model call. Fine-grained claim extraction is a later pipeline stage.
    """
    configurable = (config or {}).get("configurable", {})
    db_path = configurable.get("runtime_db_path")
    task_id = configurable.get("current_task_id")
    if not db_path or not task_id or not content.strip():
        return None
    document_id = "doc:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    content_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    evidence_id = "ev:" + hashlib.sha256(
        f"{task_id}\0{document_id}\0{content_hash}".encode("utf-8")
    ).hexdigest()[:20]
    store = RuntimeStore(db_path)
    inserted = store.save_document(
        Document(
            document_id=document_id,
            url=url,
            title=title,
            content_hash=content_hash,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    if not inserted:
        document_id = store.document_id_for_hash(content_hash) or document_id
        evidence_id = "ev:" + hashlib.sha256(
            f"{task_id}\0{document_id}\0{content_hash}".encode("utf-8")
        ).hexdigest()[:20]
    store.save_evidence(
        Evidence(
            evidence_id=evidence_id,
            task_id=task_id,
            document_id=document_id,
            excerpt=content[:2000],
            claim_text=content[:500],
            confidence=0.5,
        )
    )
    return document_id, evidence_id


def _extract_json_object(content: Any) -> dict[str, Any]:
    """Extract a JSON object from plain model text or a fenced code block."""
    text = str(content).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object")
    return value


async def ainvoke_structured_with_fallback(
    model: BaseChatModel,
    schema: Type[StructuredT],
    messages: list[Any],
    *,
    retries: int = 3,
    native_structured_output: bool = True,
    config: RunnableConfig | None = None,
    node: str = "structured_output",
) -> StructuredT:
    """Invoke structured output, then degrade to JSON text parsing if needed.

    Some OpenAI-compatible gateways accept ordinary chat but reject JSON Schema
    or tool-based structured output. The fallback keeps the runtime usable while
    preserving the typed contract at this boundary.
    """
    structured_error: Exception | None = None
    if native_structured_output:
        structured = model.with_structured_output(schema).with_retry(
            stop_after_attempt=retries
        )
        try:
            return await budgeted_ainvoke(structured, messages, config, node=node)
        except BudgetExceeded:
            raise
        except Exception as error:
            structured_error = error

    fallback_prompt = (
        "Return only one valid JSON object, with no markdown fences. "
        f"It must satisfy this schema: {schema.model_json_schema()}\n\n"
        f"Original request:\n{messages[-1].content}"
    )
    try:
        format_retries = Configuration.from_runnable_config(config).structured_json_repair_attempts
        for attempt in range(format_retries + 1):
            response = await budgeted_ainvoke(
                model, [HumanMessage(content=fallback_prompt)], config, node=f"{node}_fallback"
            )
            try:
                return schema.model_validate(_extract_json_object(response.content))
            except (ValueError, TypeError):
                if attempt == format_retries:
                    raise
                record_event(config, "structured_json_retry", node,
                             data={"attempt": attempt + 1, "reason": "invalid_json_or_schema"})
                fallback_prompt += (
                    "\nYour previous response failed JSON/schema validation. Retry the same "
                    "request with exactly one complete JSON object matching the schema."
                )
    except BudgetExceeded:
        raise
    except Exception as fallback_error:
        prefix = f"Structured output failed: {structured_error}; " if structured_error else ""
        raise RuntimeError(
            f"{prefix}JSON fallback failed: {fallback_error}"
        ) from fallback_error

##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)

LOCAL_SEARCH_DESCRIPTION = (
    "Search the fixed local research corpus. Use this for reproducible benchmark research."
)

HYBRID_SEARCH_DESCRIPTION = (
    "Search the local corpus first, then search the web only when deterministic "
    "coverage and source-diversity checks find the local results insufficient."
)

SEMANTIC_LOCAL_STRATEGIES = {
    "semantic_vector",
    "bm25_semantic_rrf",
    "semantic_rrf_mmr",
}


@lru_cache(maxsize=4)
def _get_local_semantic_retriever(
    corpus_path: str,
    model_name: str,
    model_cache_path: str,
    embedding_cache_path: str,
    threads: int,
) -> LocalSemanticRetriever:
    return LocalSemanticRetriever(
        corpus_path=Path(corpus_path),
        model_name=model_name,
        model_cache_path=Path(model_cache_path),
        embedding_cache_path=Path(embedding_cache_path),
        threads=threads,
    )


def _retrieve_local_semantic(
    query: str,
    *,
    corpus_path: str,
    model_name: str,
    model_cache_path: str,
    embedding_cache_path: str,
    threads: int,
    strategy: str,
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    mmr_lambda: float,
) -> list[Any]:
    retriever = _get_local_semantic_retriever(
        corpus_path,
        model_name,
        model_cache_path,
        embedding_cache_path,
        threads,
    )
    return retriever.retrieve(
        query,
        strategy=strategy,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        mmr_lambda=mmr_lambda,
    )


@tool(description=LOCAL_SEARCH_DESCRIPTION)
async def local_corpus_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None,
) -> str:
    """Retrieve deterministic lexical or cached semantic local-corpus chunks."""
    selected = await _rank_local_chunks(queries, max_results, config)
    return _render_local_chunks(selected, config)


def _local_candidates(selected: list[Any]) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            document_id=item.chunk.document_id,
            chunk_id=item.chunk.chunk_id,
            source="local",
            sources=["local"],
            url=item.chunk.source_url,
            title=item.chunk.title,
            text=item.chunk.text,
            score=item.score,
            rank=index,
        )
        for index, item in enumerate(selected, 1)
    ]


def _render_retrieval_candidates(
    candidates: list[RetrievalCandidate],
    config: RunnableConfig | None,
    *,
    heading: str,
) -> str:
    if not candidates:
        return "No valid search results found."
    rendered = []
    for index, candidate in enumerate(candidates, 1):
        persist_search_evidence(
            config,
            url=candidate.url or f"{candidate.source}://{candidate.document_id}",
            title=candidate.title,
            content=candidate.text,
        )
        source_names = ", ".join(candidate.sources or [candidate.source])
        rendered.append(
            f"--- {heading} {index}: {candidate.title} ---\n"
            f"Document ID: {candidate.document_id}\n"
            f"Chunk ID: {candidate.chunk_id or candidate.document_id}\n"
            f"Sources: {source_names}\n"
            f"URL: {candidate.url}\n"
            f"Fusion score: {candidate.score}\n"
            f"Relevance score: {candidate.relevance_score}"
            f"\nSource quality: {candidate.source_quality}"
            f"\nCovered facets: {', '.join(candidate.covered_facets) or 'none'}"
            f"\n\n{candidate.text}"
        )
    return "\n\n".join(rendered)


def _consecutive_low_gain_rounds(config: RunnableConfig | None) -> int:
    """Read the current run trace and count trailing low-gain web rounds."""
    configurable = (config or {}).get("configurable", {})
    if not configurable.get("observability_enabled", True):
        return 0
    event_path = configurable.get("event_log_path")
    if not event_path or not Path(event_path).is_file():
        return 0
    run_id = str(
        configurable.get("run_id") or configurable.get("thread_id") or "unscoped"
    )
    outcomes: list[bool] = []
    try:
        for line in Path(event_path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if (
                str(event.get("run_id")) == run_id
                and event.get("event_type") == "retrieval_marginal_gain"
            ):
                outcomes.append(bool(event.get("data", {}).get("useful")))
    except (OSError, json.JSONDecodeError):
        return 0
    count = 0
    for useful in reversed(outcomes):
        if useful:
            break
        count += 1
    return count


async def _rank_local_chunks(
    queries: list[str], max_results: int, config: RunnableConfig | None
) -> list[Any]:
    """Return local ranked chunks without rendering or persisting them."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.local_corpus_path:
        raise ToolException("local_corpus_path is required for local-first search")
    chunks_path = Path(configurable.local_corpus_path) / "chunks.jsonl"
    if not chunks_path.is_file():
        raise ToolException(f"Local corpus chunks not found: {chunks_path}")

    bounded_queries = queries[: configurable.max_search_queries]
    budget = budget_from_config(config)
    if budget:
        budget.consume("search_calls", len(bounded_queries))
    limit = min(max_results, configurable.max_search_results)
    retrieval_limit = (
        max(limit, configurable.local_retrieval_candidate_k)
        if configurable.search_api == SearchAPI.HYBRID
        else limit
    )
    semantic_cache_path = None
    chunks = []
    if configurable.local_retrieval_strategy in SEMANTIC_LOCAL_STRATEGIES:
        semantic_cache_path = configurable.local_embedding_cache_path or str(
            Path(configurable.local_corpus_path).parent
            / "embeddings"
            / "bge-small-en-v1.5.npz"
        )
    else:
        chunks = read_jsonl(chunks_path, CorpusChunk)

    best_by_chunk: dict[str, Any] = {}
    for query in bounded_queries:
        if semantic_cache_path is not None:
            ranking = _retrieve_local_semantic(
                query,
                corpus_path=configurable.local_corpus_path,
                model_name=configurable.local_embedding_model,
                model_cache_path=configurable.local_model_cache_path,
                embedding_cache_path=semantic_cache_path,
                threads=configurable.local_embedding_threads,
                strategy=configurable.local_retrieval_strategy,
                top_k=retrieval_limit,
                candidate_k=max(retrieval_limit, configurable.local_retrieval_candidate_k),
                rrf_k=configurable.local_retrieval_rrf_k,
                mmr_lambda=configurable.local_retrieval_mmr_lambda,
            )
        else:
            ranking = retrieve_with_strategy(
                query,
                chunks,
                strategy=configurable.local_retrieval_strategy,
                top_k=retrieval_limit,
                candidate_k=max(retrieval_limit, configurable.local_retrieval_candidate_k),
                rrf_k=configurable.local_retrieval_rrf_k,
                mmr_lambda=configurable.local_retrieval_mmr_lambda,
            )
        for ranked in ranking:
            existing = best_by_chunk.get(ranked.chunk.chunk_id)
            if existing is None or ranked.score > existing.score:
                best_by_chunk[ranked.chunk.chunk_id] = ranked
    ranked_values = sorted(
        best_by_chunk.values(), key=lambda item: (-item.score, item.chunk.chunk_id)
    )
    if configurable.search_api == SearchAPI.HYBRID:
        reranked = rerank_candidates(
            bounded_queries,
            _local_candidates(ranked_values),
            min_relevance=configurable.hybrid_min_candidate_relevance,
            min_source_quality=configurable.hybrid_min_source_quality,
            max_per_document=configurable.hybrid_max_chunks_per_document,
        )
        ranked_by_chunk = {item.chunk.chunk_id: item for item in ranked_values}
        selected_candidates = select_minimal_facet_cover(
            bounded_queries,
            reranked,
            min_results=configurable.hybrid_local_min_results,
            limit=limit,
        )
        return [ranked_by_chunk[item.chunk_id] for item in selected_candidates]
    return ranked_values[:limit]


def _render_local_chunks(selected: list[Any], config: RunnableConfig | None) -> str:
    if not selected:
        return "No matching results were found in the fixed local corpus."
    configurable = Configuration.from_runnable_config(config)
    rendered = []
    for index, ranked in enumerate(selected, 1):
        chunk = ranked.chunk
        persist_search_evidence(
            config,
            url=f"{chunk.source_url}#chunk={chunk.chunk_id}",
            title=chunk.title,
            content=chunk.text,
        )
        rendered.append(
            f"--- LOCAL SOURCE {index}: {chunk.title} ---\n"
            f"Document ID: {chunk.document_id}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"URL: {chunk.source_url}\n"
            f"Retrieval strategy: {configurable.local_retrieval_strategy}\n"
            f"Retrieval score: {ranked.score}\n\n{chunk.text}"
        )
    return "\n\n".join(rendered)


@tool(description=HYBRID_SEARCH_DESCRIPTION)
async def adaptive_hybrid_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None,
) -> str:
    """Use local evidence when sufficient and otherwise add web evidence."""
    configurable = Configuration.from_runnable_config(config)
    bounded_queries = queries[: configurable.max_search_queries]
    selected = await _rank_local_chunks(bounded_queries, max_results, config)
    raw_local_candidates = _local_candidates(selected)
    local_candidates = rerank_candidates(
        bounded_queries,
        raw_local_candidates,
        min_relevance=configurable.hybrid_min_candidate_relevance,
        min_source_quality=configurable.hybrid_min_source_quality,
        max_per_document=configurable.hybrid_max_chunks_per_document,
    )
    record_event(
        config,
        "retrieval_candidates_reranked",
        "adaptive_hybrid_search",
        data={
            "source": "local",
            "input_candidates": len(raw_local_candidates),
            "retained_candidates": len(local_candidates),
            "rejected_candidates": len(raw_local_candidates) - len(local_candidates),
        },
    )
    local_output = _render_retrieval_candidates(
        local_candidates, config, heading="LOCAL SOURCE"
    )
    assessment = assess_local_results(
        bounded_queries,
        local_candidates,
        min_results=configurable.hybrid_local_min_results,
        min_unique_documents=configurable.hybrid_local_min_unique_documents,
        min_query_term_coverage=configurable.hybrid_local_min_query_term_coverage,
        min_facet_coverage=configurable.hybrid_min_facet_coverage,
        min_facet_relevance=configurable.hybrid_min_facet_relevance,
    )
    record_event(
        config,
        "retrieval_route_selected",
        "adaptive_hybrid_search",
        data=assessment.model_dump(mode="json"),
    )
    if assessment.action == RetrievalAction.ACCEPT_LOCAL:
        return local_output

    low_gain_rounds = _consecutive_low_gain_rounds(config)
    if low_gain_rounds >= configurable.hybrid_max_consecutive_low_gain:
        record_event(
            config,
            "retrieval_stopped_low_gain",
            "adaptive_hybrid_search",
            data={
                "consecutive_low_gain_rounds": low_gain_rounds,
                "limit": configurable.hybrid_max_consecutive_low_gain,
            },
        )
        return (
            f"{local_output}\n\n--- WEB SEARCH STOPPED ---\n"
            "Further web retrieval was skipped after repeated low-gain rounds."
        )

    try:
        raw_web_candidates = await _retrieve_tavily_candidates(
            bounded_queries,
            max_results=max_results,
            topic=topic,
            config=config,
        )
    except BudgetExceeded:
        raise
    except Exception as error:
        record_event(
            config,
            "retrieval_web_escalation_failed",
            "adaptive_hybrid_search",
            data={"error_type": type(error).__name__, "error": str(error)},
        )
        return (
            f"{local_output}\n\n--- WEB ESCALATION UNAVAILABLE ---\n"
            "Local evidence did not pass the configured quality gate, but web "
            f"search failed: {type(error).__name__}."
        )
    web_candidates = rerank_candidates(
        bounded_queries,
        raw_web_candidates,
        min_relevance=configurable.hybrid_min_candidate_relevance,
        min_source_quality=configurable.hybrid_min_source_quality,
    )
    record_event(
        config,
        "retrieval_candidates_reranked",
        "adaptive_hybrid_search",
        data={
            "source": "web",
            "input_candidates": len(raw_web_candidates),
            "retained_candidates": len(web_candidates),
            "rejected_candidates": len(raw_web_candidates) - len(web_candidates),
        },
    )
    gain = assess_retrieval_gain(
        bounded_queries,
        local_candidates,
        web_candidates,
        min_new_documents=configurable.hybrid_min_new_documents,
        min_query_term_coverage_gain=(
            configurable.hybrid_min_query_term_coverage_gain
        ),
        min_facet_relevance=configurable.hybrid_min_facet_relevance,
    )
    record_event(
        config,
        "retrieval_web_escalation_succeeded",
        "adaptive_hybrid_search",
        data={
            "queries": bounded_queries,
            "results": len(web_candidates),
            "raw_results": len(raw_web_candidates),
        },
    )
    record_event(
        config,
        "retrieval_marginal_gain",
        "adaptive_hybrid_search",
        data=gain.model_dump(mode="json"),
    )
    if not gain.useful:
        return (
            f"{local_output}\n\n--- LOW-GAIN WEB SUPPLEMENT REJECTED ---\n"
            "The web results added insufficient new documents, content, or query "
            "coverage and were excluded from the evidence context."
        )

    fused_raw = fuse_candidates(
        bounded_queries,
        [local_candidates, web_candidates],
        limit=len(local_candidates) + len(web_candidates),
        rrf_k=configurable.hybrid_fusion_rrf_k,
    )
    fused_all = rerank_candidates(
        bounded_queries,
        fused_raw,
        min_relevance=configurable.hybrid_min_candidate_relevance,
        min_source_quality=configurable.hybrid_min_source_quality,
    )
    output_limit = min(max_results, configurable.max_search_results)
    fused = fused_all[:output_limit]
    record_event(
        config,
        "retrieval_candidates_fused",
        "adaptive_hybrid_search",
        data={
            "local_candidates": len(local_candidates),
            "web_candidates": len(web_candidates),
            "fused_candidates": len(fused),
            "unique_candidates": len(fused_all),
            "deduplicated": (
                len(local_candidates) + len(web_candidates) - len(fused_all)
            ),
            "truncated": max(0, len(fused_all) - len(fused)),
            "relevance_filtered": len(fused_raw) - len(fused_all),
        },
    )
    return _render_retrieval_candidates(fused, config, heading="HYBRID SOURCE")


async def _retrieve_tavily_candidates(
    queries: list[str],
    *,
    max_results: int,
    topic: Literal["general", "news", "finance"],
    config: RunnableConfig | None,
) -> list[RetrievalCandidate]:
    """Return Tavily results in the same chunk contract as local retrieval."""
    configurable = Configuration.from_runnable_config(config)
    bounded_queries = queries[: configurable.max_search_queries]
    search_results = await tavily_search_async(
        bounded_queries,
        max_results=min(max_results, configurable.max_search_results),
        topic=topic,
        include_raw_content=True,
        config=config,
    )
    unique_results: dict[str, dict[str, Any]] = {}
    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = {**result, "query": response["query"]}
    if not unique_results:
        return []

    preview_candidates = [
        RetrievalCandidate(
            document_id="web:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:20],
            source="web",
            sources=["web"],
            url=url,
            title=str(result.get("title", "")),
            text=str(result.get("content", "")),
            score=float(result.get("score", 0.0) or 0.0),
            rank=rank,
            query=str(result.get("query", "")),
        )
        for rank, (url, result) in enumerate(unique_results.items(), 1)
    ]
    preview_ranked = rerank_candidates(
        bounded_queries,
        preview_candidates,
        min_relevance=min(0.2, configurable.hybrid_min_candidate_relevance),
        min_source_quality=configurable.hybrid_min_source_quality,
    )[: min(max_results, configurable.max_search_results)]
    unique_results = {
        candidate.url: unique_results[candidate.url] for candidate in preview_ranked
    }
    if not unique_results:
        return []

    summarization_config, _ = model_config_for_node(
        configurable,
        config,
        node="summarize_webpage",
        default_model=configurable.summarization_model,
        default_max_tokens=configurable.summarization_model_max_tokens,
    )
    summarization_model = init_chat_model(**summarization_config)

    async def noop():
        return None

    summaries = await asyncio.gather(
        *[
            noop()
            if not result.get("raw_content")
            else summarize_webpage(
                summarization_model,
                result["raw_content"][: configurable.max_content_length],
                native_structured_output=configurable.native_structured_output,
                retries=configurable.max_structured_output_retries,
                config=config,
            )
            for result in unique_results.values()
        ]
    )
    candidates = []
    for rank, ((url, result), summary) in enumerate(
        zip(unique_results.items(), summaries), 1
    ):
        content = str(result.get("content", "") if summary is None else summary)
        candidates.append(
            RetrievalCandidate(
                document_id="web:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:20],
                chunk_id="web:" + hashlib.sha256(
                    f"{url}\0{content}".encode("utf-8")
                ).hexdigest()[:20],
                source="web",
                sources=["web"],
                url=url,
                title=str(result.get("title", "")),
                text=content,
                score=float(result.get("score", 0.0) or 0.0),
                rank=rank,
                query=str(result.get("query", "")),
            )
        )
    return candidates


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    candidates = await _retrieve_tavily_candidates(
        queries, max_results=max_results, topic=topic, config=config
    )
    return _render_retrieval_candidates(candidates, config, heading="WEB SOURCE")

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously.
    
    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access
        
    Returns:
        List of search result dictionaries from Tavily API
    """
    budget = budget_from_config(config)
    if budget:
        budget.consume("search_calls", len(search_queries))

    # Initialize the Tavily client with API key from config
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    
    # Execute all search queries in parallel and return results
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(
    model: BaseChatModel,
    webpage_content: str,
    *,
    native_structured_output: bool = True,
    retries: int = 3,
    config: RunnableConfig | None = None,
) -> str:
    """Summarize webpage content using AI model with timeout protection.
    
    Args:
        model: The chat model configured for summarization
        webpage_content: Raw webpage content to be summarized
        
    Returns:
        Formatted summary with key excerpts, or original content if summarization fails
    """
    try:
        # Create prompt with current date context
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # Execute summarization with timeout to prevent hanging
        summary = await asyncio.wait_for(
            ainvoke_structured_with_fallback(
                model,
                Summary,
                [HumanMessage(content=prompt_content)],
                retries=retries,
                native_structured_output=native_structured_output,
                config=config,
                node="summarize_webpage",
            ),
            timeout=60.0  # 60 second timeout for summarization
        )
        
        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # Timeout during summarization - return original content
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # Other errors during summarization - log and return original content
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP Utils
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """Exchange Supabase token for MCP access token using OAuth token exchange.
    
    Args:
        supabase_token: Valid Supabase authentication token
        base_mcp_url: Base URL of the MCP server
        
    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        
    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # Retrieve stored tokens
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token expired, clean up and return None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # Store the tokens
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """Fetch and refresh MCP tokens, obtaining new ones if needed.
    
    Args:
        config: Runtime configuration with authentication details
        
    Returns:
        Valid token dictionary, or None if unable to obtain tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # Extract Supabase token for new token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # Exchange Supabase token for MCP tokens
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.
    
    Args:
        tool: The MCP structured tool to wrap
        
    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc
            
            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                raise original_error
            
            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # For other MCP errors, re-raise the original
            raise original_error
    
    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.
    
    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts
        
    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Handle authentication if required
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: Validate configuration requirements
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # Step 3: Set up MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code
    
    # Step 4: Load tools from MCP server
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # If MCP server connection fails, return empty list
        return []
    
    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # Wrap tool with authentication handling and add to list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.
    
    Args:
        search_api: The search API provider to use (Anthropic, OpenAI, Tavily, or None)
        
    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic's native web search with usage limits
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI's web search preview functionality
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # Configure Tavily search tool with metadata
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.LOCAL:
        local_corpus_search.metadata = {
            **(local_corpus_search.metadata or {}),
            "type": "search",
            "name": "local_corpus_search",
        }
        return [local_corpus_search]

    elif search_api == SearchAPI.HYBRID:
        adaptive_hybrid_search.metadata = {
            **(adaptive_hybrid_search.metadata or {}),
            "type": "search",
            "name": "adaptive_hybrid_search",
        }
        return [adaptive_hybrid_search]
        
    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []
        
    # Default fallback for unknown search API types
    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.
    
    Args:
        response: The response object from Anthropic's API
        
    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # Return True if any web search requests were made
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        return False

def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.
    
    Args:
        response: The response object from OpenAI's API
        
    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token Limit Exceeded Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection
        
    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    
    # Step 1: Determine provider from model name if available
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # Step 2: Check provider-specific token limit patterns
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # Step 3: If provider unknown, check all providers
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an OpenAI exception
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # Check for specific OpenAI error codes
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an Anthropic exception
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is a Google/Gemini exception
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # Check for specific Google API resource exhaustion patterns
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]
    
    # No AI messages found, return original list
    return messages

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        configured_env = config.get("configurable", {}).get("model_api_key_env")
        if configured_env:
            return os.getenv(configured_env)
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
