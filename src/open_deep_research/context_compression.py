"""Budgeted, measurable compression for persisted research evidence."""

from __future__ import annotations

import re

from pydantic import BaseModel


class CompressionMetrics(BaseModel):
    input_characters: int
    output_characters: int
    compression_ratio: float
    evidence_retention: float
    required_evidence_retention: float


class CompressionResult(BaseModel):
    context: str
    selected_evidence_ids: list[str]
    metrics: CompressionMetrics


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()))


def compress_evidence(
    evidence: list[dict],
    *,
    query: str,
    max_characters: int,
    required_evidence_ids: set[str] | None = None,
) -> CompressionResult:
    """Select evidence under a hard budget and expose preservation metrics."""
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    required = required_evidence_ids or set()
    known_ids = {item["evidence_id"] for item in evidence}
    unknown_required = required - known_ids
    if unknown_required:
        raise ValueError(f"Unknown required evidence ids: {', '.join(sorted(unknown_required))}")

    query_terms = _terms(query)

    def score(item: dict) -> tuple[float, float, str]:
        content_terms = _terms(f"{item.get('claim_text', '')} {item.get('excerpt', '')}")
        relevance = len(query_terms & content_terms) / max(len(query_terms), 1)
        required_bonus = 10.0 if item["evidence_id"] in required else 0.0
        return required_bonus + relevance, float(item.get("confidence", 0.5)), item["evidence_id"]

    ranked = sorted(evidence, key=score, reverse=True)
    selected = []
    blocks = []
    used = 0
    for item in ranked:
        block = (
            f"[{item['evidence_id']}] {item.get('claim_text', '')}\n"
            f"Excerpt: {item.get('excerpt', '')}\n"
            f"Source: {item.get('source_title', '')} {item.get('source_url', '')}"
        ).strip()
        separator = 2 if blocks else 0
        remaining = max_characters - used - separator
        if remaining <= 0:
            break
        if len(block) > remaining:
            if item["evidence_id"] not in required:
                continue
            block = block[:remaining]
        blocks.append(block)
        selected.append(item["evidence_id"])
        used += len(block) + separator

    context = "\n\n".join(blocks)
    input_characters = sum(
        len(str(item.get("claim_text", ""))) + len(str(item.get("excerpt", "")))
        + len(str(item.get("source_url", ""))) + len(str(item.get("source_title", "")))
        for item in evidence
    )
    selected_set = set(selected)
    required_retention = (
        len(required & selected_set) / len(required) if required else 1.0
    )
    return CompressionResult(
        context=context,
        selected_evidence_ids=selected,
        metrics=CompressionMetrics(
            input_characters=input_characters,
            output_characters=len(context),
            compression_ratio=len(context) / input_characters if input_characters else 0.0,
            evidence_retention=len(selected) / len(evidence) if evidence else 1.0,
            required_evidence_retention=required_retention,
        ),
    )
