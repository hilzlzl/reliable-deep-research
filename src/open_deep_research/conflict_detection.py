"""Deterministic candidate detection for contradictory evidence."""

from __future__ import annotations

import hashlib
import re
from itertools import combinations

from open_deep_research.evidence import EvidenceConflict
from open_deep_research.runtime_store import RuntimeStore

NEGATIONS = {
    "no", "not", "never", "neither", "without", "cannot", "can't", "doesn't",
    "isn't", "wasn't", "不", "未", "没有", "不能", "并非", "无",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+|[\u4e00-\u9fff]", text.lower()))


def _polarity_and_signature(text: str) -> tuple[bool, set[str]]:
    tokens = _tokens(text)
    return bool(tokens.intersection(NEGATIONS)), tokens - NEGATIONS


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def detect_and_persist_conflicts(
    store: RuntimeStore,
    *,
    similarity_threshold: float = 0.5,
) -> list[EvidenceConflict]:
    """Find high-similarity evidence pairs with opposite negation polarity."""
    evidence_by_id = {item["evidence_id"]: item for item in store.list_evidence()}
    conflicts = []
    for claim in store.list_claims():
        linked = [
            evidence_by_id[evidence_id]
            for evidence_id in claim["evidence_ids"]
            if evidence_id in evidence_by_id
        ]
        for left, right in combinations(linked, 2):
            left_negative, left_signature = _polarity_and_signature(left["excerpt"])
            right_negative, right_signature = _polarity_and_signature(right["excerpt"])
            similarity = _jaccard(left_signature, right_signature)
            if left_negative == right_negative or similarity < similarity_threshold:
                continue
            ordered_ids = sorted((left["evidence_id"], right["evidence_id"]))
            conflict_id = "conflict:" + hashlib.sha256(
                f"{claim['claim_id']}\0{ordered_ids[0]}\0{ordered_ids[1]}".encode("utf-8")
            ).hexdigest()[:20]
            conflict = EvidenceConflict(
                conflict_id=conflict_id,
                claim_id=claim["claim_id"],
                evidence_id_a=ordered_ids[0],
                evidence_id_b=ordered_ids[1],
                reason=(
                    "Candidate contradiction: similar wording with opposite "
                    f"negation polarity (similarity={similarity:.2f})"
                ),
            )
            store.save_conflict(conflict)
            conflicts.append(conflict)
    return conflicts
