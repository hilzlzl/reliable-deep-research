"""Typed evidence and claim records for traceable research reports."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    CONFLICTED = "CONFLICTED"
    REJECTED = "REJECTED"


class Document(BaseModel):
    document_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = ""
    content_hash: str = Field(min_length=1)
    retrieved_at: str


class Evidence(BaseModel):
    evidence_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ResearchClaim(BaseModel):
    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNVERIFIED


class EvidenceConflict(BaseModel):
    conflict_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    evidence_id_a: str = Field(min_length=1)
    evidence_id_b: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExtractedClaim(BaseModel):
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class ClaimExtraction(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


class ReportSourceAudit(BaseModel):
    report_sources: int
    verified_sources: int
    source_precision: float | None
    unsupported_sources: list[str] = Field(default_factory=list)


def audit_report_sources(report: str, allowed_urls: set[str]) -> ReportSourceAudit:
    """Check whether report URLs occur in the persisted evidence set."""
    report_urls = {
        value.rstrip(".,;:")
        for value in re.findall(r"(?:https?|local)://[^\s)\]]+", report)
    }
    normalized_allowed = {value.rstrip(".,;:") for value in allowed_urls}
    verified = report_urls & normalized_allowed
    unsupported = sorted(report_urls - normalized_allowed)
    return ReportSourceAudit(
        report_sources=len(report_urls),
        verified_sources=len(verified),
        source_precision=(len(verified) / len(report_urls) if report_urls else None),
        unsupported_sources=unsupported,
    )
