"""Hallucination severity scoring for enterprise risk buckets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.evals.benchmark_schema import SeveritySummary


RISK_BUCKETS = ("legal", "financial", "compliance", "operational")
SEVERITY_LEVELS = ("none", "low", "medium", "high", "critical")


@dataclass
class SeverityResult:
    bucket: str
    severity: str
    reasons: List[str]
    unsupported_claim: bool


def _normalize_bucket(risk_domain: Optional[str]) -> str:
    domain = (risk_domain or "operational").lower()
    if domain in RISK_BUCKETS:
        return domain
    if domain in ("aml", "regulatory"):
        return "compliance"
    return "operational"


def _token_overlap(answer: str, context: str) -> float:
    a_tokens = set(re.findall(r"\w+", answer.lower()))
    c_tokens = set(re.findall(r"\w+", context.lower()))
    if not a_tokens:
        return 0.0
    return len(a_tokens & c_tokens) / len(a_tokens)


def score_hallucination_risk(
    answer: str,
    context: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> SeverityResult:
    """
    Score unsupported or risky answers against enterprise buckets.

    metadata may include: risk_domain, ground_truth
    """
    metadata = metadata or {}
    bucket = _normalize_bucket(metadata.get("risk_domain"))
    ground_truth = metadata.get("ground_truth", "")
    overlap = _token_overlap(answer, context)
    gt_overlap = _token_overlap(answer, ground_truth) if ground_truth else overlap

    reasons: List[str] = []
    unsupported = overlap < 0.2 and gt_overlap < 0.25
    if unsupported:
        reasons.append("low overlap with retrieved context and ground truth")

    severity = "none"
    if unsupported:
        if bucket in ("legal", "financial"):
            severity = "high"
        elif bucket == "compliance":
            severity = "medium"
        else:
            severity = "low"
    elif overlap < 0.4:
        severity = "low"
        reasons.append("partial support from context")

    return SeverityResult(
        bucket=bucket,
        severity=severity,
        reasons=reasons,
        unsupported_claim=unsupported,
    )


def aggregate_severity(results: List[SeverityResult]) -> SeveritySummary:
    summary = SeveritySummary()
    order = {level: idx for idx, level in enumerate(SEVERITY_LEVELS)}
    max_level = "none"

    for res in results:
        if res.unsupported_claim or res.severity != "none":
            summary.total_events += 1
            setattr(summary, res.bucket, getattr(summary, res.bucket) + 1)
            if order.get(res.severity, 0) > order.get(max_level, 0):
                max_level = res.severity

    summary.max_severity = max_level
    return summary
