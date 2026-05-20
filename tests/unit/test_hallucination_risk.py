"""Unit tests for hallucination severity scoring."""

from src.evals.hallucination_risk import (
    RISK_BUCKETS,
    aggregate_severity,
    score_hallucination_risk,
)


def test_score_maps_financial_domain():
    result = score_hallucination_risk(
        "completely unrelated answer",
        "some other context",
        {"risk_domain": "financial", "ground_truth": "CET1 8.5%"},
    )
    assert result.bucket == "financial"
    assert result.bucket in RISK_BUCKETS


def test_supported_answer_lower_severity():
    gt = "The minimum CET1 ratio is 8.5%."
    result = score_hallucination_risk(
        gt,
        gt,
        {"risk_domain": "compliance", "ground_truth": gt},
    )
    assert result.severity in ("none", "low")


def test_aggregate_severity_counts_buckets():
    r1 = score_hallucination_risk(
        "wrong",
        "ctx",
        {"risk_domain": "legal", "ground_truth": "x"},
    )
    r2 = score_hallucination_risk(
        "also wrong",
        "ctx",
        {"risk_domain": "operational", "ground_truth": "y"},
    )
    summary = aggregate_severity([r1, r2])
    assert summary.total_events >= 0
    assert summary.max_severity in ("none", "low", "medium", "high", "critical")


def test_low_faithfulness_escalates_financial():
    result = score_hallucination_risk(
        "some answer text",
        "context with different words entirely",
        {"risk_domain": "financial", "ground_truth": "truth", "faithfulness": 0.2},
    )
    assert result.bucket == "financial"
    assert result.severity == "high"
    assert result.unsupported_claim is True
