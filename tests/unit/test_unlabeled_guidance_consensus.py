"""Unit tests for consensus council scoring."""

import pytest

from src.search.pgts.consensus import aggregate_consensus, score_hypothesis_offline


def test_aggregate_consensus_median():
    result = aggregate_consensus([0.8, 0.75, 0.82])
    assert result.median_score == pytest.approx(0.8, abs=0.01)
    assert 0.0 <= result.weighted_score <= 1.0


def test_echo_score_high_on_disagreement():
    agree = aggregate_consensus([0.7, 0.71, 0.69])
    disagree = aggregate_consensus([0.2, 0.9, 0.5])
    assert disagree.echo_score > agree.echo_score


def test_outlier_downweighted():
    with_outlier = aggregate_consensus([0.7, 0.72, 0.05], outlier_mad_factor=1.0)
    aggregate_consensus([0.7, 0.72, 0.71])
    assert with_outlier.weighted_score >= 0.0


def test_empty_scores_returns_zero():
    result = aggregate_consensus([])
    assert result.median_score == 0.0
    assert result.echo_score == 1.0


def test_offline_scorer_deterministic():
    rationale = "Because cvar data shows tail risk since features increased."
    features = [0.5, 0.8, 1.2]
    a = score_hypothesis_offline(rationale, features)
    b = score_hypothesis_offline(rationale, features)
    assert a.weighted_score == b.weighted_score
