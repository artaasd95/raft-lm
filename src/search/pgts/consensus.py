"""Multi-evaluator consensus scoring for label-free verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]

DEFAULT_EVALUATOR_ROLES = ("clarity", "objectivity", "evidence")


@dataclass(frozen=True)
class ConsensusResult:
    """Aggregated council scores for a hypothesis."""

    median_score: float
    echo_score: float
    weighted_score: float
    evaluator_scores: List[float]


def _median_abs_deviation(values: ArrayLike) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _downweight_outliers(
    scores: ArrayLike,
    *,
    outlier_mad_factor: float = 2.5,
) -> List[float]:
    """Downweight scores that deviate strongly from the council median."""
    if len(scores) == 0:
        return []
    arr = np.asarray(scores, dtype=float)
    med = float(np.median(arr))
    mad = _median_abs_deviation(arr)
    threshold = outlier_mad_factor * max(mad, 1e-6)
    weights: List[float] = []
    for score in arr:
        deviation = abs(float(score) - med)
        weight = 0.25 if deviation > threshold else 1.0
        weights.append(weight * float(score))
    return weights


def _heuristic_axis_score(
    rationale: str,
    features: Sequence[float],
    axis: str,
) -> float:
    """Deterministic offline scorer for rubric axes."""
    text = rationale.lower()
    tokens = re.findall(r"\w+", text)
    token_count = len(tokens)

    if axis == "clarity":
        base = min(1.0, token_count / 12.0) if token_count else 0.2
        penalty = 0.1 if "???" in text or "unclear" in text else 0.0
        return max(0.0, min(1.0, base - penalty))

    if axis == "objectivity":
        subjective = sum(1 for w in ("maybe", "perhaps", "guess", "feel") if w in text)
        base = 0.75 - 0.15 * subjective
        if features:
            spread = float(np.std(np.asarray(features, dtype=float)))
            base += min(0.15, spread * 0.1)
        return max(0.0, min(1.0, base))

    if axis == "evidence":
        evidence_markers = ("because", "since", "data", "metric", "cvar", "var", "feature")
        hits = sum(1 for m in evidence_markers if m in text)
        feature_signal = min(0.4, abs(sum(features)) * 0.05) if features else 0.0
        return max(0.0, min(1.0, 0.3 + 0.1 * hits + feature_signal))

    return 0.5


def score_hypothesis_offline(
    rationale: str,
    features: Sequence[float],
    *,
    evaluator_roles: Sequence[str] = DEFAULT_EVALUATOR_ROLES,
    outlier_mad_factor: float = 2.5,
) -> ConsensusResult:
    """Score a hypothesis using deterministic evaluator heuristics."""
    raw_scores = [
        _heuristic_axis_score(rationale, features, role) for role in evaluator_roles
    ]
    return aggregate_consensus(
        raw_scores,
        outlier_mad_factor=outlier_mad_factor,
    )


def aggregate_consensus(
    evaluator_scores: Sequence[float],
    *,
    outlier_mad_factor: float = 2.5,
) -> ConsensusResult:
    """Aggregate evaluator scores via median and outlier downweighting."""
    if not evaluator_scores:
        return ConsensusResult(
            median_score=0.0,
            echo_score=1.0,
            weighted_score=0.0,
            evaluator_scores=[],
        )

    arr = np.asarray(evaluator_scores, dtype=float)
    median = float(np.median(arr))
    echo = float(np.var(arr))
    weighted = _downweight_outliers(arr, outlier_mad_factor=outlier_mad_factor)
    weighted_score = float(np.mean(weighted)) if weighted else median

    return ConsensusResult(
        median_score=median,
        echo_score=echo,
        weighted_score=weighted_score,
        evaluator_scores=[float(s) for s in arr],
    )
