"""Peer consistency verification for label-free guidance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class ConsistencyResult:
    """Result of peer consistency check."""

    consistency_score: float
    label_match: bool
    token_overlap: float
    discriminator_label: int


def mask_features(features: Sequence[float], *, mask_ratio: float = 0.5) -> List[float]:
    """Mask trailing portion of feature vector for discriminator hint."""
    if not features:
        return []
    cutoff = max(1, int(len(features) * mask_ratio))
    masked = list(features[:cutoff])
    masked.extend([0.0] * (len(features) - cutoff))
    return masked


def mask_trace(trace: str, *, mask_ratio: float = 0.5) -> str:
    """Mask trailing portion of a reasoning trace."""
    if not trace:
        return ""
    words = trace.split()
    if not words:
        return ""
    cutoff = max(1, int(len(words) * mask_ratio))
    return " ".join(words[:cutoff])


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.lower()))
    b_tokens = set(re.findall(r"\w+", b.lower()))
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _discriminator_label_from_hint(
    features: Sequence[float],
    num_classes: int,
) -> int:
    """Deterministic offline discriminator label from partial features."""
    if not features or num_classes <= 0:
        return 0
    arr = np.asarray(features, dtype=float)
    score = float(np.mean(arr))
    bucket = int((score + 1.0) * num_classes / 2.0)
    return max(0, min(num_classes - 1, bucket))


def score_consistency_offline(
    *,
    generator_rationale: str,
    generator_label: int,
    features: Sequence[float],
    num_classes: int,
    trace: str = "",
    mask_ratio: float = 0.5,
) -> ConsistencyResult:
    """
    Score peer consistency by comparing generator output to discriminator completion.

    Offline mode uses masked features/trace heuristics instead of an LLM.
    """
    masked_features = mask_features(features, mask_ratio=mask_ratio)
    discriminator_label = _discriminator_label_from_hint(masked_features, num_classes)

    masked_trace = mask_trace(trace or generator_rationale, mask_ratio=mask_ratio)
    discriminator_rationale = (
        f"{masked_trace} label_bucket={discriminator_label} "
        f"mean_feature={float(np.mean(masked_features)) if masked_features else 0.0:.3f}"
    )

    overlap = _token_overlap(generator_rationale, discriminator_rationale)
    label_match = discriminator_label == generator_label
    label_bonus = 0.35 if label_match else 0.0
    consistency = max(0.0, min(1.0, 0.5 * overlap + label_bonus + 0.15))

    return ConsistencyResult(
        consistency_score=consistency,
        label_match=label_match,
        token_overlap=overlap,
        discriminator_label=discriminator_label,
    )
