"""Engine label enrichment using risk metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.metrics.risk_metrics import compute_cvar, compute_var, losses_from_simple_returns


def _pseudo_returns_from_features(features: List[float]) -> np.ndarray:
    """Derive a short return series from feature vector (deterministic)."""
    if not features:
        return np.array([0.0])
    arr = np.asarray(features, dtype=float)
    if arr.size == 1:
        return arr
    diffs = np.diff(arr)
    scale = max(float(np.std(arr)), 1e-6)
    return diffs / scale


def has_explicit_label(row: Dict[str, Any]) -> bool:
    """Return True when the row carries an explicit supervised label."""
    return "label" in row and row.get("label") is not None


def compute_engine_label_metrics(
    row: Dict[str, Any],
    *,
    alpha: float = 0.95,
    engine_version: str,
) -> Dict[str, Any]:
    """Compute VaR/CVaR engine_labels without assigning a label bucket."""
    if "returns" in row and row["returns"]:
        returns = np.asarray(row["returns"], dtype=float)
    else:
        returns = _pseudo_returns_from_features(list(row.get("features") or []))

    losses = losses_from_simple_returns(returns)
    var_loss = compute_var(losses, alpha=alpha)
    cvar_loss = compute_cvar(losses, alpha=alpha)
    tail_pressure = float(cvar_loss / max(var_loss, 1e-9))

    return {
        "var": var_loss,
        "cvar": cvar_loss,
        "tail_pressure": tail_pressure,
        "alpha": alpha,
        "engine_version": engine_version,
        "metrics": ["var", "cvar", "tail_pressure"],
    }


def synthesize_label_from_metrics(
    engine_labels: Dict[str, Any],
    *,
    num_classes: int,
) -> int:
    """Derive a label bucket from engine risk metrics (legacy engine policy)."""
    cvar_loss = float(engine_labels["cvar"])
    tail_pressure = float(engine_labels["tail_pressure"])
    bucket_score = cvar_loss + 0.1 * tail_pressure
    return min(num_classes - 1, int(bucket_score * num_classes) % num_classes)


def enrich_engine_labels(
    row: Dict[str, Any],
    *,
    num_classes: int,
    alpha: float = 0.95,
    engine_version: str,
    synthesize_missing: bool = False,
) -> Dict[str, Any]:
    """
    Compute engine_labels dict and optional scalar label bucket.

    Uses VaR/CVaR on loss samples derived from returns (explicit or feature-derived).
    When ``synthesize_missing`` is False and no explicit label exists, the result
    omits ``label`` so callers can raise or run unlabeled guidance.
    """
    engine_labels = compute_engine_label_metrics(
        row,
        alpha=alpha,
        engine_version=engine_version,
    )

    result: Dict[str, Any] = {"engine_labels": engine_labels}
    if has_explicit_label(row):
        result["label"] = int(row["label"])
    elif synthesize_missing:
        result["label"] = synthesize_label_from_metrics(
            engine_labels,
            num_classes=num_classes,
        )
    return result


def enrich_row_labels(
    row: Dict[str, Any],
    *,
    num_classes: int,
    alpha: float = 0.95,
    engine_version: str,
    synthesize_missing: bool = False,
) -> Dict[str, Any]:
    """Apply enrichment and merge into pipeline row."""
    enriched = enrich_engine_labels(
        row,
        num_classes=num_classes,
        alpha=alpha,
        engine_version=engine_version,
        synthesize_missing=synthesize_missing,
    )
    merged = {
        **row,
        "engine_labels": enriched["engine_labels"],
        "engine_version": engine_version,
    }
    if "label" in enriched:
        merged["label"] = enriched["label"]
    return merged
