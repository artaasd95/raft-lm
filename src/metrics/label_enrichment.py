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


def enrich_engine_labels(
    row: Dict[str, Any],
    *,
    num_classes: int,
    alpha: float = 0.95,
    engine_version: str,
) -> Dict[str, Any]:
    """
    Compute engine_labels dict and scalar label bucket from row features or returns.

    Uses VaR/CVaR on loss samples derived from returns (explicit or feature-derived).
    """
    if "returns" in row and row["returns"]:
        returns = np.asarray(row["returns"], dtype=float)
    else:
        returns = _pseudo_returns_from_features(list(row.get("features") or []))

    losses = losses_from_simple_returns(returns)
    var_loss = compute_var(losses, alpha=alpha)
    cvar_loss = compute_cvar(losses, alpha=alpha)
    tail_pressure = float(cvar_loss / max(var_loss, 1e-9))

    if "label" in row:
        label = int(row["label"])
    else:
        bucket_score = cvar_loss + 0.1 * tail_pressure
        label = min(num_classes - 1, int(bucket_score * num_classes) % num_classes)

    engine_labels = {
        "var": var_loss,
        "cvar": cvar_loss,
        "tail_pressure": tail_pressure,
        "alpha": alpha,
        "engine_version": engine_version,
        "metrics": ["var", "cvar", "tail_pressure"],
    }
    return {"label": label, "engine_labels": engine_labels}


def enrich_row_labels(
    row: Dict[str, Any],
    *,
    num_classes: int,
    alpha: float = 0.95,
    engine_version: str,
) -> Dict[str, Any]:
    """Apply enrichment and merge into pipeline row."""
    enriched = enrich_engine_labels(
        row,
        num_classes=num_classes,
        alpha=alpha,
        engine_version=engine_version,
    )
    return {
        **row,
        "label": enriched["label"],
        "engine_labels": enriched["engine_labels"],
        "engine_version": engine_version,
    }
