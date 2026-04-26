"""
Task performance metrics.

Standard metrics for classification, regression, and decision tasks.
Works with NumPy arrays; PyTorch tensors are converted when available.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and callable(x.detach):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x)


def _classification_pair(predictions: Any, targets: Any) -> Tuple[np.ndarray, np.ndarray]:
    p = _to_numpy(predictions)
    t = _to_numpy(targets)
    if p.ndim > 1:
        p = p.argmax(axis=-1)
    return p.reshape(-1), t.reshape(-1)


def accuracy(predictions: Any, targets: Any) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth class indices
    """
    p, t = _classification_pair(predictions, targets)
    return float((p == t).mean())


def mse(predictions: Any, targets: Any) -> float:
    """Mean Squared Error."""
    p = _to_numpy(predictions).reshape(-1)
    t = _to_numpy(targets).reshape(-1)
    return float(((p - t) ** 2).mean())


def mae(predictions: Any, targets: Any) -> float:
    """Mean Absolute Error."""
    p = _to_numpy(predictions).reshape(-1)
    t = _to_numpy(targets).reshape(-1)
    return float(np.abs(p - t).mean())


def f1_score(predictions: Any, targets: Any) -> float:
    """F1 score for binary classification."""
    p, t = _classification_pair(predictions, targets)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return float(2 * (precision * recall) / (precision + recall))
