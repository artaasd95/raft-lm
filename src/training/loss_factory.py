"""Shared loss builder for training backends."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from src.losses.base_losses import CrossEntropyLoss, MSELoss

LOSS_ALIASES = {
    "ce": "CrossEntropyLoss",
    "cvar_penalized": "CVaRLoss",
    "tail_aware": "TailAwareLoss",
    "crossentropyloss": "CrossEntropyLoss",
    "mseloss": "MSELoss",
    "cvarloss": "CVaRLoss",
    "tailawareloss": "TailAwareLoss",
}


def normalize_loss_type(loss_spec: Dict[str, Any]) -> str:
    """Resolve policy alias or class name to canonical loss type."""
    raw = loss_spec.get("type") or loss_spec.get("loss") or "CrossEntropyLoss"
    key = str(raw).strip()
    resolved = LOSS_ALIASES.get(key.lower(), key)
    return resolved


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build a loss module from experiment config."""
    loss_spec = config["training"]["loss"]
    loss_type = normalize_loss_type(loss_spec)
    alpha = float(loss_spec.get("alpha", 0.95))
    tail_weight = float(loss_spec.get("tail_weight", 0.5))

    if loss_type == "CrossEntropyLoss":
        return CrossEntropyLoss()
    if loss_type == "MSELoss":
        return MSELoss()
    if loss_type == "CVaRLoss":
        from src.losses.risk_losses import CVaRLoss

        base = nn.CrossEntropyLoss(reduction="none")
        return CVaRLoss(alpha=alpha, base_loss=base)
    if loss_type == "TailAwareLoss":
        from src.losses.risk_losses import TailAwareLoss

        base = nn.CrossEntropyLoss(reduction="none")
        return TailAwareLoss(alpha=alpha, tail_weight=tail_weight, base_loss=base)
    raise ValueError(f"Unsupported loss type: {loss_type}")
