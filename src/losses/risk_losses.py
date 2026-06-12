"""
Risk-aware loss functions.

Uses :func:`src.metrics.risk_metrics.batch_cvar_from_losses` so training objectives
match evaluation CVaR (see docs/RISK-METHODS-REQUIREMENTS.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from ..metrics.risk_metrics import batch_cvar_from_losses


class CVaRLoss(nn.Module):
    """
    Batch CVaR (Expected Shortfall) on per-example base losses.

    Computes the mean of the worst (1-α) fraction of element-wise losses in the batch.
    """

    def __init__(self, alpha: float = 0.95, base_loss: Optional[nn.Module] = None):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.base_loss = base_loss or nn.MSELoss(reduction="none")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.base_loss(predictions, targets)
        return batch_cvar_from_losses(losses, self.alpha)


class TailAwareLoss(nn.Module):
    """
    Emphasizes tail errors: blend of mean loss and CVaR of losses.

    ``tail_weight`` ∈ [0, 1] mixes toward the CVaR term (larger → more tail focus).
    """

    def __init__(
        self,
        alpha: float = 0.95,
        tail_weight: float = 0.5,
        base_loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 <= tail_weight <= 1.0:
            raise ValueError("tail_weight must be in [0, 1]")
        self.alpha = alpha
        self.tail_weight = tail_weight
        self.base_loss = base_loss or nn.MSELoss(reduction="none")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.base_loss(predictions, targets).reshape(-1)
        mean_term = losses.mean()
        tail_term = batch_cvar_from_losses(losses, self.alpha)
        return (1.0 - self.tail_weight) * mean_term + self.tail_weight * tail_term
