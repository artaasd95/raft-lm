"""Loss functions for probabilistic reasoning tasks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationLoss(nn.Module):
    """KL supervision plus confidence calibration penalty."""

    def __init__(self, confidence_weight: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.confidence_weight = confidence_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + self.eps)
        kl_term = F.kl_div(log_probs, target_distribution, reduction="batchmean")

        pred_confidence = probs.max(dim=-1).values
        target_confidence = target_distribution.max(dim=-1).values
        confidence_term = F.mse_loss(pred_confidence, target_confidence)
        return kl_term + self.confidence_weight * confidence_term
