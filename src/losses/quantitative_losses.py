"""Loss functions for quantitative reasoning tasks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintViolationLoss(nn.Module):
    """Regression loss augmented with explicit bound-violation penalties."""

    def __init__(self, violation_weight: float = 1.0) -> None:
        super().__init__()
        self.violation_weight = violation_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target_value: torch.Tensor,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ) -> torch.Tensor:
        base = F.mse_loss(prediction, target_value)
        low_violation = torch.relu(lower_bound - prediction)
        high_violation = torch.relu(prediction - upper_bound)
        violation = (low_violation + high_violation).mean()
        return base + self.violation_weight * violation
