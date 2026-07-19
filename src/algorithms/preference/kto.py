"""KTO unpaired preference loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kto_loss(
    desirable_logps: torch.Tensor,
    undesirable_logps: torch.Tensor,
    ref_desirable_logps: torch.Tensor,
    ref_undesirable_logps: torch.Tensor,
    beta: float = 0.1,
    lambda_d: float = 1.0,
    lambda_u: float = 1.0,
) -> torch.Tensor:
    """Kahneman-Tversky Optimization style loss (simplified)."""
    d_logits = beta * (desirable_logps - ref_desirable_logps)
    u_logits = beta * (undesirable_logps - ref_undesirable_logps)
    loss_d = -lambda_d * F.logsigmoid(d_logits).mean()
    loss_u = -lambda_u * F.logsigmoid(-u_logits).mean()
    return loss_d + loss_u
