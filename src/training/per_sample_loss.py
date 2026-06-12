"""Per-sample loss helpers for evaluation and callbacks."""

from __future__ import annotations

import torch


def per_sample_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute one loss value per sample using reduction='none'.

    Classification uses cross-entropy; regression uses mean MSE over non-batch dims.
    """
    if targets.dtype in (torch.long, torch.int32, torch.int64) and outputs.ndim >= 2:
        return torch.nn.functional.cross_entropy(outputs, targets, reduction="none")
    losses = torch.nn.functional.mse_loss(outputs, targets, reduction="none")
    if losses.ndim == 1:
        return losses
    return losses.reshape(losses.shape[0], -1).mean(dim=1)
