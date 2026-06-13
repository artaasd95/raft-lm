"""Losses for tool-selection aware training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToolSelectionLoss(nn.Module):
    """Cross-entropy loss constrained by tool availability masks."""

    def __init__(self, invalid_tool_penalty: float = 0.2) -> None:
        super().__init__()
        self.invalid_tool_penalty = invalid_tool_penalty

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tool_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Push unavailable tools to a large negative value before CE.
        masked_logits = logits.masked_fill(tool_mask <= 0, -1e9)
        ce = F.cross_entropy(masked_logits, labels)

        probs = torch.softmax(logits, dim=-1)
        invalid_prob_mass = (probs * (1 - tool_mask.float())).sum(dim=-1).mean()
        return ce + self.invalid_tool_penalty * invalid_prob_mass
