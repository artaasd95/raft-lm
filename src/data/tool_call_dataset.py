"""Dataset for tool-aware action selection supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToolAwareSample:
    """Single tool-aware sample for classification over tool choices."""

    features: torch.Tensor
    label: torch.Tensor
    tool_mask: torch.Tensor


class ToolAwareDataset(Dataset):
    """Stores model inputs, labels, and tool availability masks."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        tool_masks: torch.Tensor,
    ) -> None:
        n = features.size(0)
        if labels.size(0) != n or tool_masks.size(0) != n:
            raise ValueError("features, labels, and tool_masks must have same length")
        self._features = features
        self._labels = labels
        self._tool_masks = tool_masks

    def __len__(self) -> int:
        return int(self._features.size(0))

    def __getitem__(self, idx: int) -> ToolAwareSample:
        return ToolAwareSample(
            features=self._features[idx],
            label=self._labels[idx],
            tool_mask=self._tool_masks[idx],
        )
