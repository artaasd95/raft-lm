"""Dataset for probabilistic risk targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ProbabilisticSample:
    """Single probabilistic supervision sample."""

    features: torch.Tensor
    target_distribution: torch.Tensor


class ProbabilisticDataset(Dataset):
    """Stores features with probability-distribution labels."""

    def __init__(self, features: torch.Tensor, target_distributions: torch.Tensor) -> None:
        if features.size(0) != target_distributions.size(0):
            raise ValueError("features and target_distributions must have same length")
        if target_distributions.ndim != 2:
            raise ValueError("target_distributions must be rank-2 [N, C]")
        self._features = features
        self._targets = target_distributions

    def __len__(self) -> int:
        return int(self._features.size(0))

    def __getitem__(self, idx: int) -> ProbabilisticSample:
        return ProbabilisticSample(
            features=self._features[idx],
            target_distribution=self._targets[idx],
        )
