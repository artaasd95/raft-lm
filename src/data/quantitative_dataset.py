"""Dataset for quantitative constraint-aware supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class QuantitativeSample:
    """Single quantitative sample with bound constraints."""

    features: torch.Tensor
    target_value: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor


class QuantitativeDataset(Dataset):
    """Stores scalar/value targets with lower and upper bounds."""

    def __init__(
        self,
        features: torch.Tensor,
        target_values: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> None:
        n = features.size(0)
        if any(t.size(0) != n for t in (target_values, lower_bounds, upper_bounds)):
            raise ValueError("all tensors must share leading dimension")
        self._features = features
        self._targets = target_values
        self._lower = lower_bounds
        self._upper = upper_bounds

    def __len__(self) -> int:
        return int(self._features.size(0))

    def __getitem__(self, idx: int) -> QuantitativeSample:
        return QuantitativeSample(
            features=self._features[idx],
            target_value=self._targets[idx],
            lower_bound=self._lower[idx],
            upper_bound=self._upper[idx],
        )
