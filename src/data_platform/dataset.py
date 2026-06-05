"""Torch dataset backed by EngineLabelRow splits."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.data_platform.cards import EngineLabelRow


class EngineLabelDataset(Dataset):
    """Dataset from processed EngineLabelRow JSONL splits."""

    def __init__(
        self,
        rows: List[EngineLabelRow],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rows = rows
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        features = torch.tensor(row.features, dtype=torch.float32)
        label = torch.tensor(row.label, dtype=torch.long)
        return features, label
