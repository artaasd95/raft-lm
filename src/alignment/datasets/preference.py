"""Preference dataset from PreferencePair JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from src.data_platform.cards import PreferencePair


class PreferenceDataset(Dataset):
    """Load chosen/rejected pairs for DPO/KTO."""

    def __init__(self, path: str, max_samples: Optional[int] = None) -> None:
        self.samples: List[PreferencePair] = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self.samples.append(PreferencePair.from_dict(data))
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.samples[idx]
        return {
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
            "pair_id": pair.pair_id,
        }

    @classmethod
    def from_pairs(cls, pairs: List[PreferencePair]) -> "PreferenceDataset":
        ds = cls.__new__(cls)
        ds.samples = list(pairs)
        return ds
