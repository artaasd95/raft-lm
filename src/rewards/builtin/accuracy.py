"""Task accuracy reward."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.domain.trajectory import RewardBatch
from src.rewards.base import BaseReward


class TaskAccuracyReward(BaseReward):
    name = "task_accuracy"

    def compute(self, batch: Mapping[str, Any]) -> RewardBatch:
        if "correct" in batch:
            vals = np.asarray(batch["correct"], dtype=np.float32)
        elif "labels" in batch and "predictions" in batch:
            labels = np.asarray(batch["labels"])
            preds = np.asarray(batch["predictions"])
            vals = (labels == preds).astype(np.float32)
        else:
            vals = np.asarray(batch.get("rewards", [0.0]), dtype=np.float32)
        return RewardBatch(values=vals)
