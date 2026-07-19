"""PnL-based reward for finance env."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.domain.trajectory import RewardBatch
from src.rewards.base import BaseReward


class PnLReward(BaseReward):
    name = "pnl"

    def compute(self, batch: Mapping[str, Any]) -> RewardBatch:
        returns = batch.get("returns") or batch.get("rewards") or [0.0]
        return RewardBatch(values=np.asarray(returns, dtype=np.float32))
