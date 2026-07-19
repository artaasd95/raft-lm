"""Risk-aware CVaR penalty reward."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.domain.trajectory import RewardBatch
from src.metrics.risk_metrics import compute_cvar
from src.rewards.base import BaseReward


class RiskCVaRReward(BaseReward):
    name = "risk_cvar"

    def __init__(self, alpha: float = 0.05, scale: float = 1.0) -> None:
        self.alpha = alpha
        self.scale = scale

    def compute(self, batch: Mapping[str, Any]) -> RewardBatch:
        losses = batch.get("losses") or batch.get("returns")
        if losses is None:
            return RewardBatch(values=np.zeros(1, dtype=np.float32))
        arr = np.asarray(losses, dtype=np.float64)
        if arr.ndim == 1 and arr.size > 1:
            cvar = compute_cvar(-arr, alpha=1.0 - self.alpha)
            penalty = -self.scale * float(cvar)
            return RewardBatch(values=np.full(arr.shape, penalty, dtype=np.float32))
        return RewardBatch(values=-self.scale * np.abs(arr).astype(np.float32))
