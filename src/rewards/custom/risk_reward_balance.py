"""Composite reward balancing PnL and tail risk."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.domain.trajectory import RewardBatch
from src.rewards.base import BaseReward


class RiskRewardBalanceReward(BaseReward):
    """Weighted balance between return-seeking and CVaR-style tail penalty."""

    name = "risk_reward_balance"

    def __init__(
        self,
        pnl_weight: float = 0.6,
        risk_weight: float = 0.4,
        alpha: float = 0.05,
    ) -> None:
        if pnl_weight + risk_weight <= 0:
            raise ValueError("pnl_weight + risk_weight must be positive")
        self.pnl_weight = pnl_weight
        self.risk_weight = risk_weight
        self.alpha = alpha

    def compute(self, batch: Mapping[str, Any]) -> RewardBatch:
        returns = batch.get("returns")
        if returns is None:
            n = len(batch.get("completions") or batch.get("responses") or [0])
            returns = np.zeros(n, dtype=np.float64)
        arr = np.asarray(returns, dtype=np.float64)
        var_idx = max(1, int(len(arr) * self.alpha))
        sorted_returns = np.sort(arr)
        cvar = float(sorted_returns[:var_idx].mean()) if sorted_returns.size else 0.0
        pnl_term = arr.astype(np.float32)
        risk_penalty = np.full_like(pnl_term, abs(min(0.0, cvar)), dtype=np.float32)
        total = self.pnl_weight * pnl_term - self.risk_weight * risk_penalty
        return RewardBatch(
            values=total,
            components={"pnl": pnl_term, "risk_penalty": risk_penalty},
        )
