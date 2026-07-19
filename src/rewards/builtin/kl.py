"""KL penalty reward (policy vs reference log-probs)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.domain.trajectory import RewardBatch
from src.rewards.base import BaseReward


class KLPenaltyReward(BaseReward):
    name = "kl_penalty"

    def compute(self, batch: Mapping[str, Any]) -> RewardBatch:
        policy = batch.get("policy_logprobs")
        ref = batch.get("ref_logprobs")
        if policy is None or ref is None:
            return RewardBatch(values=np.zeros(1, dtype=np.float32))
        p = np.asarray(policy, dtype=np.float64)
        r = np.asarray(ref, dtype=np.float64)
        kl = np.maximum(p - r, 0.0)
        if kl.ndim > 1:
            kl = kl.sum(axis=-1)
        return RewardBatch(values=-kl.astype(np.float32))
