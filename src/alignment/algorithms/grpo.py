"""Group Relative Policy Optimization advantages."""

from __future__ import annotations

import numpy as np


def compute_grpo_advantages(rewards: np.ndarray, group_size: int) -> np.ndarray:
    """
    Group-relative advantages: subtract group mean within each group.

    Args:
        rewards: flat array of completion rewards
        group_size: number of completions per prompt group
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    n = rewards.size
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if n % group_size != 0:
        pad = group_size - (n % group_size)
        rewards = np.pad(rewards, (0, pad), mode="constant")
    groups = rewards.reshape(-1, group_size)
    group_mean = groups.mean(axis=1, keepdims=True)
    adv = groups - group_mean
    return adv.reshape(-1)[:n].astype(np.float32)
