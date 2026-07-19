"""Group-in-Group Policy Optimization (GiGPO) advantage estimation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np


def compute_episode_advantages(episode_returns: np.ndarray, group_size: int) -> np.ndarray:
    """Macro relative advantages: reward minus group mean per episode group."""
    returns = np.asarray(episode_returns, dtype=np.float64)
    n = returns.size
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if n % group_size != 0:
        pad = group_size - (n % group_size)
        returns = np.pad(returns, (0, pad), mode="constant")
    groups = returns.reshape(-1, group_size)
    group_mean = groups.mean(axis=1, keepdims=True)
    adv = groups - group_mean
    return adv.reshape(-1)[:n].astype(np.float32)


def compute_step_advantages(
    state_keys: Sequence[Hashable],
    step_returns: np.ndarray,
    gamma: float = 0.95,
) -> np.ndarray:
    """
    Micro relative advantages via anchor-state grouping.

    Actions reaching the same state key are grouped; each step return is
    compared to the mean return of its anchor group.
    """
    step_returns = np.asarray(step_returns, dtype=np.float64)
    buckets: Dict[Hashable, List[int]] = defaultdict(list)
    for idx, key in enumerate(state_keys):
        buckets[key].append(idx)
    advantages = np.zeros_like(step_returns, dtype=np.float32)
    for indices in buckets.values():
        group_vals = step_returns[indices]
        mean_val = float(group_vals.mean()) if group_vals.size else 0.0
        for i in indices:
            advantages[i] = float(step_returns[i] - mean_val)
    return advantages


def compute_gigpo_advantages(
    episode_returns: np.ndarray,
    state_keys: Sequence[Hashable],
    step_rewards: np.ndarray,
    group_size: int,
    step_reward_gamma: float = 0.95,
    episode_reward_weight: float = 1.0,
    step_reward_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine episode-level and step-level GiGPO advantages.

    Returns (combined_advantages, episode_adv, step_adv).
    """
    episode_adv = compute_episode_advantages(episode_returns, group_size)
    discounted = np.asarray(step_rewards, dtype=np.float64)
    step_adv = compute_step_advantages(state_keys, discounted, gamma=step_reward_gamma)
    combined = (
        episode_reward_weight * episode_adv + step_reward_weight * step_adv
    ).astype(np.float32)
    return combined, episode_adv, step_adv
