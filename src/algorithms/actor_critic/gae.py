"""GAE advantage estimation."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        mask = 1.0 - float(dones[t])
        nv = next_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
