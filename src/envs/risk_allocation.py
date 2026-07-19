"""Risk allocation Gymnasium environment."""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RiskAllocationEnv(gym.Env):
    """
    Simple portfolio allocation MDP.

    State: [cash_weight, asset_return_mean, asset_vol]
    Action: discrete allocation bucket (0=conservative .. n-1=aggressive)
    Reward: simulated return minus CVaR-style penalty
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_dim: int = 3,
        n_actions: int = 5,
        episode_len: int = 32,
        seed: Optional[int] = None,
        cvar_penalty: float = 0.3,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.episode_len = episode_len
        self.cvar_penalty = cvar_penalty
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._state = np.zeros(obs_dim, dtype=np.float32)
        self._returns_history: list[float] = []

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._returns_history = []
        self._state = self._rng.normal(0, 0.1, size=self.obs_dim).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        risk_frac = (action + 1) / self.n_actions
        shock = float(self._rng.normal(0.001, 0.02))
        ret = risk_frac * shock - (1.0 - risk_frac) * 0.001
        self._returns_history.append(ret)
        tail = np.sort(self._returns_history)[: max(1, len(self._returns_history) // 5)]
        cvar_pen = self.cvar_penalty * float(-np.mean(tail)) if len(tail) else 0.0
        reward = ret - cvar_pen
        self._state = self._rng.normal(ret, 0.05, size=self.obs_dim).astype(np.float32)
        self._step += 1
        terminated = self._step >= self.episode_len
        truncated = False
        return self._state.copy(), reward, terminated, truncated, {"return": ret}
