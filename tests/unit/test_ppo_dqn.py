"""Unit tests for PPO and DQN updates."""

import numpy as np
import pytest

pytest.importorskip("torch")

from src.rl.algorithms.dqn import DQNAgent
from src.rl.algorithms.ppo import PPOAgent
from src.rl.buffers.replay import ReplayBuffer
from src.rl.buffers.rollout import RolloutBuffer
from src.rl.envs.risk_allocation import RiskAllocationEnv


def test_risk_allocation_env_api():
    env = RiskAllocationEnv(seed=42)
    state, _ = env.reset(seed=42)
    assert state.shape == (3,)
    next_state, reward, term, trunc, info = env.step(0)
    assert next_state.shape == (3,)
    assert isinstance(reward, float)


def test_ppo_update_finite_loss():
    agent = PPOAgent(3, 5, device="cpu")
    buf = RolloutBuffer()
    state = np.zeros(3, dtype=np.float32)
    for _ in range(8):
        a, lp, v = agent.select_action(state)
        buf.add(state, a, 0.1, False, lp, v)
    stats = agent.update(buf)
    assert np.isfinite(stats["loss"])


def test_dqn_target_update():
    agent = DQNAgent(3, 5, device="cpu")
    buf = ReplayBuffer()
    s = np.zeros(3, dtype=np.float32)
    for i in range(50):
        buf.push(s, i % 5, 0.1, s, False)
    stats = agent.update(buf, batch_size=16)
    assert np.isfinite(stats["loss"])
