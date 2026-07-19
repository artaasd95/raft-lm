"""Unit tests for RL buffers and GAE."""

import numpy as np
import pytest

pytest.importorskip("torch")

from src.algorithms.actor_critic.gae import compute_gae
from src.buffers.replay import ReplayBuffer
from src.buffers.rollout import RolloutBuffer


def test_rollout_buffer_shapes():
    buf = RolloutBuffer()
    for i in range(3):
        buf.add(np.array([i, 0.0], dtype=np.float32), i, float(i), i == 2)
    arr = buf.as_arrays()
    assert arr["states"].shape == (3, 2)
    assert arr["actions"].shape == (3,)


def test_replay_buffer_sample():
    buf = ReplayBuffer(capacity=100)
    s = np.zeros(2, dtype=np.float32)
    for i in range(40):
        buf.push(s + i, i % 3, float(i), s + i + 1, False)
    batch = buf.sample(16)
    assert batch[0].shape == (16, 2)


def test_gae_hand_computed():
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == (3,)
    assert ret.shape == (3,)
    assert adv[-1] != 0.0
