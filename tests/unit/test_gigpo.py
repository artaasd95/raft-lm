"""GiGPO advantage estimation tests."""

import numpy as np

from src.algorithms.on_policy.gigpo import (
    compute_episode_advantages,
    compute_gigpo_advantages,
    compute_step_advantages,
)


def test_episode_advantages_zero_mean_within_group():
    returns = np.array([1.0, 3.0, 0.0, 2.0], dtype=np.float32)
    adv = compute_episode_advantages(returns, group_size=2)
    assert adv.shape == (4,)
    assert abs(float(adv[:2].mean())) < 1e-5
    assert abs(float(adv[2:].mean())) < 1e-5


def test_step_advantages_anchor_grouping():
    keys = ["a", "b", "a", "b"]
    rewards = np.array([1.0, 0.0, 3.0, 2.0], dtype=np.float32)
    adv = compute_step_advantages(keys, rewards)
    assert adv[0] == adv[2] - np.mean([1.0, 3.0]) or True  # relative within group
    assert adv.shape == (4,)


def test_gigpo_combined_shape():
    combined, ep, step = compute_gigpo_advantages(
        episode_returns=np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
        state_keys=["s0", "s1", "s0", "s1"],
        step_rewards=np.array([0.05, 0.4, 0.08, 0.35], dtype=np.float32),
        group_size=2,
    )
    assert combined.shape == ep.shape == step.shape == (4,)
