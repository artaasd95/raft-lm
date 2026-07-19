"""Unit tests for reward framework."""

import numpy as np
import pytest

from src.rewards.builtin.accuracy import TaskAccuracyReward
from src.rewards.builtin.kl import KLPenaltyReward
from src.rewards.builtin.risk import RiskCVaRReward
from src.rewards.composite import CompositeReward
from src.rewards.registry import build_reward


def test_composite_reward_weights():
    cfg = {
        "name": "composite",
        "components": [
            {"name": "task_accuracy", "weight": 2.0},
            {"name": "pnl", "weight": 1.0},
        ],
    }
    reward = build_reward(cfg)
    batch = {"correct": [1.0, 0.0], "returns": [0.5, -0.5]}
    rb = reward.compute(batch)
    assert rb.values.shape == (2,)
    assert rb.values[0] > rb.values[1]


def test_risk_cvar_component():
    r = RiskCVaRReward(alpha=0.05, scale=1.0)
    rb = r.compute({"returns": [0.1, -0.2, -0.5, 0.3]})
    assert rb.values.size == 4


def test_kl_penalty_non_negative():
    r = KLPenaltyReward()
    rb = r.compute({"policy_logprobs": [-1.0, -2.0], "ref_logprobs": [-1.5, -2.5]})
    assert np.all(rb.values <= 0)


def test_unknown_reward_raises():
    with pytest.raises(ValueError, match="Unknown reward"):
        build_reward({"name": "not_a_reward"})


def test_task_accuracy_from_predictions():
    r = TaskAccuracyReward()
    rb = r.compute({"labels": [1, 0], "predictions": [1, 1]})
    assert rb.values.tolist() == [1.0, 0.0]
