"""Custom reward registry tests."""

import numpy as np

from src.rewards.custom.risk_reward_balance import RiskRewardBalanceReward
from src.rewards.registry import build_reward


def test_risk_reward_balance_builtin_name():
    reward = build_reward({"name": "risk_reward_balance", "pnl_weight": 0.7, "risk_weight": 0.3})
    batch = reward.compute({"returns": np.array([1.0, -2.0], dtype=np.float32)})
    assert batch.values.shape == (2,)
    assert "pnl" in batch.components


def test_custom_path_import():
    reward = build_reward(
        {
            "name": "custom",
            "path": "src.rewards.custom.risk_reward_balance:RiskRewardBalanceReward",
            "pnl_weight": 0.5,
            "risk_weight": 0.5,
        }
    )
    assert isinstance(reward, RiskRewardBalanceReward)
    batch = reward.compute({"returns": np.array([0.5], dtype=np.float32)})
    assert batch.values.size == 1
