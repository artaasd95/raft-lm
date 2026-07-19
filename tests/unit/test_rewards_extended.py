"""Extended reward registry tests."""

from src.rewards.base import BaseReward
from src.rewards.builtin.format import FormatComplianceReward
from src.rewards.builtin.pnl import PnLReward
from src.rewards.registry import build_reward


def test_pnl_reward():
    reward = build_reward({"name": "pnl"})
    assert isinstance(reward, PnLReward)
    batch = reward.compute({"completions": ['{"pnl": 0.5}']})
    assert len(batch.values) == 1


def test_format_reward():
    reward = build_reward({"name": "format_compliance"})
    assert isinstance(reward, FormatComplianceReward)
    batch = reward.compute({"completions": ['{"ok": true}']})
    assert batch.values[0] >= 0.0


def test_base_reward_contract():
    reward = build_reward({"name": "task_accuracy"})
    assert isinstance(reward, BaseReward)
    batch = reward.compute({"correct": [1.0]})
    assert float(batch.values[0]) == 1.0
