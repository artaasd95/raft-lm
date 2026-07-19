"""Built-in reward components."""

from src.rewards.builtin.accuracy import TaskAccuracyReward
from src.rewards.builtin.format import FormatComplianceReward
from src.rewards.builtin.kl import KLPenaltyReward
from src.rewards.builtin.pnl import PnLReward
from src.rewards.builtin.risk import RiskCVaRReward

__all__ = [
    "TaskAccuracyReward",
    "FormatComplianceReward",
    "KLPenaltyReward",
    "PnLReward",
    "RiskCVaRReward",
]
