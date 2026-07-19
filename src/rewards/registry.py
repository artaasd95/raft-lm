"""Reward registry — build from YAML name."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from src.rewards.base import BaseReward
from src.rewards.builtin.accuracy import TaskAccuracyReward
from src.rewards.builtin.format import FormatComplianceReward
from src.rewards.builtin.kl import KLPenaltyReward
from src.rewards.builtin.pnl import PnLReward
from src.rewards.builtin.risk import RiskCVaRReward

_REGISTRY: Dict[str, type] = {
    "task_accuracy": TaskAccuracyReward,
    "format_compliance": FormatComplianceReward,
    "kl_penalty": KLPenaltyReward,
    "pnl": PnLReward,
    "risk_cvar": RiskCVaRReward,
}


def build_reward(cfg: Mapping[str, Any]) -> BaseReward:
    name = str(cfg.get("name", "composite"))
    if name == "composite":
        from src.rewards.composite import CompositeReward

        return CompositeReward.from_config(cfg)
    cls = _REGISTRY.get(name)
    if cls is None:
        supported = ", ".join(sorted(list(_REGISTRY.keys()) + ["composite"]))
        raise ValueError(f"Unknown reward {name!r}. Supported: {supported}")
    if name == "risk_cvar":
        params = dict(cfg)
        params.pop("name", None)
        return RiskCVaRReward(
            alpha=float(params.get("alpha", 0.05)),
            scale=float(params.get("scale", 1.0)),
        )
    return cls()
