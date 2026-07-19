"""Reward registry — build from YAML name."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Mapping, Type

from src.rewards.base import BaseReward
from src.rewards.builtin.accuracy import TaskAccuracyReward
from src.rewards.builtin.format import FormatComplianceReward
from src.rewards.builtin.kl import KLPenaltyReward
from src.rewards.builtin.pnl import PnLReward
from src.rewards.builtin.risk import RiskCVaRReward

_LAZY_LOADERS: Dict[str, Callable[[], Type[BaseReward]]] = {}  # reserved for lazy plugins
_REGISTRY: Dict[str, Type[BaseReward]] = {
    "task_accuracy": TaskAccuracyReward,
    "format_compliance": FormatComplianceReward,
    "kl_penalty": KLPenaltyReward,
    "pnl": PnLReward,
    "risk_cvar": RiskCVaRReward,
}


def _load_custom_class(path: str) -> type:
    if ":" in path:
        module_path, class_name = path.split(":", 1)
    else:
        module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseReward):
        raise TypeError(f"Custom reward {path!r} must subclass BaseReward")
    return cls


def build_reward(cfg: Mapping[str, Any]) -> BaseReward:
    name = str(cfg.get("name", "composite"))
    if name == "composite":
        from src.rewards.composite import CompositeReward

        return CompositeReward.from_config(cfg)
    custom_path = cfg.get("path") or cfg.get("class_path")
    if custom_path:
        cls = _load_custom_class(str(custom_path))
        params = {k: v for k, v in dict(cfg).items() if k not in {"name", "path", "class_path"}}
        return cls(**params)
    if name == "risk_reward_balance":
        from src.rewards.custom.risk_reward_balance import RiskRewardBalanceReward

        return RiskRewardBalanceReward(
            pnl_weight=float(cfg.get("pnl_weight", 0.6)),
            risk_weight=float(cfg.get("risk_weight", 0.4)),
            alpha=float(cfg.get("alpha", 0.05)),
        )
    reward_cls = _REGISTRY.get(name)
    if reward_cls is None:
        supported = ", ".join(
            sorted(list(_REGISTRY.keys()) + ["composite", "risk_reward_balance", "custom path"])
        )
        raise ValueError(f"Unknown reward {name!r}. Supported: {supported}")
    if name == "risk_cvar":
        params = dict(cfg)
        params.pop("name", None)
        return RiskCVaRReward(
            alpha=float(params.get("alpha", 0.05)),
            scale=float(params.get("scale", 1.0)),
        )
    return reward_cls()
