"""Weighted composition of reward components."""

from __future__ import annotations

from typing import Any, List, Mapping, Tuple

import numpy as np

from src.domain.trajectory import RewardBatch
from src.rewards.base import BaseReward


class CompositeReward(BaseReward):
    """Sum weighted reward components from YAML config."""

    name = "composite"

    def __init__(self, components: list[tuple[BaseReward, float]]) -> None:
        self._components = components

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CompositeReward":
        from src.rewards.registry import build_reward

        items = cfg.get("components") or []
        built: List[Tuple[BaseReward, float]] = []
        for item in items:
            weight = float(item.get("weight", 1.0))
            comp_cfg = {"name": item["name"], **(item.get("params") or {})}
            built.append((build_reward(comp_cfg), weight))
        return cls(built)

    def compute(self, batch: Mapping[str, Any]) -> RewardBatch:
        if not self._components:
            n = len(batch.get("rewards", batch.get("labels", [])) or [0])
            return RewardBatch(values=np.zeros(max(n, 1), dtype=np.float32))

        total: np.ndarray | None = None
        components: dict[str, np.ndarray] = {}
        for reward_fn, weight in self._components:
            rb = reward_fn.compute(batch)
            scaled = rb.values * weight
            total = scaled if total is None else total + scaled
            components[reward_fn.name] = rb.values
        assert total is not None
        return RewardBatch(values=total.astype(np.float32), components=components)
