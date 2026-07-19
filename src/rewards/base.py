"""Reward function base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from src.domain.trajectory import RewardBatch


class BaseReward(ABC):
    """Compute scalar rewards from a batch dict."""

    name: str = "base"

    @abstractmethod
    def compute(self, batch: Mapping[str, Any]) -> RewardBatch:
        """Return reward values and optional component breakdown."""
