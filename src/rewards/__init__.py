"""Extensible reward functions for RL and alignment."""

from src.rewards.base import BaseReward
from src.rewards.registry import build_reward

__all__ = ["BaseReward", "build_reward"]


def __getattr__(name: str):
    if name == "CompositeReward":
        from src.rewards.composite import CompositeReward

        return CompositeReward
    raise AttributeError(name)
