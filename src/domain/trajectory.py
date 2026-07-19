"""RL trajectory and reward batch types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class Transition:
    """Single environment or rollout step."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Sequence of transitions."""

    transitions: List[Transition] = field(default_factory=list)

    def returns(self, gamma: float = 0.99) -> List[float]:
        out: List[float] = []
        g = 0.0
        for t in reversed(self.transitions):
            g = t.reward + gamma * g * (0.0 if t.done else 1.0)
            out.append(g)
        out.reverse()
        return out


@dataclass
class RewardBatch:
    """Output of a reward function."""

    values: np.ndarray
    components: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values.size else 0.0
