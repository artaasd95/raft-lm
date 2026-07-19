"""Experience replay buffer for off-policy RL."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class ReplayTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self._data: Deque[ReplayTransition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._data.append(
            ReplayTransition(state, action, reward, next_state, done)
        )

    def __len__(self) -> int:
        return len(self._data)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if len(self._data) < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        idx = np.random.choice(len(self._data), batch_size, replace=False)
        batch = [self._data[i] for i in idx]
        states = np.stack([b.state for b in batch])
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.stack([b.next_state for b in batch])
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def to_list(self) -> List[ReplayTransition]:
        return list(self._data)
