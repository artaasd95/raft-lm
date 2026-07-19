"""Deep Q-Network (from scratch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.rl.buffers.replay import ReplayBuffer
from src.rl.networks.mlp import QNetwork


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 0.1
    target_update_interval: int = 10


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[DQNConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self._steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.config.epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(x)
            return int(q.argmax(dim=-1).item())

    def update(self, buffer: ReplayBuffer, batch_size: int = 32) -> dict[str, float]:
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._steps += 1
        if self._steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return {"loss": float(loss.item())}

    def hard_update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())
