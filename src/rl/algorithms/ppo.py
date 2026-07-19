"""Proximal Policy Optimization (from scratch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.rl.algorithms.gae import compute_gae
from src.rl.buffers.rollout import RolloutBuffer
from src.rl.networks.mlp import ActorCritic


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    train_epochs: int = 4
    entropy_coef: float = 0.01


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        data = buffer.as_arrays()
        if len(data["rewards"]) == 0:
            return {"loss": 0.0}
        advantages, returns = compute_gae(
            data["rewards"],
            data["values"],
            data["dones"],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        states = torch.as_tensor(data["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(data["actions"], dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(
            data["log_probs"], dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.config.train_epochs):
            logits, values = self.net(states)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, ret_t)
            loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
        return {"loss": total_loss / self.config.train_epochs}

    def select_action(self, state: np.ndarray) -> tuple[int, float, float]:
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.net.act(x.squeeze(0))
