"""Classical env RL backends (PPO / DQN)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from gymnasium.spaces import Box, Discrete

from src.algorithms.actor_critic.ppo import PPOAgent, PPOConfig
from src.algorithms.value.dqn import DQNAgent, DQNConfig
from src.buffers.replay import ReplayBuffer
from src.buffers.rollout import RolloutBuffer
from src.envs.risk_allocation import RiskAllocationEnv
from src.trainers.base import TrainingBackend
from src.utils.reproducibility import get_device, set_seed


class PPOEnvBackend(TrainingBackend):
    """Train PPO on RiskAllocationEnv."""

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        device = str(get_device(
            None if config["training"]["device"] == "auto" else config["training"]["device"]
        ))
        algo = config.get("algorithm", {})
        env = RiskAllocationEnv(seed=config["training"]["seed"])
        obs_space = env.observation_space
        action_space = env.action_space
        assert isinstance(obs_space, Box) and obs_space.shape is not None
        assert isinstance(action_space, Discrete)
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.n)
        agent = PPOAgent(
            obs_dim,
            action_dim,
            PPOConfig(
                clip_eps=float(algo.get("clip_eps", 0.2)),
                gamma=float(algo.get("gamma", 0.99)),
                gae_lambda=float(algo.get("gae_lambda", 0.95)),
            ),
            device=device,
        )
        num_epochs = config["training"]["num_epochs"]
        episode_rewards: list[float] = []
        for _ in range(num_epochs):
            state, _ = env.reset(seed=config["training"]["seed"])
            buffer = RolloutBuffer()
            ep_reward = 0.0
            done = False
            while not done:
                action, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.add(state, action, reward, done, log_prob, value)
                ep_reward += reward
                state = next_state
            stats = agent.update(buffer)
            episode_rewards.append(ep_reward)
            if exp_logger is not None:
                exp_logger.log_metrics({"episode_reward": ep_reward, **stats}, step=len(episode_rewards))
        metrics = {
            "mean_episode_reward": float(sum(episode_rewards) / max(len(episode_rewards), 1)),
            "final_loss": stats.get("loss", 0.0),
        }
        _write_metrics(run_dir, metrics)
        return metrics


class DQNEnvBackend(TrainingBackend):
    """Train DQN on RiskAllocationEnv."""

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        device = str(get_device(
            None if config["training"]["device"] == "auto" else config["training"]["device"]
        ))
        algo = config.get("algorithm", {})
        env = RiskAllocationEnv(seed=config["training"]["seed"])
        obs_space = env.observation_space
        action_space = env.action_space
        assert isinstance(obs_space, Box) and obs_space.shape is not None
        assert isinstance(action_space, Discrete)
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.n)
        agent = DQNAgent(
            obs_dim,
            action_dim,
            DQNConfig(gamma=float(algo.get("gamma", 0.99))),
            device=device,
        )
        buffer = ReplayBuffer(capacity=5000)
        num_epochs = config["training"]["num_epochs"]
        losses: list[float] = []
        for ep in range(num_epochs):
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.push(state, action, reward, next_state, done)
                state = next_state
                if len(buffer) >= 32:
                    stats = agent.update(buffer, batch_size=32)
                    losses.append(stats["loss"])
            if exp_logger is not None:
                exp_logger.log_metrics({"dqn_loss": losses[-1] if losses else 0.0}, step=ep)
        metrics = {
            "mean_dqn_loss": float(sum(losses) / max(len(losses), 1)),
            "buffer_size": len(buffer),
        }
        _write_metrics(run_dir, metrics)
        return metrics


def _write_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
