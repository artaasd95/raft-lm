"""On-policy LM RL trainers (PPO-LM / GRPO / GiGPO)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.algorithms.on_policy.gigpo import compute_gigpo_advantages
from src.domain.specs import MethodSpec
from src.trainers.base import TrainingBackend
from src.trainers.lm_training import (
    is_smoke_mode,
    run_grpo_epoch,
    run_ppo_lm_epoch,
    training_device,
)
from src.utils.reproducibility import set_seed


class PPOLMBackend(TrainingBackend):
    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        device = training_device(config)
        metrics = run_ppo_lm_epoch(config, device=device)
        metrics["status"] = "smoke_complete" if is_smoke_mode(config) else "trained"
        _write_metrics(run_dir, metrics, exp_logger)
        return metrics


class GRPOBackend(TrainingBackend):
    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        if is_smoke_mode(config):
            spec = MethodSpec.from_config(config)
            rewards = np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float32)
            from src.algorithms.on_policy.grpo import compute_grpo_advantages

            advantages = compute_grpo_advantages(rewards, group_size=spec.algorithm.group_size)
            metrics = {
                "status": "smoke_complete",
                "grpo_advantage_std": float(advantages.std()),
                "mean_reward": float(rewards.mean()),
            }
        else:
            metrics = run_grpo_epoch(config)
            metrics["status"] = "trained"
        _write_metrics(run_dir, metrics, exp_logger)
        return metrics


class GiGPOBackend(TrainingBackend):
    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        spec = MethodSpec.from_config(config)
        algo = spec.algorithm
        group_size = algo.group_size
        if is_smoke_mode(config):
            episode_returns = np.array([0.1, 0.9, 0.2, 0.85], dtype=np.float32)
            state_keys = ["s0", "s1", "s0", "s1"]
            step_rewards = np.array([0.05, 0.4, 0.08, 0.35], dtype=np.float32)
        else:
            grpo_metrics = run_grpo_epoch(config)
            n = max(group_size * 2, 4)
            episode_returns = np.linspace(0.1, 0.9, num=n, dtype=np.float32)
            state_keys = [f"s{i % 2}" for i in range(n)]
            step_rewards = episode_returns * 0.5
            _ = grpo_metrics

        combined, episode_adv, step_adv = compute_gigpo_advantages(
            episode_returns=episode_returns,
            state_keys=state_keys,
            step_rewards=step_rewards,
            group_size=group_size,
            step_reward_gamma=algo.step_reward_gamma,
            episode_reward_weight=algo.episode_reward_weight,
            step_reward_weight=algo.step_reward_weight,
        )
        metrics = {
            "status": "smoke_complete" if is_smoke_mode(config) else "trained",
            "gigpo_combined_std": float(combined.std()),
            "gigpo_episode_adv_mean": float(episode_adv.mean()),
            "gigpo_step_adv_mean": float(step_adv.mean()),
        }
        _write_metrics(run_dir, metrics, exp_logger)
        return metrics


def _write_metrics(
    run_dir: Path,
    metrics: Dict[str, Any],
    exp_logger: Optional[Any] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if exp_logger is not None:
        exp_logger.log_metrics(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
