"""On-policy LM RL trainers (PPO-LM / GRPO / GiGPO)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.algorithms.on_policy.gigpo import compute_gigpo_advantages
from src.algorithms.on_policy.grpo import compute_grpo_advantages
from src.algorithms.on_policy.ppo_lm import ppo_lm_loss
from src.algorithms.preference.kl import kl_divergence
from src.algorithms.rollouts.collector import RolloutCollector
from src.domain.specs import MethodSpec
from src.generation import build_generator
from src.rewards.registry import build_reward
from src.trainers.base import TrainingBackend
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
        spec = MethodSpec.from_config(config)
        reward_fn = build_reward(config.get("reward") or {"name": "task_accuracy"})
        prompts = _prompts_from_config(config)
        collector = RolloutCollector(reward_fn=reward_fn)
        generator = build_generator(config)

        def generate_fn(prompt: str) -> tuple[str, float]:
            return generator.generate(prompt)

        collector.collect(prompts, generate_fn)
        rewards = collector.rewards_array()
        old_lps = torch.tensor([s.policy_logprob for s in collector.samples], dtype=torch.float32)
        new_lps = old_lps + torch.randn_like(old_lps) * 0.01
        advantages = torch.tensor(rewards, dtype=torch.float32)
        loss = ppo_lm_loss(new_lps, old_lps, advantages, clip_eps=spec.algorithm.clip_eps)
        kl = kl_divergence(new_lps, old_lps)
        metrics = {
            "status": "smoke_complete",
            "ppo_lm_loss": float(loss.item()),
            "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
            "kl_to_ref": float(kl.item()),
        }
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
        spec = MethodSpec.from_config(config)
        group_size = spec.algorithm.group_size
        rewards = np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float32)
        advantages = compute_grpo_advantages(rewards, group_size=group_size)
        metrics = {
            "status": "smoke_complete",
            "grpo_advantage_std": float(advantages.std()),
            "mean_reward": float(rewards.mean()),
        }
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
        episode_returns = np.array([0.1, 0.9, 0.2, 0.85], dtype=np.float32)
        state_keys = ["s0", "s1", "s0", "s1"]
        step_rewards = np.array([0.05, 0.4, 0.08, 0.35], dtype=np.float32)
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
            "status": "smoke_complete",
            "gigpo_combined_std": float(combined.std()),
            "gigpo_episode_adv_mean": float(episode_adv.mean()),
            "gigpo_step_adv_mean": float(step_adv.mean()),
        }
        _write_metrics(run_dir, metrics, exp_logger)
        return metrics


def _prompts_from_config(config: Dict[str, Any]) -> list[str]:
    data = config.get("data", {})
    prompts = data.get("prompts")
    if prompts:
        return list(prompts)
    return ["Assess tail risk for a long call spread."]


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
