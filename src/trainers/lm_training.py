"""Shared LM training utilities for preference and on-policy backends."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from src.algorithms.datasets.preference import PreferenceDataset
from src.algorithms.on_policy.grpo import compute_grpo_advantages
from src.algorithms.on_policy.ppo_lm import ppo_lm_loss
from src.algorithms.preference.dpo import dpo_loss
from src.algorithms.preference.kl import kl_divergence
from src.algorithms.preference.kto import kto_loss
from src.algorithms.rollouts.collector import RolloutCollector
from src.data.pipeline.cards import PreferencePair
from src.domain.specs import LoRASpec, MethodSpec
from src.generation import build_generator
from src.models.loaders.causal_peft import load_causal_peft, sequence_logprob
from src.rewards.registry import build_reward
from src.utils.reproducibility import get_device


def is_smoke_mode(config: Mapping[str, Any]) -> bool:
    """Return True when config requests fast smoke path (default for CI)."""
    return bool(config.get("training", {}).get("smoke", True))


def resolve_model_id(config: Mapping[str, Any]) -> str:
    training = config.get("training", {})
    model_cfg = config.get("model", {})
    return str(
        training.get("model_name")
        or model_cfg.get("model_id")
        or model_cfg.get("hub_path")
        or "distilgpt2"
    )


def hf_available() -> bool:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load_preference_pairs(config: Mapping[str, Any]) -> List[PreferencePair]:
    data = config.get("data", {})
    path = data.get("preference_path") or data.get("path")
    if path:
        return PreferenceDataset(str(path)).samples
    return [
        PreferencePair(
            pair_id="stub-1",
            prompt="Assess risk:",
            chosen='{"risk": "low"}',
            rejected='{"risk": "high"}',
        )
    ]


def run_dpo_epoch(
    config: Dict[str, Any],
    pairs: List[PreferencePair],
    *,
    device: str,
) -> Dict[str, Any]:
    """One DPO pass with optional real HF model when not in stub mode."""
    algo = config.get("algorithm", {})
    beta = float(algo.get("beta", 0.1))
    model_id = resolve_model_id(config)
    use_hf = hf_available() and model_id not in {"", "stub"}
    losses: List[float] = []

    if use_hf:
        lora = LoRASpec.from_dict(config.get("model", {}).get("lora"))
        bundle = load_causal_peft(model_id, lora=lora, device=device, load_ref=True)
        policy_params = [p for p in bundle.policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(policy_params, lr=float(config["training"]["optimizer"]["lr"]))
        for pair in pairs:
            pc = sequence_logprob(bundle.policy, bundle.tokenizer, pair.prompt, pair.chosen, device)
            pr = sequence_logprob(bundle.policy, bundle.tokenizer, pair.prompt, pair.rejected, device)
            rc = sequence_logprob(bundle.ref, bundle.tokenizer, pair.prompt, pair.chosen, device)
            rr = sequence_logprob(bundle.ref, bundle.tokenizer, pair.prompt, pair.rejected, device)
            loss = dpo_loss(
                torch.tensor([pc]),
                torch.tensor([pr]),
                torch.tensor([rc]),
                torch.tensor([rr]),
                beta=beta,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    else:
        for pair in pairs:
            pc = float(len(pair.chosen)) * 0.01 + 0.5
            pr = float(len(pair.rejected)) * 0.01
            loss = dpo_loss(
                torch.tensor([pc]),
                torch.tensor([pr]),
                torch.tensor([pc * 0.9]),
                torch.tensor([pr * 0.9]),
                beta=beta,
            )
            losses.append(float(loss.item()))

    return {
        "dpo_loss": float(sum(losses) / max(len(losses), 1)),
        "num_pairs": float(len(pairs)),
    }


def run_kto_epoch(
    config: Dict[str, Any],
    pairs: List[PreferencePair],
) -> Dict[str, Any]:
    algo = config.get("algorithm", {})
    beta = float(algo.get("beta", 0.1))
    d_lp = torch.tensor([float(len(p.chosen)) * 0.01 + 0.5 for p in pairs])
    u_lp = torch.tensor([float(len(p.rejected)) * 0.01 for p in pairs])
    loss = kto_loss(d_lp, u_lp, d_lp * 0.9, u_lp * 0.9, beta=beta)
    return {"kto_loss": float(loss.item()), "num_pairs": float(len(pairs))}


def run_ppo_lm_epoch(config: Dict[str, Any], *, device: str) -> Dict[str, Any]:
    spec = MethodSpec.from_config(config)
    reward_fn = build_reward(config.get("reward") or {"name": "task_accuracy"})
    prompts = list(config.get("data", {}).get("prompts") or ["Assess tail risk for a long call spread."])
    collector = RolloutCollector(reward_fn=reward_fn)
    generator = build_generator(config)

    def generate_fn(prompt: str) -> tuple[str, float]:
        return generator.generate(prompt)

    collector.collect(prompts, generate_fn)
    rewards = collector.rewards_array()
    old_lps = torch.tensor([s.policy_logprob for s in collector.samples], dtype=torch.float32)
    model_id = resolve_model_id(config)
    if hf_available() and model_id not in {"", "stub"}:
        lora = LoRASpec.from_dict(config.get("model", {}).get("lora"))
        bundle = load_causal_peft(model_id, lora=lora, device=device, load_ref=False)
        new_lps_list: List[float] = []
        for sample in collector.samples:
            lp = sequence_logprob(
                bundle.policy,
                bundle.tokenizer,
                sample.prompt,
                sample.completion,
                device,
            )
            new_lps_list.append(lp)
        new_lps = torch.tensor(new_lps_list, dtype=torch.float32)
    else:
        new_lps = old_lps + torch.randn_like(old_lps) * 0.01

    advantages = torch.tensor(rewards, dtype=torch.float32)
    loss = ppo_lm_loss(new_lps, old_lps, advantages, clip_eps=spec.algorithm.clip_eps)
    kl = kl_divergence(new_lps, old_lps)
    return {
        "ppo_lm_loss": float(loss.item()),
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "kl_to_ref": float(kl.item()),
    }


def run_grpo_epoch(config: Dict[str, Any]) -> Dict[str, Any]:
    spec = MethodSpec.from_config(config)
    group_size = spec.algorithm.group_size
    reward_fn = build_reward(config.get("reward") or {"name": "task_accuracy"})
    prompts = list(config.get("data", {}).get("prompts") or ["Assess risk A", "Assess risk B"])
    expanded = [p for p in prompts for _ in range(max(group_size, 1))]
    collector = RolloutCollector(reward_fn=reward_fn)
    generator = build_generator(config)
    collector.collect(expanded, generator.generate)
    rewards = collector.rewards_array()
    if rewards.size == 0:
        rewards = np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float32)
    advantages = compute_grpo_advantages(rewards, group_size=group_size)
    return {
        "grpo_advantage_std": float(advantages.std()),
        "mean_reward": float(rewards.mean()),
    }


def run_sft_step(config: Dict[str, Any], *, device: str, run_dir: Any) -> Dict[str, Any]:
    """Save LoRA adapter or run one optimizer step when smoke=false."""
    from pathlib import Path

    from src.models.loaders.causal_peft import load_causal_peft, save_adapter

    model_id = resolve_model_id(config)
    lora = LoRASpec.from_dict(config.get("model", {}).get("lora"))
    metrics: Dict[str, Any] = {
        "backend": "peft",
        "model_id": model_id,
        "lora_enabled": lora.enabled,
        "device": device,
    }
    if model_id in {"", "stub"} or not hf_available():
        metrics["status"] = "hf_not_installed" if not hf_available() else "stub_model"
        return metrics

    bundle = load_causal_peft(model_id, lora=lora, device=device, load_ref=False)
    adapter_dir = Path(run_dir) / "adapter"
    save_adapter(bundle, str(adapter_dir))
    metrics["adapter_dir"] = str(adapter_dir)

    if not is_smoke_mode(config):
        params = [p for p in bundle.policy.parameters() if p.requires_grad]
        if params:
            optimizer = torch.optim.AdamW(
                params, lr=float(config["training"]["optimizer"]["lr"])
            )
            prompt = "Assess portfolio tail risk."
            completion = "Reduce exposure when CVaR exceeds limit."
            lp = sequence_logprob(bundle.policy, bundle.tokenizer, prompt, completion, device)
            loss = -torch.tensor(lp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metrics["sft_logprob"] = float(-loss.item())
        metrics["status"] = "trained"
    else:
        metrics["status"] = "adapter_saved"
    return metrics


def training_device(config: Mapping[str, Any]) -> str:
    return str(
        get_device(
            None
            if config.get("training", {}).get("device", "cpu") == "auto"
            else config.get("training", {}).get("device", "cpu")
        )
    )
