"""Configuration specs for training methods, rewards, and LoRA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SUPPORTED_METHODS = frozenset(
    {
        "supervised",
        "dpo",
        "kto",
        "ppo_lm",
        "grpo",
        "ppo_env",
        "dqn_env",
    }
)

UNSLOTH_ALLOWED_METHODS = frozenset({"supervised"})


@dataclass
class LoRASpec:
    """LoRA adapter configuration."""

    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LoRASpec":
        if not data:
            return cls(enabled=False)
        return cls(
            enabled=bool(data.get("enabled", True)),
            r=int(data.get("r", 16)),
            lora_alpha=int(data.get("lora_alpha", 32)),
            lora_dropout=float(data.get("lora_dropout", 0.05)),
            target_modules=list(data.get("target_modules", ["q_proj", "v_proj"])),
            bias=str(data.get("bias", "none")),
        )


@dataclass
class RewardSpec:
    """Reward recipe reference."""

    name: str = "composite"
    components: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "RewardSpec":
        if not cfg:
            return cls()
        return cls(
            name=str(cfg.get("name", "composite")),
            components=list(cfg.get("components") or []),
        )


@dataclass
class AlgorithmSpec:
    """Algorithm hyperparameters."""

    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    kl_coef: float = 0.05
    group_size: int = 4
    beta: float = 0.1
    ref_free: bool = False

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "AlgorithmSpec":
        if not cfg:
            return cls()
        return cls(
            clip_eps=float(cfg.get("clip_eps", 0.2)),
            gae_lambda=float(cfg.get("gae_lambda", 0.95)),
            gamma=float(cfg.get("gamma", 0.99)),
            kl_coef=float(cfg.get("kl_coef", 0.05)),
            group_size=int(cfg.get("group_size", 4)),
            beta=float(cfg.get("beta", 0.1)),
            ref_free=bool(cfg.get("ref_free", False)),
        )


@dataclass
class MethodSpec:
    """Top-level method discriminator."""

    method: str = "supervised"
    reward: RewardSpec = field(default_factory=RewardSpec)
    algorithm: AlgorithmSpec = field(default_factory=AlgorithmSpec)
    lora: LoRASpec = field(default_factory=LoRASpec)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MethodSpec":
        method = str(config.get("method", "supervised"))
        if method not in SUPPORTED_METHODS:
            supported = ", ".join(sorted(SUPPORTED_METHODS))
            raise ValueError(f"Unsupported method {method!r}. Supported: {supported}")
        lora_data = config.get("lora") or config.get("model", {}).get("lora")
        return cls(
            method=method,
            reward=RewardSpec.from_config(config.get("reward")),
            algorithm=AlgorithmSpec.from_config(config.get("algorithm")),
            lora=LoRASpec.from_dict(lora_data if isinstance(lora_data, dict) else None),
        )
