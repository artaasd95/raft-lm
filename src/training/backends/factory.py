"""Training backend factory."""

from __future__ import annotations

from src.training.backends.base import TrainingBackend
from src.training.constants import METHOD_TO_BACKEND, SUPPORTED_BACKENDS


def resolve_backend(config: dict) -> str:
    """Resolve backend from method + training.backend."""
    method = config.get("method", "supervised")
    backend = config.get("training", {}).get("backend")
    if backend:
        return backend
    return METHOD_TO_BACKEND.get(method, "mlp")


def get_training_backend(name: str) -> TrainingBackend:
    if name not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"Unsupported training.backend {name!r}. Supported: {supported}")
    if name == "mlp":
        from src.training.backends.mlp_backend import MLPBackend

        return MLPBackend()
    if name == "unsloth":
        from src.training.backends.unsloth_trainer import UnslothTrainer

        return UnslothTrainer()
    if name == "peft":
        from src.training.backends.peft_trainer import PeftTrainerBackend

        return PeftTrainerBackend()
    if name == "ddp":
        from src.training.backends.distributed_backend import DistributedDDPBackend

        return DistributedDDPBackend()
    if name == "fsdp":
        from src.training.backends.distributed_backend import DistributedFSDPBackend

        return DistributedFSDPBackend()
    if name == "dpo":
        from src.training.backends.alignment_backend import DPOBackend

        return DPOBackend()
    if name == "kto":
        from src.training.backends.alignment_backend import KTOBackend

        return KTOBackend()
    if name == "ppo_lm":
        from src.training.backends.lm_rl_backend import PPOLMBackend

        return PPOLMBackend()
    if name == "grpo":
        from src.training.backends.lm_rl_backend import GRPOBackend

        return GRPOBackend()
    if name == "ppo_env":
        from src.training.backends.env_rl_backend import PPOEnvBackend

        return PPOEnvBackend()
    if name == "dqn_env":
        from src.training.backends.env_rl_backend import DQNEnvBackend

        return DQNEnvBackend()
    raise ValueError(f"Unsupported training.backend {name!r}")
