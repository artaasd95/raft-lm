"""Training backend factory."""

from __future__ import annotations

from src.training.backends.base import TrainingBackend
from src.training.backends.mlp_backend import MLPBackend
from src.training.backends.unsloth_trainer import UnslothTrainer

SUPPORTED_BACKENDS = {"mlp", "unsloth"}


def get_training_backend(name: str) -> TrainingBackend:
    if name not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"Unsupported training.backend {name!r}. Supported: {supported}")
    if name == "mlp":
        return MLPBackend()
    if name == "unsloth":
        return UnslothTrainer()
    raise ValueError(f"Unsupported training.backend {name!r}")
