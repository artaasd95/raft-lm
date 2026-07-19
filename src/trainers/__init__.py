"""Training backends and orchestration."""

from src.trainers.base import TrainingBackend
from src.trainers.factory import get_training_backend, resolve_backend

__all__ = ["TrainingBackend", "get_training_backend", "resolve_backend"]
