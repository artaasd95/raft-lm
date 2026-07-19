"""Training-time text generation (mock/stub for rollouts)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> Tuple[str, float]:
        """Return completion and policy log-probability estimate."""
