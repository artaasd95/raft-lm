"""Deterministic mock generator for CI and smoke trainers."""

from __future__ import annotations

import json
from typing import Tuple

from src.generation.base import BaseGenerator


class MockGenerator(BaseGenerator):
    """Returns risk-aware JSON completions without external LLM calls."""

    def __init__(self, risk_level: str = "low") -> None:
        self.risk_level = risk_level

    def generate(self, prompt: str) -> Tuple[str, float]:
        completion = json.dumps({"risk": self.risk_level, "action": "hold"})
        logprob = -float(len(prompt) + len(completion)) * 0.01
        return completion, logprob


def build_generator(config: dict) -> BaseGenerator:
    gen_cfg = config.get("generation") or {}
    backend = str(gen_cfg.get("backend", "mock"))
    if backend != "mock":
        raise ValueError(
            f"Unsupported generation.backend {backend!r}. Only 'mock' is available "
            "(inference/serving adapters were removed)."
        )
    return MockGenerator(risk_level=str(gen_cfg.get("risk_level", "low")))
