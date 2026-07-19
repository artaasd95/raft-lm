"""Rollout collection for LM RL (mock-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.rewards.base import BaseReward


@dataclass
class RolloutSample:
    prompt: str
    completion: str
    reward: float
    policy_logprob: float = 0.0
    ref_logprob: float = 0.0


@dataclass
class RolloutCollector:
    """Generate completions and score with a reward function."""

    reward_fn: Optional[BaseReward] = None
    samples: List[RolloutSample] = field(default_factory=list)

    def collect(
        self,
        prompts: List[str],
        generate_fn: Callable[[str], tuple[str, float]],
        ref_logprob_fn: Optional[Callable[[str, str], float]] = None,
    ) -> List[RolloutSample]:
        self.samples.clear()
        for prompt in prompts:
            completion, logprob = generate_fn(prompt)
            ref_lp = ref_logprob_fn(prompt, completion) if ref_logprob_fn else logprob
            reward = 0.0
            if self.reward_fn is not None:
                rb = self.reward_fn.compute(
                    {"completions": [completion], "policy_logprobs": [logprob], "ref_logprobs": [ref_lp]}
                )
                reward = float(rb.values[0])
            sample = RolloutSample(
                prompt=prompt,
                completion=completion,
                reward=reward,
                policy_logprob=logprob,
                ref_logprob=ref_lp,
            )
            self.samples.append(sample)
        return list(self.samples)

    def rewards_array(self) -> np.ndarray:
        return np.asarray([s.reward for s in self.samples], dtype=np.float32)
