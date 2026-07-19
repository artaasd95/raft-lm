"""KL divergence helper for LM RL."""

from __future__ import annotations

import torch


def kl_divergence(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(policy || ref) from per-token log-probs."""
    log_ratio = policy_logprobs - ref_logprobs
    return torch.clamp(log_ratio, min=0.0).mean()
