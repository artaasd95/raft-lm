"""PPO loss for language model policy (token-level simplified)."""

from __future__ import annotations

import torch


def ppo_lm_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(log_probs - old_log_probs)
    adv = advantages.detach()
    if adv.ndim == 1 and log_probs.ndim > 1:
        adv = adv.unsqueeze(-1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.min(surr1, surr2).mean()
