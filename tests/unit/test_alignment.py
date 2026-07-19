"""Unit tests for alignment algorithms."""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from src.algorithms.preference.dpo import dpo_loss
from src.algorithms.on_policy.grpo import compute_grpo_advantages
from src.algorithms.preference.kl import kl_divergence
from src.algorithms.preference.kto import kto_loss
from src.algorithms.on_policy.ppo_lm import ppo_lm_loss
from src.algorithms.datasets.preference import PreferenceDataset
from src.data.pipeline.cards import PreferencePair


def test_dpo_loss_prefers_chosen():
    loss_good = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([-2.0]),
        torch.tensor([-1.0]),
        torch.tensor([-1.0]),
        beta=0.1,
    )
    loss_bad = dpo_loss(
        torch.tensor([-2.0]),
        torch.tensor([0.0]),
        torch.tensor([-1.0]),
        torch.tensor([-1.0]),
        beta=0.1,
    )
    assert loss_good.item() < loss_bad.item()


def test_kto_loss_finite():
    loss = kto_loss(
        torch.tensor([0.0]),
        torch.tensor([-1.0]),
        torch.tensor([0.0]),
        torch.tensor([-1.0]),
    )
    assert torch.isfinite(loss)


def test_kl_zero_when_equal():
    lp = torch.tensor([-1.0, -2.0])
    assert kl_divergence(lp, lp).item() == 0.0


def test_grpo_group_relative():
    rewards = np.array([1.0, 3.0, 0.0, 2.0], dtype=np.float32)
    adv = compute_grpo_advantages(rewards, group_size=2)
    assert adv.shape == (4,)
    assert np.isclose(adv[:2].mean(), 0.0, atol=1e-5)


def test_ppo_lm_clip():
    old = torch.tensor([-1.0, -1.5])
    new = torch.tensor([-1.01, -1.4])
    adv = torch.tensor([1.0, -1.0])
    loss = ppo_lm_loss(new, old, adv, clip_eps=0.2)
    assert torch.isfinite(loss)


def test_preference_dataset_from_pairs():
    pairs = [
        PreferencePair(
            pair_id="1",
            prompt="p",
            chosen="a",
            rejected="b",
        )
    ]
    ds = PreferenceDataset.from_pairs(pairs)
    assert len(ds) == 1
    assert ds[0]["chosen"] == "a"
