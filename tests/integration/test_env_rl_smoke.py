"""Integration smoke for env RL backends."""

import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.training.backends.env_rl_backend import DQNEnvBackend, PPOEnvBackend


@pytest.mark.timeout(60)
def test_ppo_env_smoke():
    config = {
        "training": {"seed": 42, "num_epochs": 3, "device": "cpu"},
        "algorithm": {"clip_eps": 0.2, "gae_lambda": 0.95, "gamma": 0.99},
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        metrics = PPOEnvBackend().run(config, run_dir)
        assert "mean_episode_reward" in metrics
        assert (run_dir / "metrics.json").exists()


@pytest.mark.timeout(60)
def test_dqn_env_smoke():
    config = {
        "training": {"seed": 42, "num_epochs": 5, "device": "cpu"},
        "algorithm": {"gamma": 0.99},
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        metrics = DQNEnvBackend().run(config, run_dir)
        assert "mean_dqn_loss" in metrics
