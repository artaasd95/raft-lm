"""Shared pytest fixtures for RAFT-LM."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.envs.risk_allocation import RiskAllocationEnv
from src.generation.mock import MockGenerator, build_generator


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_mlp_config() -> dict:
    return {
        "method": "supervised",
        "model": {
            "type": "SimpleMLP",
            "input_dim": 8,
            "hidden_dim": 16,
            "output_dim": 3,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "data": {
            "train_size": 32,
            "val_size": 16,
            "test_size": 16,
            "batch_size": 8,
            "num_workers": 0,
        },
        "training": {
            "seed": 42,
            "device": "cpu",
            "num_epochs": 1,
            "smoke": True,
            "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0},
        },
        "evaluation": {"metrics": ["accuracy", "test_loss"]},
    }


@pytest.fixture
def mock_generator() -> MockGenerator:
    return MockGenerator(risk_level="low")


@pytest.fixture
def mock_generator_from_config(tiny_mlp_config):
    tiny_mlp_config["generation"] = {"backend": "mock", "risk_level": "medium"}
    return build_generator(tiny_mlp_config)


@pytest.fixture
def risk_env() -> RiskAllocationEnv:
    return RiskAllocationEnv(seed=42)


@pytest.fixture
def tmp_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return run_dir
