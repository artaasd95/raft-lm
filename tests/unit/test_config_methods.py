"""Config validation for RL methods."""

from copy import deepcopy

import pytest

from src.utils.config import DEFAULT_CONFIG, validate_config


def test_ppo_env_config_validates():
    cfg = {
        "method": "ppo_env",
        "experiment_name": "ppo_smoke",
        "description": "",
        "training": {
            "backend": "ppo_env",
            "num_epochs": 2,
            "seed": 42,
            "device": "cpu",
            "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0},
            "loss": {"type": "CrossEntropyLoss"},
        },
    }
    assert validate_config(cfg) is True


def test_dpo_config_validates():
    cfg = {
        "method": "dpo",
        "experiment_name": "dpo_smoke",
        "description": "",
        "model": {"type": "hf_lora", "model_id": "stub"},
        "training": {
            "backend": "dpo",
            "num_epochs": 1,
            "seed": 42,
            "device": "cpu",
            "optimizer": {"type": "Adam", "lr": 0.0002, "weight_decay": 0.0},
            "loss": {"type": "CrossEntropyLoss"},
        },
    }
    assert validate_config(cfg) is True


def test_unsloth_rejected_for_dpo():
    cfg = {
        "method": "dpo",
        "experiment_name": "bad",
        "description": "",
        "model": {"type": "hf_lora", "model_id": "stub"},
        "training": {
            "backend": "unsloth",
            "num_epochs": 1,
            "seed": 42,
            "device": "cpu",
            "optimizer": {"type": "Adam", "lr": 0.0002, "weight_decay": 0.0},
            "loss": {"type": "CrossEntropyLoss"},
        },
    }
    with pytest.raises(ValueError, match="Unsloth"):
        validate_config(cfg)


def test_ddp_backend_validates():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["training"]["backend"] = "ddp"
    assert validate_config(cfg) is True


def test_factory_backend_sync():
    from src.trainers.constants import SUPPORTED_BACKENDS

    cfg = deepcopy(DEFAULT_CONFIG)
    for backend in SUPPORTED_BACKENDS:
        if backend in {"dpo", "kto", "ppo_lm", "grpo", "gigpo", "ppo_env", "dqn_env", "ray"}:
            continue
        cfg["training"]["backend"] = backend
        assert validate_config(cfg) is True
