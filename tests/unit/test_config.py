"""Unit tests for experiment configuration handling."""

from copy import deepcopy

import pytest

from src.utils.config import DEFAULT_CONFIG, resolve_config, validate_config


def test_resolve_config_applies_defaults():
    config = resolve_config(
        {
            "experiment_name": "tiny",
            "model": {"input_dim": 4, "output_dim": 2},
            "data": {"train_size": 16, "val_size": 8, "test_size": 8},
            "training": {"num_epochs": 1},
        }
    )

    assert config["config_version"] == 1
    assert config["model"]["type"] == "SimpleMLP"
    assert config["model"]["hidden_dim"] == DEFAULT_CONFIG["model"]["hidden_dim"]
    assert config["data"]["batch_size"] == DEFAULT_CONFIG["data"]["batch_size"]
    assert config["training"]["optimizer"]["type"] == "Adam"
    assert validate_config(config) is True


def test_validate_config_requires_top_level_sections():
    with pytest.raises(ValueError, match="Missing required config field: model"):
        validate_config({"data": {}, "training": {}})


def test_validate_config_rejects_unknown_model_type():
    config = deepcopy(DEFAULT_CONFIG)
    config["model"]["type"] = "UnknownModel"

    with pytest.raises(ValueError, match="model.type"):
        validate_config(config)


def test_validate_config_rejects_invalid_ranges():
    config = deepcopy(DEFAULT_CONFIG)
    config["data"]["batch_size"] = 0

    with pytest.raises(ValueError, match="data.batch_size"):
        validate_config(config)


def test_validate_config_rejects_unknown_nested_field():
    config = deepcopy(DEFAULT_CONFIG)
    config["training"]["optimizer"]["momentum"] = 0.9

    with pytest.raises(ValueError, match="training.optimizer.momentum"):
        validate_config(config)


def test_validate_unsloth_hf_lora_config():
    config = {
        "config_version": 1,
        "experiment_name": "unsloth_test",
        "description": "",
        "model": {
            "type": "hf_lora",
            "model_id": "qwen2.5-0.5b",
            "lora": {"r": 8},
            "quantization": {"load_in_4bit": True},
        },
        "data": {
            "dataset_type": "SFTJsonl",
            "data_source": "distilled",
            "distilled_corpus": "risk_sft_v1",
            "batch_size": 2,
            "num_workers": 0,
            "max_seq_length": 256,
        },
        "training": {
            "backend": "unsloth",
            "num_epochs": 1,
            "seed": 42,
            "device": "cpu",
            "optimizer": {"type": "Adam", "lr": 0.0002, "weight_decay": 0.0},
            "loss": {"type": "CrossEntropyLoss"},
        },
        "evaluation": {"metrics": ["perplexity", "cvar"]},
        "output": {"results_dir": "experiments/results", "adapters_dir": "experiments/adapters"},
    }
    assert validate_config(config) is True
