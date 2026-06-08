"""Integration tests for --loss CLI override."""

import json

import pytest

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

from scripts.train import run_training


def _base_config(results_dir: str) -> dict:
    return {
        "config_version": 1,
        "experiment_name": "loss_cli",
        "model": {
            "type": "SimpleMLP",
            "input_dim": 4,
            "hidden_dim": 8,
            "output_dim": 3,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "data": {
            "train_size": 16,
            "val_size": 8,
            "test_size": 8,
            "batch_size": 4,
            "num_workers": 0,
        },
        "training": {
            "num_epochs": 1,
            "optimizer": {"type": "Adam", "lr": 0.01, "weight_decay": 0.0},
            "loss": {"type": "ce"},
            "seed": 99,
            "device": "cpu",
        },
        "evaluation": {"metrics": ["accuracy", "cvar"]},
        "logging": {"save_checkpoints": True},
        "output": {"results_dir": results_dir},
    }


@pytest.mark.parametrize("loss", ["ce", "cvar_penalized", "tail_aware"])
def test_loss_cli_runs(tmp_path, loss):
    config_path = tmp_path / f"config_{loss}.json"
    config_path.write_text(json.dumps(_base_config(str(tmp_path / loss))), encoding="utf-8")
    run_dir = run_training(str(config_path), loss_override=loss)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "test_metrics" in metrics
    assert "accuracy" in metrics["test_metrics"]
    assert "cvar" in metrics["test_metrics"]
