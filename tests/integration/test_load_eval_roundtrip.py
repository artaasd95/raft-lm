"""Integration test: train → load checkpoint → evaluate."""

import json

import pytest

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

from scripts.train import run_training
from src.evaluation.report import evaluate_checkpoint


def _tiny_config(results_dir: str) -> dict:
    return {
        "config_version": 1,
        "experiment_name": "load_eval_roundtrip",
        "model": {
            "type": "SimpleMLP",
            "input_dim": 4,
            "hidden_dim": 8,
            "output_dim": 3,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "data": {
            "dataset_type": "SyntheticRiskDataset",
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
            "seed": 7,
            "device": "cpu",
        },
        "evaluation": {"metrics": ["accuracy", "cvar", "f1_score"]},
        "logging": {"save_checkpoints": True},
        "output": {"results_dir": results_dir},
    }


def test_train_load_eval_roundtrip(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_tiny_config(str(tmp_path / "runs"))), encoding="utf-8")

    run_dir = run_training(str(config_path))
    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    assert checkpoint.exists()

    report = evaluate_checkpoint(checkpoint, config_path)
    assert "task_metrics" in report
    assert "risk_metrics" in report
    assert "provenance" in report
    assert report["provenance"]["loader_source"] == "pytorch_checkpoint"
    assert "accuracy" in report["task_metrics"]
    assert "cvar" in report["risk_metrics"]
