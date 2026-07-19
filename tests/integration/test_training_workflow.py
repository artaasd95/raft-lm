"""
Integration tests for training workflow.

Tests complete training pipeline from data loading to evaluation.
"""

import json

import pytest
import torch

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

from scripts.train import run_training  # noqa: E402


def _tiny_config(results_dir: str) -> dict:
    return {
        "config_version": 1,
        "experiment_name": "tiny_training_workflow",
        "description": "Tiny integration test config",
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
            "train_size": 24,
            "val_size": 12,
            "test_size": 12,
            "batch_size": 6,
            "num_workers": 0,
        },
        "training": {
            "num_epochs": 2,
            "optimizer": {
                "type": "Adam",
                "lr": 0.01,
                "weight_decay": 0.0,
            },
            "loss": {
                "type": "CrossEntropyLoss",
            },
            "seed": 123,
            "device": "cpu",
        },
        "evaluation": {
            "metrics": ["accuracy", "cvar"],
        },
        "logging": {
            "log_interval": 1,
            "save_checkpoints": True,
            "checkpoint_interval": 1,
        },
        "output": {
            "results_dir": results_dir,
        },
    }


def _write_config(tmp_path, config: dict) -> str:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestTrainingWorkflow:
    """Test suite for end-to-end training workflow."""
    
    def test_complete_training_run(self, tmp_path):
        """Test a complete training run on toy data."""
        config_path = _write_config(tmp_path, _tiny_config(str(tmp_path / "results")))

        run_dir = run_training(config_path)

        assert (run_dir / "resolved_config.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "run_info.json").exists()
        assert (run_dir / "checkpoints" / "best_model.pt").exists()

        resolved_config = _read_json(run_dir / "resolved_config.json")
        metrics = _read_json(run_dir / "metrics.json")
        run_info = _read_json(run_dir / "run_info.json")

        assert resolved_config["training"]["seed"] == 123
        assert len(metrics["train_metrics"]) == 2
        assert len(metrics["val_metrics"]) == 2
        assert "accuracy" in metrics["test_metrics"]
        assert "cvar" in metrics["test_metrics"]
        assert run_info["seed"] == 123
        assert "git_commit" in run_info
        assert "timestamp" in run_info
        assert "started_at" in run_info
        assert "completed_at" in run_info
    
    def test_checkpoint_save_load(self, tmp_path):
        """Saved best checkpoint can be loaded with weights_only=True."""
        config_path = _write_config(tmp_path, _tiny_config(str(tmp_path / "results")))
        run_dir = run_training(config_path)
        checkpoint = run_dir / "checkpoints" / "best_model.pt"
        assert checkpoint.exists()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state

    def test_training_with_validation(self, tmp_path):
        """Validation metrics are recorded during training."""
        config_path = _write_config(tmp_path, _tiny_config(str(tmp_path / "results")))
        run_dir = run_training(config_path)
        metrics = _read_json(run_dir / "metrics.json")
        assert metrics["val_metrics"]
        assert all("val_loss" in row for row in metrics["val_metrics"])


class TestExperimentWorkflow:
    """Test suite for experiment workflow."""
    
    def test_config_loading(self, tmp_path):
        """Test loading experiment configuration."""
        config_path = _write_config(tmp_path, _tiny_config(str(tmp_path / "results")))

        run_dir = run_training(config_path, seed_override=321)
        resolved_config = _read_json(run_dir / "resolved_config.json")

        assert resolved_config["training"]["seed"] == 321
    
    def test_reproducibility(self, tmp_path):
        """Test that results are reproducible with same seed."""
        config_path = _write_config(tmp_path, _tiny_config(str(tmp_path / "results")))

        first_run = run_training(config_path)
        second_run = run_training(config_path)

        first_metrics = _read_json(first_run / "metrics.json")
        second_metrics = _read_json(second_run / "metrics.json")

        assert first_metrics["train_metrics"] == second_metrics["train_metrics"]
        assert first_metrics["val_metrics"] == second_metrics["val_metrics"]
        assert first_metrics["test_metrics"] == second_metrics["test_metrics"]
    
    def test_multi_seed_experiment(self, tmp_path):
        """Different seeds produce different synthetic training outcomes."""
        config_path = _write_config(tmp_path, _tiny_config(str(tmp_path / "results")))
        first = run_training(config_path, seed_override=1)
        second = run_training(config_path, seed_override=999)
        first_info = _read_json(first / "run_info.json")
        second_info = _read_json(second / "run_info.json")
        assert first_info["seed"] != second_info["seed"]


# Placeholder for future integration tests
# TODO: Add tests for evaluation workflow
# TODO: Add tests for experiment comparison
# TODO: Add tests for full R&D workflow

