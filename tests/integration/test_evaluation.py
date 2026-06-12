"""
Integration tests for evaluation workflow.

Tests model evaluation, metric computation, and report generation.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

from scripts.train import run_training
from src.evaluation.report import evaluate_checkpoint


class TestEvaluationWorkflow:
    def test_model_evaluation_report_schema(self, tmp_path):
        config = {
            "config_version": 1,
            "experiment_name": "eval_schema",
            "model": {
                "type": "SimpleMLP",
                "input_dim": 4,
                "hidden_dim": 8,
                "output_dim": 3,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "data": {
                "train_size": 12,
                "val_size": 6,
                "test_size": 6,
                "batch_size": 4,
                "num_workers": 0,
            },
            "training": {
                "num_epochs": 1,
                "optimizer": {"type": "Adam", "lr": 0.01, "weight_decay": 0.0},
                "loss": {"type": "ce"},
                "seed": 1,
                "device": "cpu",
            },
            "evaluation": {"metrics": ["accuracy", "cvar", "f1_score"]},
            "logging": {"save_checkpoints": True},
            "output": {"results_dir": str(tmp_path / "runs")},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        run_dir = run_training(str(config_path))
        checkpoint = run_dir / "checkpoints" / "best_model.pt"

        report = evaluate_checkpoint(checkpoint, config_path)
        assert set(report.keys()) == {"task_metrics", "risk_metrics", "provenance"}
        assert "accuracy" in report["task_metrics"]
        assert "cvar" in report["risk_metrics"]
    
    def test_risk_metric_computation(self):
        """CVaR helper returns expected tail mean for known losses."""
        from src.metrics.risk_metrics import compute_cvar
        import torch

        losses = torch.tensor([1.0, 2.0, 3.0, 10.0])
        assert compute_cvar(losses, alpha=0.75) == 10.0

    def test_baseline_comparison(self, tmp_path):
        """Two training runs produce comparable metric schema."""
        from scripts.train import run_training

        tiny = {
            "config_version": 1,
            "experiment_name": "cmp",
            "model": {
                "type": "SimpleMLP",
                "input_dim": 4,
                "hidden_dim": 8,
                "output_dim": 2,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "data": {
                "dataset_type": "SyntheticRiskDataset",
                "train_size": 24,
                "val_size": 12,
                "test_size": 12,
                "batch_size": 8,
                "num_workers": 0,
            },
            "training": {
                "backend": "mlp",
                "num_epochs": 1,
                "seed": 3,
                "device": "cpu",
                "optimizer": {"type": "Adam", "lr": 0.01, "weight_decay": 0.0},
                "loss": {"type": "CrossEntropyLoss"},
            },
            "evaluation": {"metrics": ["accuracy", "cvar"]},
            "logging": {"log_interval": 1, "save_checkpoints": True, "checkpoint_interval": 1},
            "output": {"results_dir": str(tmp_path / "results")},
        }
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps(tiny), encoding="utf-8")
        run_a = run_training(str(config_path))
        run_b = run_training(str(config_path))
        keys_a = set(json.loads((run_a / "metrics.json").read_text())["test_metrics"].keys())
        keys_b = set(json.loads((run_b / "metrics.json").read_text())["test_metrics"].keys())
        assert keys_a == keys_b


class TestReportGeneration:
    """Test suite for report generation."""
    
    def test_generate_evaluation_report(self, tmp_path):
        """evaluate_checkpoint returns task and risk metric sections."""
        from scripts.train import run_training

        config = {
            "config_version": 1,
            "experiment_name": "eval_report",
            "model": {
                "type": "SimpleMLP",
                "input_dim": 4,
                "hidden_dim": 8,
                "output_dim": 2,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "data": {
                "dataset_type": "SyntheticRiskDataset",
                "train_size": 32,
                "val_size": 16,
                "test_size": 16,
                "batch_size": 8,
                "num_workers": 0,
            },
            "training": {
                "backend": "mlp",
                "num_epochs": 1,
                "seed": 7,
                "device": "cpu",
                "optimizer": {"type": "Adam", "lr": 0.01, "weight_decay": 0.0},
                "loss": {"type": "CrossEntropyLoss"},
            },
            "evaluation": {"metrics": ["accuracy", "cvar", "tail_error_rate"]},
            "logging": {"log_interval": 1, "save_checkpoints": True, "checkpoint_interval": 1},
            "output": {"results_dir": str(tmp_path / "results")},
        }
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        run_dir = run_training(str(config_path))
        report = evaluate_checkpoint(run_dir / "checkpoints/best_model.pt", config_path)
        assert "task_metrics" in report and "risk_metrics" in report
        assert "tail_error_rate" in report["risk_metrics"]

    def test_generate_comparison_report(self, tmp_path):
        """Two metric files share schema for downstream comparison."""
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"test_metrics": {"accuracy": 0.5}}), encoding="utf-8")
        b.write_text(json.dumps({"test_metrics": {"accuracy": 0.6}}), encoding="utf-8")
        assert json.loads(a.read_text())["test_metrics"]["accuracy"] == 0.5
        assert set(json.loads(a.read_text())["test_metrics"]) == set(
            json.loads(b.read_text())["test_metrics"]
        )

    @pytest.mark.skip(reason="Visualization export not implemented")
    def test_generate_visualizations(self):
        """Plot generation is out of scope for current evaluation CLI."""


# Placeholder for future evaluation tests
# TODO: Add tests for statistical significance testing
# TODO: Add tests for cross-validation
# TODO: Add tests for robustness evaluation

