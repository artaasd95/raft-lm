"""Smoke tests for Unsloth training backend wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.trainers.factory import SUPPORTED_BACKENDS, get_training_backend
from src.utils.config import load_config, resolve_config, validate_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs/training/unsloth_lora_example.yaml"


class TestUnslothSmoke:
    def test_example_yaml_loads_and_validates(self):
        config = resolve_config(load_config(str(EXAMPLE_CONFIG)))
        assert config["training"]["backend"] == "unsloth"
        assert config["model"]["type"] == "hf_lora"
        assert config["model"]["model_id"] == "qwen2.5-0.5b"
        assert validate_config(config) is True

    def test_factory_registers_unsloth_backend(self):
        assert "unsloth" in SUPPORTED_BACKENDS
        backend = get_training_backend("unsloth")
        assert backend.__class__.__name__ == "UnslothTrainer"

    def test_factory_default_mlp_unchanged(self):
        backend = get_training_backend("mlp")
        assert backend.__class__.__name__ == "MLPBackend"

    def test_distilled_corpus_stub_exists(self):
        corpus = REPO_ROOT / "data/distilled/risk_sft_v1"
        assert (corpus / "train.jsonl").exists()
        assert (corpus / "manifest.json").exists()
        manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["corpus_id"] == "risk_sft_v1"

    @pytest.mark.gpu
    def test_unsloth_one_epoch_produces_adapter(self, tmp_path):
        unsloth = pytest.importorskip("unsloth", reason="Unsloth not installed")
        del unsloth

        import os

        if not os.environ.get("RAFT_MODELS_ROOT"):
            pytest.skip("RAFT_MODELS_ROOT not set")

        from scripts.train import run_training

        config = resolve_config(load_config(str(EXAMPLE_CONFIG)))
        config["output"]["results_dir"] = str(tmp_path / "results")
        config["output"]["adapters_dir"] = str(tmp_path / "adapters")
        config_path = tmp_path / "unsloth_smoke.yaml"
        import yaml

        config_path.write_text(yaml.dump(config), encoding="utf-8")

        run_dir = run_training(str(config_path))
        metrics_path = run_dir / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "test_metrics" in metrics
        adapter_dir = Path(metrics["test_metrics"].get("adapter_dir", ""))
        assert adapter_dir.is_dir()
        assert (adapter_dir / "training_config.json").exists()
