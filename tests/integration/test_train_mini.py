"""Mini training integration test for CI (under 30s on CPU)."""

import json

import pytest

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

from scripts.train import run_training
from src.utils.config import load_config, resolve_config

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
LOCKED_CONFIG = REPO_ROOT / "configs" / "risk_training_v1_locked.yaml"


@pytest.mark.timeout(30)
def test_mini_train_artifacts(tmp_path):
    assert LOCKED_CONFIG.exists()

    config = resolve_config(load_config(str(LOCKED_CONFIG)))
    config["output"] = {"results_dir": str(tmp_path / "results")}
    config["training"]["num_epochs"] = 2
    config_path = tmp_path / "mini.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_dir = run_training(str(config_path))

    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "run_info.json").exists()
    assert (run_dir / "checkpoints" / "best_model.pt").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "test_metrics" in metrics
    assert "accuracy" in metrics["test_metrics"]
