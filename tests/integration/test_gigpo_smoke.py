"""GiGPO training smoke."""

from pathlib import Path

from src.application.train import run_training_orchestrated

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gigpo_smoke(tmp_path):
    run_dir, metrics = run_training_orchestrated(
        config_path=str(REPO_ROOT / "configs/methods/gigpo.yaml"),
        run_dir=tmp_path / "run",
    )
    assert run_dir.exists()
    assert metrics.get("status") == "smoke_complete"
    assert "gigpo_combined_std" in metrics
