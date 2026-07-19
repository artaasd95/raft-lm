"""PEFT SFT backend smoke."""

from pathlib import Path

from src.application.train import run_training_orchestrated

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_peft_smoke(tmp_path):
    run_dir, metrics = run_training_orchestrated(
        config_path=str(REPO_ROOT / "configs/methods/sft_lora.yaml"),
        run_dir=tmp_path / "run",
    )
    assert run_dir.exists()
    assert metrics.get("backend") == "peft"
    assert metrics.get("status") in {"adapter_saved", "hf_not_installed", "stub_model", "load_skipped"}
