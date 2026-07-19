"""GPU training tests — not run in default CI (manual validation only)."""

from pathlib import Path

import pytest

from src.application.train import run_training_orchestrated

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.gpu
def test_gpu_peft_one_step(tmp_path):
    config_path = REPO_ROOT / "configs/methods/sft_lora.yaml"
    run_dir, metrics = run_training_orchestrated(
        config_path=str(config_path),
        run_dir=tmp_path / "peft",
    )
    assert run_dir.exists()
    assert metrics.get("status") in {"adapter_saved", "trained", "hf_not_installed"}


@pytest.mark.gpu
def test_gpu_dpo_one_step(tmp_path):
    run_dir, metrics = run_training_orchestrated(
        config_path=str(REPO_ROOT / "configs/methods/dpo_risk.yaml"),
        run_dir=tmp_path / "dpo",
    )
    assert "dpo_loss" in metrics


@pytest.mark.gpu
def test_gpu_ppo_lm_one_step(tmp_path):
    run_dir, metrics = run_training_orchestrated(
        config_path=str(REPO_ROOT / "configs/methods/ppo_lm.yaml"),
        run_dir=tmp_path / "ppo_lm",
    )
    assert "ppo_lm_loss" in metrics
