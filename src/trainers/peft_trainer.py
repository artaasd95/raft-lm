"""Transformers + PEFT SFT backend (non-Unsloth)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.trainers.base import TrainingBackend
from src.trainers.lm_training import run_sft_step, training_device
from src.utils.reproducibility import set_seed


class PeftTrainerBackend(TrainingBackend):
    """SFT via transformers+peft when Unsloth is not selected."""

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        device = training_device(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            metrics = run_sft_step(config, device=device, run_dir=run_dir)
        except Exception as exc:  # pragma: no cover - hub/network failures
            metrics = {
                "backend": "peft",
                "device": device,
                "status": "load_skipped",
                "error": str(exc)[:200],
            }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if exp_logger is not None:
            exp_logger.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
        return metrics
