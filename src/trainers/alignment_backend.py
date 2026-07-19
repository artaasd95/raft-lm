"""Preference alignment backends (DPO / KTO)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.trainers.base import TrainingBackend
from src.trainers.lm_training import (
    is_smoke_mode,
    load_preference_pairs,
    run_dpo_epoch,
    run_kto_epoch,
    training_device,
)
from src.utils.reproducibility import set_seed


class DPOBackend(TrainingBackend):
    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        pairs = load_preference_pairs(config)
        device = training_device(config)
        metrics = run_dpo_epoch(config, pairs, device=device)
        if is_smoke_mode(config):
            metrics["status"] = "smoke_complete"
        else:
            metrics["status"] = "trained"
        _write_metrics(run_dir, metrics, exp_logger)
        return metrics


class KTOBackend(TrainingBackend):
    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        pairs = load_preference_pairs(config)
        metrics = run_kto_epoch(config, pairs)
        if is_smoke_mode(config):
            metrics["status"] = "smoke_complete"
        else:
            metrics["status"] = "trained"
        _write_metrics(run_dir, metrics, exp_logger)
        return metrics


def _write_metrics(
    run_dir: Path,
    metrics: Dict[str, Any],
    exp_logger: Optional[Any] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if exp_logger is not None:
        exp_logger.log_metrics(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
