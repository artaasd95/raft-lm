"""Unified evaluation report for trained checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.evaluation.checkpoint_eval import build_dataloaders, evaluate_model
from src.models.base_models import SimpleMLP
from src.models.loaders.unified import UnifiedModelLoader
from src.training.loss_factory import build_loss
from src.utils.config import load_config, resolve_config
from src.utils.reproducibility import get_device


def build_model_from_config(config: Dict[str, Any]) -> torch.nn.Module:
    model_config = config["model"]
    if model_config["type"] != "SimpleMLP":
        raise ValueError(f"Unsupported model type: {model_config['type']}")
    return SimpleMLP(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        output_dim=model_config["output_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    data_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load checkpoint and produce task + risk metrics report."""
    config = resolve_config(load_config(str(config_path)))
    device = get_device(
        None if config["training"]["device"] == "auto" else config["training"]["device"]
    )
    model = build_model_from_config(config)
    loader = UnifiedModelLoader()
    loaded = loader.load(config, model, checkpoint_path=str(checkpoint_path))
    model = loaded.module.to(device)

    _, _, test_loader = build_dataloaders(config, data_config_path=data_config_path)
    criterion = build_loss(config)
    raw_metrics = evaluate_model(
        model=model,
        criterion=criterion,
        data_loader=test_loader,
        device=device,
        metric_names=config["evaluation"]["metrics"],
        config=config,
    )

    task_metrics = {
        k: raw_metrics[k]
        for k in ("accuracy", "f1_score", "mse", "mae", "test_loss", "num_samples")
        if k in raw_metrics
    }
    risk_metrics: Dict[str, Any] = {
        k: raw_metrics[k]
        for k in ("cvar", "constraint_violation_rate", "perplexity", "tail_error_rate")
        if k in raw_metrics
    }

    return {
        "task_metrics": task_metrics,
        "risk_metrics": risk_metrics,
        "provenance": {
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
            "loader_source": loaded.source,
            "device": str(device),
        },
    }
