"""MLP baseline training backend (existing train.py path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.evaluation.checkpoint_eval import (
    build_dataloaders,
    build_dataloaders_from_platform,
    evaluate_model,
)
from src.models.base_models import SimpleMLP
from src.models.loaders.unified import UnifiedModelLoader
from src.trainers.base import TrainingBackend
from src.trainers.base_trainer import BaseTrainer
from src.training.callbacks import build_callbacks
from src.training.loss_factory import build_loss

REPO_ROOT = Path(__file__).resolve().parents[3]

# Backward-compatible aliases for tests
_build_dataloaders = build_dataloaders
_build_dataloaders_from_platform = build_dataloaders_from_platform
_evaluate_model = evaluate_model


class MLPBackend(TrainingBackend):
    """Wraps SimpleMLP + BaseTrainer (unchanged MLP behavior)."""

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        from src.utils.reproducibility import get_device, set_seed

        set_seed(config["training"]["seed"])
        device = get_device(
            None if config["training"]["device"] == "auto" else config["training"]["device"]
        )
        train_loader, val_loader, test_loader = build_dataloaders(
            config, data_config_path=data_config_path
        )
        model = _build_model(config)
        loader = UnifiedModelLoader()
        loaded = loader.load(config, model)
        model = loaded.module
        criterion = build_loss(config)
        optimizer = _build_optimizer(config, model)

        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
        )
        callbacks = build_callbacks(config, exp_logger) if exp_logger is not None else []
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["training"]["num_epochs"],
            save_dir=str(run_dir),
            callbacks=callbacks,
        )
        return evaluate_model(
            model=trainer.model,
            criterion=criterion,
            data_loader=test_loader,
            device=device,
            metric_names=config["evaluation"]["metrics"],
            config=config,
        )


def _build_model(config: Dict[str, Any]) -> torch.nn.Module:
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


def _build_optimizer(
    config: Dict[str, Any],
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    optimizer_config = config["training"]["optimizer"]
    optimizer_type = optimizer_config["type"]
    kwargs = {
        "lr": optimizer_config["lr"],
        "weight_decay": optimizer_config["weight_decay"],
    }
    if optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
