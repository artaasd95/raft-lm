"""MLP baseline training backend (existing train.py path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.data.dataloaders import create_train_val_test_loaders
from src.data.datasets import SyntheticRiskDataset
from src.data_platform.config import load_pipeline_config
from src.data_platform.dataset import EngineLabelDataset
from src.data_platform.pipeline import load_engine_label_splits, run_pipeline
from src.losses.base_losses import CrossEntropyLoss, MSELoss
from src.training.loss_factory import build_loss
from src.metrics.risk_metrics import compute_cvar, constraint_violation_rate
from src.metrics.task_metrics import accuracy, f1_score, mae, mse
from src.models.base_models import SimpleMLP
from src.models.loaders.unified import UnifiedModelLoader
from src.training.backends.base import TrainingBackend
from src.training.base_trainer import BaseTrainer
from src.training.callbacks import build_callbacks

REPO_ROOT = Path(__file__).resolve().parents[3]


class MLPBackend(TrainingBackend):
    """Wraps SimpleMLP + BaseTrainer (unchanged MLP behavior)."""

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        from src.utils.reproducibility import get_device

        device = get_device(
            None if config["training"]["device"] == "auto" else config["training"]["device"]
        )
        train_loader, val_loader, test_loader = _build_dataloaders(
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
        return _evaluate_model(
            model=trainer.model,
            criterion=criterion,
            data_loader=test_loader,
            device=device,
            metric_names=config["evaluation"]["metrics"],
        )


def _build_dataloaders(
    config: Dict[str, Any],
    data_config_path: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    if data_config_path is not None:
        return _build_dataloaders_from_platform(config, data_config_path)

    data_config = config["data"]
    model_config = config["model"]
    seed = config["training"]["seed"]

    train_dataset = _build_synthetic_dataset(
        size=data_config["train_size"],
        input_dim=model_config["input_dim"],
        output_dim=model_config["output_dim"],
        seed=seed,
        split="train",
        data_config=data_config,
    )
    val_dataset = _build_synthetic_dataset(
        size=data_config["val_size"],
        input_dim=model_config["input_dim"],
        output_dim=model_config["output_dim"],
        seed=seed + 1,
        split="val",
        data_config=data_config,
    )
    test_dataset = _build_synthetic_dataset(
        size=data_config["test_size"],
        input_dim=model_config["input_dim"],
        output_dim=model_config["output_dim"],
        seed=seed + 2,
        split="test",
        data_config=data_config,
    )

    return create_train_val_test_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / candidate


def _build_dataloaders_from_platform(
    config: Dict[str, Any],
    data_config_path: str,
) -> Tuple[Any, Any, Any]:
    pipeline_path = _resolve_path(data_config_path)
    pipeline_config = load_pipeline_config(pipeline_path)
    processed_dir = pipeline_config.resolved_output_dir(REPO_ROOT)
    if not (processed_dir / "train.jsonl").exists():
        run_pipeline(pipeline_config, REPO_ROOT)

    train_rows, val_rows, test_rows = load_engine_label_splits(processed_dir)
    if not train_rows:
        raise ValueError(f"No training rows in {processed_dir}")

    feature_dim = pipeline_config.label.feature_dim
    num_classes = pipeline_config.label.num_classes
    config["model"]["input_dim"] = feature_dim
    config["model"]["output_dim"] = num_classes

    train_dataset = EngineLabelDataset(train_rows, metadata={"source": "data_platform"})
    val_dataset = EngineLabelDataset(val_rows or train_rows[:1], metadata={"split": "val"})
    test_dataset = EngineLabelDataset(test_rows or train_rows[:1], metadata={"split": "test"})

    data_config = config["data"]
    return create_train_val_test_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )


def _build_synthetic_dataset(
    size: int,
    input_dim: int,
    output_dim: int,
    seed: int,
    split: str,
    data_config: Dict[str, Any],
) -> SyntheticRiskDataset:
    generator = torch.Generator().manual_seed(seed)
    data = torch.randn(size, input_dim, generator=generator)
    weights = torch.linspace(1.0, -1.0, steps=input_dim)
    noise = 0.1 * torch.randn(size, generator=generator)
    scores = data @ weights + noise
    quantiles = torch.linspace(0.0, 1.0, steps=output_dim + 1)[1:-1]
    thresholds = torch.quantile(scores, quantiles)
    labels = torch.bucketize(scores, thresholds).long()

    scenario_params = dict(data_config.get("scenario_params", {}))
    scenario_params.update({"split": split, "seed": seed})

    return SyntheticRiskDataset(
        data=data,
        labels=labels,
        tail_index=data_config.get("tail_index"),
        scenario_params=scenario_params,
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


def _evaluate_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Any,
    device: torch.device,
    metric_names: List[str],
) -> Dict[str, Any]:
    model.eval()
    outputs_list = []
    targets_list = []
    losses = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1
            outputs_list.append(outputs.detach().cpu())
            targets_list.append(targets.detach().cpu())
            losses.append(_per_sample_loss(outputs, targets, criterion).detach().cpu())

    outputs_all = torch.cat(outputs_list)
    targets_all = torch.cat(targets_list)
    losses_all = torch.cat(losses)

    metrics: Dict[str, Any] = {
        "test_loss": total_loss / num_batches,
        "num_samples": int(targets_all.numel()),
    }
    for metric_name in metric_names:
        if metric_name == "accuracy":
            metrics["accuracy"] = accuracy(outputs_all, targets_all)
        elif metric_name == "f1_score":
            metrics["f1_score"] = f1_score(outputs_all, targets_all)
        elif metric_name == "mse":
            metrics["mse"] = mse(outputs_all, targets_all)
        elif metric_name == "mae":
            metrics["mae"] = mae(outputs_all, targets_all)
        elif metric_name == "cvar":
            metrics["cvar"] = compute_cvar(losses_all, alpha=0.95)
        elif metric_name == "constraint_violation_rate":
            metrics["constraint_violation_rate"] = constraint_violation_rate(
                losses_all,
                threshold=1.0,
            )
        elif metric_name == "perplexity":
            metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["test_loss"])).item())

    return metrics


def _per_sample_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    if isinstance(criterion, CrossEntropyLoss):
        return torch.nn.functional.cross_entropy(outputs, targets, reduction="none")
    if isinstance(criterion, MSELoss):
        losses = torch.nn.functional.mse_loss(outputs, targets, reduction="none")
        return losses.reshape(losses.shape[0], -1).mean(dim=1)
    loss = criterion(outputs, targets)
    if loss.dim() == 0:
        return loss.reshape(1)
    return loss.reshape(-1)
