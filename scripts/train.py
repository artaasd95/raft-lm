"""
Training script for Raft-LM experiments.

Usage:
    python scripts/train.py --config experiments/configs/my_experiment.json
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataloaders import create_train_val_test_loaders
from src.data.datasets import SyntheticRiskDataset
from src.data_platform.config import load_pipeline_config
from src.data_platform.dataset import EngineLabelDataset
from src.data_platform.pipeline import load_engine_label_splits, run_pipeline
from src.losses.base_losses import CrossEntropyLoss, MSELoss
from src.metrics.risk_metrics import compute_cvar, constraint_violation_rate
from src.metrics.task_metrics import accuracy, f1_score, mae, mse
from src.models.base_models import SimpleMLP
from src.logging.experiment_logger import create_experiment_logger
from src.training.base_trainer import BaseTrainer
from src.training.policies.registry import get_policy_registry
from src.utils.config import load_config, resolve_config, save_config, validate_config
from src.utils.reproducibility import get_device, set_seed

POLICIES_DIR = REPO_ROOT / "experiments/configs/policies"


def run_training(
    config_path: str,
    seed_override: Optional[int] = None,
    data_config_path: Optional[str] = None,
    policy_id: Optional[str] = None,
) -> Path:
    """
    Run a complete config-driven training workflow.

    Args:
        config_path: Path to experiment configuration file
        seed_override: Optional seed that overrides the config value

    Returns:
        Path to the experiment run directory
    """
    config_file = _resolve_path(config_path)
    config = resolve_config(load_config(str(config_file)))
    if policy_id is not None:
        registry = get_policy_registry(policies_dir=POLICIES_DIR)
        config = registry.apply_to_config(config, policy_id)
    if seed_override is not None:
        config["training"]["seed"] = seed_override
    validate_config(config)

    seed = config["training"]["seed"]
    set_seed(seed)
    requested_device = config["training"]["device"]
    device = get_device(None if requested_device == "auto" else requested_device)
    run_dir = _create_run_dir(config)
    started_at = datetime.now(timezone.utc)

    save_config(config, str(run_dir / "resolved_config.json"))
    _write_json(
        run_dir / "run_info.json",
        _build_run_info(
            config_file=config_file,
            config=config,
            device=device,
            started_at=started_at,
        ),
    )

    logging_config = config.get("logging", {})
    exp_logger = create_experiment_logger(
        logging_config.get("experiment_backend", "local"),
        run_dir=run_dir,
        experiment_name=config["experiment_name"],
    )
    exp_logger.log_params(
        {
            "seed": config["training"]["seed"],
            "policy_id": config.get("policy_id"),
            "data_config": data_config_path,
        }
    )

    train_loader, val_loader, test_loader = _build_dataloaders(
        config, data_config_path=data_config_path
    )
    model = _build_model(config)
    criterion = _build_loss(config)
    optimizer = _build_optimizer(config, model)

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["training"]["num_epochs"],
        save_dir=str(run_dir),
    )

    test_metrics = _evaluate_model(
        model=trainer.model,
        criterion=criterion,
        data_loader=test_loader,
        device=device,
        metric_names=config["evaluation"]["metrics"],
    )
    _update_metrics_file(run_dir / "metrics.json", test_metrics)
    exp_logger.log_metrics({k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))})
    exp_logger.log_artifact(str(run_dir / "metrics.json"), name="metrics")
    exp_logger.finish()
    _write_json(
        run_dir / "run_info.json",
        _build_run_info(
            config_file=config_file,
            config=config,
            device=device,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        ),
    )

    return run_dir


def main():
    """
    Main training function.
    
    Loads configuration, sets up model and data, runs training.
    """
    parser = argparse.ArgumentParser(description='Train a Raft-LM model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        default=None,
        help='Path to data-platform pipeline YAML (configs/data/*.yaml)',
    )
    parser.add_argument(
        '--policy',
        type=str,
        default=None,
        help='Policy bundle id (experiments/configs/policies/<id>.yaml|json)',
    )

    args = parser.parse_args()

    run_dir = run_training(
        args.config,
        seed_override=args.seed,
        data_config_path=args.data_config,
        policy_id=args.policy,
    )
    print(f"Training complete. Results saved to: {run_dir}")


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / candidate


def _create_run_dir(config: Dict[str, Any]) -> Path:
    results_root = Path(config["output"]["results_dir"])
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = _slugify(config["experiment_name"])
    seed = config["training"]["seed"]
    run_dir = results_root / f"{timestamp}_{experiment_name}_seed{seed}"

    suffix = 1
    while run_dir.exists():
        run_dir = results_root / f"{timestamp}_{experiment_name}_seed{seed}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "experiment"


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


def _build_loss(config: Dict[str, Any]) -> torch.nn.Module:
    loss_type = config["training"]["loss"]["type"]
    if loss_type == "CrossEntropyLoss":
        return CrossEntropyLoss()
    if loss_type == "MSELoss":
        return MSELoss()
    raise ValueError(f"Unsupported loss type: {loss_type}")


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
    return loss.reshape(1)


def _build_run_info(
    config_file: Path,
    config: Dict[str, Any],
    device: torch.device,
    started_at: datetime,
    completed_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    info = {
        "config_path": str(config_file),
        "config_version": config["config_version"],
        "experiment_name": config["experiment_name"],
        "seed": config["training"]["seed"],
        "device": str(device),
        "started_at": started_at.isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if completed_at is not None:
        info["completed_at"] = completed_at.isoformat()
    return info


def _update_metrics_file(metrics_path: Path, test_metrics: Dict[str, Any]) -> None:
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics["test_metrics"] = test_metrics
    _write_json(metrics_path, metrics)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()

