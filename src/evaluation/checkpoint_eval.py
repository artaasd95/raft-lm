"""Checkpoint evaluation helpers shared by MLP backend and report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.data.dataloaders import create_train_val_test_loaders
from src.data.datasets import SyntheticRiskDataset
from src.data.pipeline.cards import EngineLabelRow
from src.data.pipeline.config import load_pipeline_config
from src.data.pipeline.dataset import EngineLabelDataset
from src.data.pipeline.pipeline import load_engine_label_splits, run_pipeline
from src.metrics.risk_metrics import compute_cvar, constraint_violation_rate
from src.metrics.task_metrics import accuracy, f1_score, mae, mse
from src.search.errors import MissingLabelError
from src.search.orchestrator import apply_guidance_to_engine_rows
from src.training.per_sample_loss import per_sample_loss

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_dataloaders(
    config: Dict[str, Any],
    data_config_path: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    if data_config_path is not None:
        return build_dataloaders_from_platform(config, data_config_path)

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


def build_dataloaders_from_platform(
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

    guidance_block = dict(config.get("training", {}).get("unlabeled_guidance") or {})
    guidance_block.setdefault("num_classes", num_classes)
    if pipeline_config.label.unlabeled_guidance:
        guidance_block = {
            **pipeline_config.label.unlabeled_guidance,
            **guidance_block,
        }
    if pipeline_config.label.policy == "guidance":
        guidance_block["enabled"] = True

    train_rows = _ensure_row_labels(
        train_rows,
        guidance_block=guidance_block,
        num_classes=num_classes,
    )
    val_rows = _ensure_row_labels(
        val_rows,
        guidance_block=guidance_block,
        num_classes=num_classes,
    )
    test_rows = _ensure_row_labels(
        test_rows,
        guidance_block=guidance_block,
        num_classes=num_classes,
    )

    if not val_rows:
        raise ValueError(f"No validation rows in {processed_dir}")
    if not test_rows:
        raise ValueError(f"No test rows in {processed_dir}")

    train_dataset = EngineLabelDataset(train_rows, metadata={"source": "data_platform"})
    val_dataset = EngineLabelDataset(val_rows, metadata={"split": "val"})
    test_dataset = EngineLabelDataset(test_rows, metadata={"split": "test"})

    data_config = config["data"]
    return create_train_val_test_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )


def evaluate_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Any,
    device: torch.device,
    metric_names: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model.eval()
    outputs_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    losses: List[torch.Tensor] = []
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
            losses.append(per_sample_loss(outputs, targets).detach().cpu())

    outputs_all = torch.cat(outputs_list)
    targets_all = torch.cat(targets_list)
    losses_all = torch.cat(losses)

    if num_batches == 0:
        raise ValueError("Cannot evaluate model: data loader is empty")

    loss_cfg = (config or {}).get("training", {}).get("loss", {})
    eval_alpha = float(loss_cfg.get("alpha", 0.95))
    violation_threshold = float(loss_cfg.get("violation_threshold", 1.0))

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
            metrics["cvar"] = compute_cvar(losses_all, alpha=eval_alpha)
        elif metric_name == "constraint_violation_rate":
            metrics["constraint_violation_rate"] = constraint_violation_rate(
                losses_all,
                threshold=violation_threshold,
            )
        elif metric_name == "perplexity":
            metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["test_loss"])).item())
        elif metric_name == "tail_error_rate":
            threshold = (
                float(torch.quantile(losses_all, 0.9).item())
                if losses_all.numel() > 1
                else float(metrics["test_loss"])
            )
            metrics["tail_error_rate"] = float((losses_all > threshold).float().mean().item())

    return metrics


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / candidate


def _rows_missing_labels_from_engine(rows: Sequence[EngineLabelRow]) -> List[str]:
    return [row.record_id for row in rows if row.label is None]


def _ensure_row_labels(
    rows: List[EngineLabelRow],
    *,
    guidance_block: Dict[str, Any],
    num_classes: int,
) -> List[EngineLabelRow]:
    missing = _rows_missing_labels_from_engine(rows)
    if missing and not guidance_block.get("enabled", False):
        raise MissingLabelError(missing, hint="training.unlabeled_guidance.enabled")

    dict_rows = apply_guidance_to_engine_rows(
        rows,
        guidance_config=guidance_block,
        num_classes=num_classes,
        hint="training.unlabeled_guidance.enabled",
    )
    return [EngineLabelRow.from_dict(row) for row in dict_rows]


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
