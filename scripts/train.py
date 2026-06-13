"""
Training script for Raft-LM experiments.

Usage:
    python scripts/train.py --config experiments/configs/my_experiment.json
    python scripts/train.py --config configs/training/unsloth_lora_example.yaml
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.logging.experiment_logger import create_experiment_logger
from src.llm_integration.checkpoint_export import CheckpointExporter
from src.llm_integration.factory import create_llm_provider_for_name
from src.training.backends.factory import get_training_backend
from src.training.policies.registry import get_policy_registry
from src.utils.config import load_config, resolve_config, save_config, validate_config
from src.utils.reproducibility import get_device, get_git_commit_hash, set_seed

POLICIES_DIR = REPO_ROOT / "experiments/configs/policies"


def run_training(
    config_path: str,
    seed_override: Optional[int] = None,
    data_config_path: Optional[str] = None,
    policy_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    loss_override: Optional[str] = None,
    backend_override: Optional[str] = None,
    llm_provider: Optional[str] = None,
    export_for_rada: bool = False,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
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
    resolved_checkpoint: Optional[Path] = None
    if checkpoint_path is not None:
        resolved_checkpoint = _resolve_path(checkpoint_path)
        config["training"]["resume_from_checkpoint"] = str(resolved_checkpoint)
    if loss_override is not None:
        config["training"].setdefault("loss", {})["type"] = loss_override
    if backend_override is not None:
        config["training"]["backend"] = backend_override
    if llm_provider is not None:
        # Validate provider alias/config early so runs fail fast.
        create_llm_provider_for_name(llm_provider)
        config["runtime_llm_provider"] = llm_provider
    if epochs_override is not None:
        config["training"]["num_epochs"] = epochs_override
    if batch_size_override is not None:
        config.setdefault("data", {})["batch_size"] = batch_size_override
    validate_config(config)

    seed = config["training"]["seed"]
    set_seed(seed)
    requested_device = config["training"]["device"]
    device = get_device(None if requested_device == "auto" else requested_device)
    run_dir = _create_run_dir(config, checkpoint_path=resolved_checkpoint)
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
            "backend": config["training"].get("backend", "mlp"),
        }
    )

    backend_name = config["training"].get("backend", "mlp")
    backend = get_training_backend(backend_name)
    test_metrics = backend.run(
        config=config,
        run_dir=run_dir,
        data_config_path=data_config_path,
        exp_logger=exp_logger,
    )

    _update_metrics_file(run_dir / "metrics.json", test_metrics)
    exp_logger.log_metrics(
        {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
    )
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

    if export_for_rada:
        best_checkpoint = run_dir / "checkpoints" / "best_model.pt"
        if best_checkpoint.exists():
            exporter = CheckpointExporter(run_dir / "exports" / "rada")
            result = exporter.export_for_rada(
                best_checkpoint,
                adapter_config={
                    "backend": config["training"].get("backend", "mlp"),
                    "weights_key": "model_state_dict",
                },
                model_id=config.get("model", {}).get("type"),
            )
            _write_json(
                run_dir / "export_for_rada.json",
                {
                    "export_dir": str(result.export_dir),
                    "manifest_path": str(result.manifest_path),
                    "checkpoint_path": str(result.checkpoint_path),
                },
            )

    return run_dir


def main():
    """Load configuration, set up model and data, run training."""
    parser = argparse.ArgumentParser(description="Train a Raft-LM model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file (JSON or YAML)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Path to data-platform pipeline YAML (configs/data/*.yaml)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Policy bundle id (experiments/configs/policies/<id>.yaml|json)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint path",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ce", "cvar_penalized", "tail_aware"],
        default=None,
        help="Override training loss (ce, cvar_penalized, tail_aware)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mlp", "unsloth", "ddp", "fsdp"],
        default=None,
        help="Override training backend (mlp, unsloth, ddp, fsdp)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="Optional provider alias or config path to validate runtime LLM wiring.",
    )
    parser.add_argument(
        "--export-for-rada",
        action="store_true",
        help="Export best checkpoint into RADA-compatible handoff format.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs for quick smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override data.batch_size for smoke tests.",
    )

    args = parser.parse_args()

    run_dir = run_training(
        args.config,
        seed_override=args.seed,
        data_config_path=args.data_config,
        policy_id=args.policy,
        checkpoint_path=args.checkpoint,
        loss_override=args.loss,
        backend_override=args.backend,
        llm_provider=args.llm_provider,
        export_for_rada=args.export_for_rada,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
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


def _infer_run_dir_from_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    """Return the experiment run directory that owns a checkpoint path."""
    resolved = checkpoint_path.resolve()
    if resolved.name == "checkpoints" and resolved.is_dir():
        return resolved.parent
    if "checkpoints" in resolved.parts:
        idx = resolved.parts.index("checkpoints")
        return Path(*resolved.parts[:idx])
    if resolved.is_file():
        parent = resolved.parent
        if parent.name == "checkpoints":
            return parent.parent
        return parent
    return None


def _create_run_dir(
    config: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
) -> Path:
    if checkpoint_path is not None:
        inferred = _infer_run_dir_from_checkpoint(checkpoint_path)
        if inferred is not None and inferred.exists():
            inferred.mkdir(parents=True, exist_ok=True)
            return inferred

    runpod_run_dir = os.environ.get("RUNPOD_RUN_DIR")
    if runpod_run_dir:
        run_dir = Path(runpod_run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

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
        "git_commit": get_git_commit_hash(str(REPO_ROOT)),
        "backend": config["training"].get("backend", "mlp"),
        "device": str(device),
        "started_at": started_at.isoformat(),
        "timestamp": started_at.isoformat(),
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


if __name__ == "__main__":
    main()
