"""Training orchestrator with method dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.domain.specs import UNSLOTH_ALLOWED_METHODS, MethodSpec
from src.trainers.factory import get_training_backend, resolve_backend
from src.utils.config import load_config, resolve_config, validate_config


def run_training_orchestrated(
    config_path: str,
    run_dir: Optional[Path] = None,
    **kwargs: Any,
) -> tuple[Path, Dict[str, Any]]:
    """
    Load config, validate method/backend rules, dispatch to backend.

    Returns (run_dir, metrics).
    """
    config = resolve_config(load_config(config_path))
    method = config.get("method", "supervised")
    MethodSpec.from_config(config)
    backend_name = resolve_backend(config)
    config.setdefault("training", {})["backend"] = backend_name

    if backend_name == "unsloth" and method not in UNSLOTH_ALLOWED_METHODS:
        raise ValueError(
            f"Unsloth backend only supports method in {sorted(UNSLOTH_ALLOWED_METHODS)}, got {method!r}. "
            "Use training.backend: peft for DPO/PPO/GRPO/GiGPO."
        )

    validate_config(config)
    backend = get_training_backend(backend_name)
    if run_dir is None:
        run_dir = Path(config.get("output", {}).get("results_dir", "experiments/results"))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = backend.run(config=config, run_dir=run_dir, **kwargs)
    return run_dir, metrics
