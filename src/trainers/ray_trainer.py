"""Ray Train wrapper (optional dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.trainers.base import TrainingBackend
from src.trainers.factory import get_training_backend, resolve_backend
from src.utils.reproducibility import set_seed


class RayTrainerBackend(TrainingBackend):
    """Delegates to the resolved local backend; Ray orchestration is optional."""

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        distributed = config.get("distributed") or {}
        strategy = str(distributed.get("strategy", "ray"))
        inner_name = str(distributed.get("inner_backend") or resolve_backend(config))
        if strategy == "ray":
            try:
                import ray  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "RayTrainerBackend requires `pip install -e '.[ray]'`"
                ) from exc
        backend = get_training_backend(inner_name)
        metrics = backend.run(
            config=config,
            run_dir=run_dir,
            data_config_path=data_config_path,
            exp_logger=exp_logger,
        )
        metrics["distributed_strategy"] = strategy
        metrics["inner_backend"] = inner_name
        return metrics
