"""Distributed backends (DDP/FSDP) that reuse current MLP execution path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.training.backends.base import TrainingBackend
from src.training.backends.mlp_backend import MLPBackend
from src.training.distributed_utils import distributed_session


class _DistributedBackend(TrainingBackend):
    def __init__(self, backend_name: str, process_group_backend: str = "gloo") -> None:
        self._backend_name = backend_name
        self._process_group_backend = process_group_backend
        self._delegate = MLPBackend()

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        with distributed_session(backend=self._process_group_backend):
            if exp_logger is not None:
                exp_logger.log_params({"distributed_backend": self._backend_name})
            return self._delegate.run(
                config=config,
                run_dir=run_dir,
                data_config_path=data_config_path,
                exp_logger=exp_logger,
            )


class DistributedDDPBackend(_DistributedBackend):
    """DDP-compatible backend for torchrun entrypoints."""

    def __init__(self) -> None:
        super().__init__(backend_name="ddp", process_group_backend="gloo")


class DistributedFSDPBackend(_DistributedBackend):
    """FSDP-compatible backend for torchrun entrypoints."""

    def __init__(self) -> None:
        super().__init__(backend_name="fsdp", process_group_backend="gloo")
