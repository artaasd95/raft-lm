"""Training backend protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingBackend(ABC):
    """Pluggable training engine (MLP baseline or Unsloth LoRA)."""

    @abstractmethod
    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute training and return test metrics."""
