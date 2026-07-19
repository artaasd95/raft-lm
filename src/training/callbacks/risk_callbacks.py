"""Loss decomposition and engine-label alignment callbacks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from src.logging.experiment_logger import BaseExperimentLogger
from src.metrics.risk_metrics import batch_cvar_from_losses


class LossDecompositionCallback:
    """Log mean loss vs CVaR tail component each epoch."""

    def __init__(self, logger: BaseExperimentLogger, alpha: float = 0.95) -> None:
        self.logger = logger
        self.alpha = alpha

    def on_epoch_losses(self, epoch: int, per_sample_losses: torch.Tensor) -> None:
        losses = per_sample_losses.detach().reshape(-1)
        mean_loss = float(losses.mean().item())
        cvar_loss = float(batch_cvar_from_losses(losses, self.alpha).item())
        self.logger.log_metrics(
            {
                "loss_mean": mean_loss,
                "loss_cvar": cvar_loss,
                "loss_tail_fraction": cvar_loss / max(mean_loss, 1e-9),
            },
            step=epoch,
        )


class EngineLabelAlignmentCallback:
    """Log label prediction mismatch rate when engine_labels present in batch metadata."""

    def __init__(self, logger: BaseExperimentLogger) -> None:
        self.logger = logger

    def on_epoch_end(
        self,
        epoch: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        engine_label_buckets: Optional[List[int]] = None,
    ) -> None:
        if not engine_label_buckets:
            return
        pred = predictions.argmax(dim=-1).cpu().tolist()
        buckets = engine_label_buckets[: len(pred)]
        if not buckets:
            return
        mismatches = sum(int(p != b) for p, b in zip(pred, buckets))
        rate = mismatches / len(buckets)
        self.logger.log_metrics({"engine_label_alignment_error": rate}, step=epoch)


def build_callbacks(
    config: Dict[str, Any],
    logger: BaseExperimentLogger,
) -> List[Any]:
    """Build callbacks when logging.callbacks is explicitly enabled."""
    logging_cfg = config.get("logging", {})
    if not logging_cfg.get("callbacks"):
        return []
    alpha = float(config.get("training", {}).get("loss", {}).get("alpha", 0.95))
    return [
        LossDecompositionCallback(logger, alpha=alpha),
        EngineLabelAlignmentCallback(logger),
    ]
