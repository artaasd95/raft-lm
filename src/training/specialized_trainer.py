"""Specialized trainer wrappers for probabilistic, quantitative, and tool-aware tasks."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer


class _SpecializedTrainer(BaseTrainer):
    """Base helper that supports dict/dataclass-like batches."""

    def _unpack(self, batch: Any) -> Dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return batch
        if hasattr(batch, "__dict__"):
            return {
                key: value
                for key, value in vars(batch).items()
                if isinstance(value, torch.Tensor)
            }
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")


class ProbabilisticReasoningTrainer(_SpecializedTrainer):
    """Trainer expecting probabilistic targets in each batch."""

    def train_epoch(self, train_loader: DataLoader, *, collect_per_sample_losses: bool = False):
        self.model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            fields = self._unpack(batch)
            features = fields["features"].to(self.device)
            target_distribution = fields["target_distribution"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, target_distribution)
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
            n += 1
        return {"train_loss": total / max(n, 1)}


class QuantitativeReasoningTrainer(_SpecializedTrainer):
    """Trainer expecting quantitative bounds per sample."""

    def train_epoch(self, train_loader: DataLoader, *, collect_per_sample_losses: bool = False):
        self.model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            fields = self._unpack(batch)
            features = fields["features"].to(self.device)
            target_value = fields["target_value"].to(self.device)
            lower_bound = fields["lower_bound"].to(self.device)
            upper_bound = fields["upper_bound"].to(self.device)
            self.optimizer.zero_grad()
            prediction = self.model(features)
            loss = self.criterion(prediction, target_value, lower_bound, upper_bound)
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
            n += 1
        return {"train_loss": total / max(n, 1)}


class ToolAwareReasoningTrainer(_SpecializedTrainer):
    """Trainer expecting tool masks in each batch."""

    def train_epoch(self, train_loader: DataLoader, *, collect_per_sample_losses: bool = False):
        self.model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            fields = self._unpack(batch)
            features = fields["features"].to(self.device)
            labels = fields["label"].to(self.device)
            tool_mask = fields["tool_mask"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels, tool_mask)
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
            n += 1
        return {"train_loss": total / max(n, 1)}
