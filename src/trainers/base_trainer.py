"""
Base trainer class for Raft-LM.

All specialized trainers should inherit from this base class.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.per_sample_loss import per_sample_loss


class BaseTrainer:
    """
    Base trainer class for risk-aware learning.

    Provides the core training loop structure. All specialized trainers
    (e.g., CVaRTrainer, DPOTrainer, PPOTrainer) should inherit from this.

    Attributes:
        model: The neural network model
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device for training (cpu/cuda)
        config: Training configuration dictionary
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize the base trainer.

        Args:
            model: Neural network model
            optimizer: Optimizer for training
            criterion: Loss function
            device: Training device
            config: Training configuration
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Metrics tracking
        self.train_metrics: list[dict[str, float]] = []
        self.val_metrics: list[dict[str, float]] = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        *,
        collect_per_sample_losses: bool = False,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics for this epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        per_sample_chunks: List[torch.Tensor] = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if collect_per_sample_losses:
                per_sample_chunks.append(per_sample_loss(outputs, targets).detach())

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        if num_batches == 0:
            return {"train_loss": 0.0}
        avg_loss = epoch_loss / num_batches
        metrics: Dict[str, Any] = {"train_loss": avg_loss}
        if per_sample_chunks:
            metrics["_per_sample_losses"] = torch.cat(per_sample_chunks)
        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise ValueError("Validation data loader is empty")
        avg_loss = val_loss / num_batches
        return {'val_loss': avg_loss}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints and metrics
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = save_path / 'checkpoints'
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(
                train_loader,
                collect_per_sample_losses=bool(callbacks),
            )
            per_sample = train_metrics.pop("_per_sample_losses", None)
            self.train_metrics.append(train_metrics)
            if callbacks and per_sample is not None:
                for cb in callbacks:
                    if hasattr(cb, "on_epoch_losses"):
                        cb.on_epoch_losses(epoch, per_sample)

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_metrics.append(val_metrics)

            # Log progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_metrics['train_loss']:.4f} - "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Save checkpoint if best
            if save_dir and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(checkpoint_path / 'best_model.pt')

            # Save metrics
            if save_dir:
                self.save_metrics(save_path / 'metrics.json')

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        if 'config' in checkpoint:
            self.config = checkpoint['config']

    def save_metrics(self, metrics_path: str | Path) -> None:
        """
        Save training metrics to JSON file.

        Args:
            metrics_path: Path to save metrics
        """
        metrics = {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)


# Placeholder for specialized trainers
# TODO: Add CVaRTrainer for risk-aware supervised learning
# TODO: Add DPOTrainer for preference learning
# TODO: Add PPOTrainer for constrained RL
# TODO: Add NashTrainer for multi-agent learning

