"""
Base trainer class for Raft-LM.

All specialized trainers should inherit from this base class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json


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
        self.train_metrics = []
        self.val_metrics = []
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
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
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
        avg_loss = epoch_loss / num_batches
        return {'train_loss': avg_loss}
    
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
        
        avg_loss = val_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[str] = None
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
            train_metrics = self.train_epoch(train_loader)
            self.train_metrics.append(train_metrics)
            
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
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def save_metrics(self, metrics_path: str) -> None:
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

