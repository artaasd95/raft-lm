"""
Base loss functions for risk-aware learning.

Standard PyTorch losses wrapped for consistency.
"""

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """
    Base loss class for risk-aware learning.
    
    All custom losses should inherit from this class.
    """
    
    def __init__(self):
        """Initialize the base loss."""
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement forward()")


class MSELoss(BaseLoss):
    """Mean Squared Error loss for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse(predictions, targets)


class CrossEntropyLoss(BaseLoss):
    """Cross Entropy loss for classification tasks."""
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(predictions, targets)


# Placeholder for standard loss variants
# TODO: Add weighted losses for imbalanced data
# TODO: Add focal loss for hard examples

