"""
Task performance metrics.

Standard metrics for classification, regression, and decision tasks.
"""

import torch
import numpy as np
from typing import Union


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth class indices
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        MSE value
    """
    return ((predictions - targets) ** 2).mean().item()


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        MAE value
    """
    return (predictions - targets).abs().mean().item()


def f1_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute F1 score for binary classification.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth class indices
        
    Returns:
        F1 score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


# Placeholder for additional task metrics
# TODO: Add multi-class F1 score
# TODO: Add confusion matrix
# TODO: Add ROC-AUC
# TODO: Add calibration error

