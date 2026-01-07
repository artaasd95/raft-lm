"""
Reproducibility utilities.

Functions to ensure reproducible experiments across runs.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device: Specific device string ('cpu', 'cuda', 'cuda:0', etc.)
                If None, automatically selects cuda if available
        
    Returns:
        PyTorch device object
    """
    if device is not None:
        return torch.device(device)
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Placeholder for future reproducibility utilities
# TODO: Add environment capture (Python version, package versions)
# TODO: Add hardware info logging
# TODO: Add deterministic dataloader configuration

