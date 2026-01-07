"""
Dataset classes for Raft-LM.

Base dataset implementations for risk-aware learning tasks.
Supports:
- Classification tasks (risk level classification)
- Regression tasks (CVaR prediction)
- Decision tasks (risk-aware choice)
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple


class BaseRiskDataset(Dataset):
    """
    Base dataset class for risk-aware learning tasks.
    
    All custom datasets should inherit from this class.
    
    Attributes:
        data: Input features
        labels: Target labels/values
        metadata: Additional information (e.g., scenario parameters)
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Input features tensor
            labels: Target labels/values tensor
            metadata: Optional metadata dictionary
        """
        self.data = data
        self.labels = labels
        self.metadata = metadata or {}
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        return self.data[idx], self.labels[idx]


class SyntheticRiskDataset(BaseRiskDataset):
    """
    Dataset for synthetic risk scenarios.
    
    Used for initial testing with known ground truth.
    Supports heavy-tailed distributions and stress scenarios.
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        tail_index: Optional[float] = None,
        scenario_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize synthetic risk dataset.
        
        Args:
            data: Input features
            labels: Target labels/values
            tail_index: Tail index (alpha) for heavy-tailed distributions
            scenario_params: Parameters used to generate scenarios
        """
        metadata = {
            'tail_index': tail_index,
            'scenario_params': scenario_params or {}
        }
        super().__init__(data, labels, metadata)


# Placeholder for future dataset classes
# TODO: Add RealFinancialDataset
# TODO: Add PositionDataset for trading scenarios
# TODO: Add PreferenceDataset for DPO training

