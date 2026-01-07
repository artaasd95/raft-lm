"""
DataLoader utilities for Raft-LM.

Provides convenient dataloader creation with appropriate configurations
for risk-aware learning tasks.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader with standard configuration.
    
    Args:
        dataset: PyTorch Dataset object
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data each epoch
        num_workers: Number of subprocesses for data loading
        pin_memory: Whether to use pinned memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_train_val_test_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> tuple:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Placeholder for specialized dataloaders
# TODO: Add risk-stratified sampling
# TODO: Add tail-event oversampling
# TODO: Add curriculum learning dataloader

