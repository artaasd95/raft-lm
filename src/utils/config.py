"""
Configuration management utilities.

Handles loading, saving, and validating experiment configurations.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration contains required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_fields = ['model', 'data', 'training']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    return True


# Placeholder for future config utilities
# TODO: Add config merging for experiment variations
# TODO: Add config versioning
# TODO: Add config templates for common experiments

