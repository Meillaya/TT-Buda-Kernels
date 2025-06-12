"""
Configuration utilities for TT-Buda Kernels.

This module provides configuration validation, loading, and saving utilities.
"""

import json
import yaml
from typing import Any, Dict, Union
from pathlib import Path
from dataclasses import asdict, is_dataclass

from .logging import get_logger

logger = get_logger(__name__)


def validate_config(config: Any) -> None:
    """
    Validate a configuration object.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if hasattr(config, 'backend'):
        valid_backends = ["tenstorrent", "cuda", "cpu"]
        if config.backend not in valid_backends:
            raise ValueError(f"Invalid backend '{config.backend}'. Must be one of {valid_backends}")
    
    if hasattr(config, 'optimization_level'):
        valid_levels = ["O0", "O1", "O2", "O3"]
        if config.optimization_level not in valid_levels:
            raise ValueError(f"Invalid optimization level '{config.optimization_level}'. Must be one of {valid_levels}")
    
    if hasattr(config, 'precision'):
        valid_precisions = ["fp32", "fp16", "bf16", "int8"]
        if config.precision not in valid_precisions:
            raise ValueError(f"Invalid precision '{config.precision}'. Must be one of {valid_precisions}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (.json or .yaml)
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        raise


def save_config(config: Union[Dict[str, Any], Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary or dataclass to save
        config_path: Path where to save the configuration (.json or .yaml)
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict if needed
    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config
    
    try:
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {str(e)}")
        raise 