"""
SurgicalAI configuration module.

This module handles configuration loading and management for the SurgicalAI system.
"""

import os
import yaml
from pathlib import Path

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict with configuration data
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Default configuration path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
