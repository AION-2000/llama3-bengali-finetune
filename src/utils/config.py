# src/utils/config.py

import json
import os
from typing import Dict, Any

def save_config(config: Dict[str, Any], file_path: str):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(config, f, indent=2)

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(file_path, "r") as f:
        return json.load(f)