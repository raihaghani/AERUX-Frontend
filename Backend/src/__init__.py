"""
Aerux Final - Intracranial Aneurysm Detection, Localization, and Segmentation
A professional deep learning pipeline for multi-task aneurysm analysis.

Author: Senior Computer Vision Engineer
Version: 1.0.0
Date: 2025-01-08
"""

__version__ = "1.0.0"
__author__ = "Senior Computer Vision Engineer"
__description__ = "Multi-task aneurysm detection, localization, and segmentation"

# Package imports for easy access
from src.config.config import Config, get_default_config
from src.datasets.dataset import AneurysmDataset, create_data_splits, create_dataloaders

__all__ = [
    'Config',
    'get_default_config',
    'AneurysmDataset',
    'create_data_splits',
    'create_dataloaders'
]
