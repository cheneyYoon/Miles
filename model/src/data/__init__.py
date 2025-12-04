"""
Data module for the Miles project.
Handles dataset downloading, preprocessing, feature engineering, and PyTorch Dataset classes.
"""

from .download import download_dataset, load_cached_dataset
from .preprocessing import TextPreprocessor, preprocess_dataset
from .feature_engineering import EngagementFeatureEngineer, create_viral_labels
from .dataset import ViralShortsDataset, create_data_loaders
from .dataset_adapter import (
    adapt_dataset_columns,
    prepare_dataset_for_training,
    get_available_scalar_features,
    get_dataset_summary
)

__all__ = [
    'download_dataset',
    'load_cached_dataset',
    'TextPreprocessor',
    'preprocess_dataset',
    'EngagementFeatureEngineer',
    'create_viral_labels',
    'ViralShortsDataset',
    'create_data_loaders',
    'adapt_dataset_columns',
    'prepare_dataset_for_training',
    'get_available_scalar_features',
    'get_dataset_summary',
]
