"""
PyTorch Dataset class for multimodal viral video prediction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from .preprocessing import TextPreprocessor, ImagePreprocessor
from .feature_engineering import create_viral_labels, create_scalar_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViralShortsDataset(Dataset):
    """
    PyTorch Dataset for multimodal viral video prediction.

    Loads and processes:
    - Text features (title, description) -> BERT tokens
    - Image features (thumbnails) -> ResNet-compatible tensors
    - Scalar features (engagement metrics, temporal features)
    - Labels (viral classification + engagement velocity regression)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_column: str = 'title',
        image_column: Optional[str] = 'thumbnail_path',
        scalar_columns: Optional[List[str]] = None,
        label_column: str = 'is_viral',
        velocity_column: str = 'engagement_velocity',
        image_dir: Optional[Union[str, Path]] = None,
        text_max_length: int = 128,
        image_size: int = 224,
        augment_images: bool = False,
        use_images: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            dataframe: Preprocessed DataFrame with features and labels
            text_column: Column containing text data
            image_column: Column containing image paths/filenames
            scalar_columns: List of scalar feature columns
            label_column: Column with binary viral labels
            velocity_column: Column with engagement velocity (regression target)
            image_dir: Directory containing thumbnail images
            text_max_length: Maximum text sequence length
            image_size: Image size for ResNet (224x224)
            augment_images: Whether to apply data augmentation
            use_images: Whether to load images (set False for text-only mode)
        """
        self.df = dataframe.reset_index(drop=True)
        self.text_column = text_column
        self.image_column = image_column
        self.scalar_columns = scalar_columns
        self.label_column = label_column
        self.velocity_column = velocity_column
        self.image_dir = Path(image_dir) if image_dir else None
        self.use_images = use_images

        # Initialize preprocessors
        self.text_preprocessor = TextPreprocessor()
        self.image_preprocessor = ImagePreprocessor(
            image_size=image_size,
            normalize=True,
            augment=augment_images
        )
        self.text_max_length = text_max_length

        # Auto-detect scalar columns if not provided
        if self.scalar_columns is None:
            self.scalar_columns = self._auto_detect_scalar_columns()

        # Validate dataset
        self._validate_dataset()

        logger.info(f"Initialized dataset with {len(self)} samples")
        logger.info(f"Text column: {self.text_column}")
        logger.info(f"Scalar features: {len(self.scalar_columns)} columns")
        logger.info(f"Using images: {self.use_images}")

    def _auto_detect_scalar_columns(self) -> List[str]:
        """Auto-detect numerical scalar feature columns."""
        scalar_cols = []

        # Look for common feature patterns
        patterns = [
            '_per_hour', 'upload_', 'age_hours', 'duration',
            'day_of_week', 'is_weekend', 'month'
        ]

        for col in self.df.columns:
            # Skip label and ID columns
            if col in [self.label_column, self.velocity_column, 'video_id', self.text_column, self.image_column]:
                continue

            # Check if column matches patterns and is numerical
            if any(pattern in col for pattern in patterns):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    scalar_cols.append(col)

        if not scalar_cols:
            logger.warning("No scalar features auto-detected. Using empty feature vector.")

        return scalar_cols

    def _validate_dataset(self):
        """Validate that required columns exist."""
        # Check text column
        if self.text_column not in self.df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in DataFrame")

        # Check label column
        if self.label_column not in self.df.columns:
            logger.warning(f"Label column '{self.label_column}' not found. Labels will be None.")

        # Check velocity column
        if self.velocity_column not in self.df.columns:
            logger.warning(f"Velocity column '{self.velocity_column}' not found. Velocity will be None.")

        # Check scalar columns
        missing_scalars = [col for col in self.scalar_columns if col not in self.df.columns]
        if missing_scalars:
            logger.warning(f"Missing scalar columns: {missing_scalars}")
            self.scalar_columns = [col for col in self.scalar_columns if col in self.df.columns]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - 'text': Dict with input_ids, attention_mask, token_type_ids [1, max_length]
            - 'image': Tensor [3, H, W] or None
            - 'scalars': Tensor [num_features]
            - 'label': Tensor [1] (binary classification)
            - 'velocity': Tensor [1] (regression target)
        """
        row = self.df.iloc[idx]

        # 1. Process text
        text = row[self.text_column] if pd.notna(row[self.text_column]) else ""
        text_features = self.text_preprocessor.tokenize(
            text,
            max_length=self.text_max_length,
            return_tensors='pt'
        )

        # Remove batch dimension (added by tokenizer)
        text_features = {k: v.squeeze(0) for k, v in text_features.items()}

        # 2. Process image
        image_tensor = None
        if self.use_images and self.image_column and self.image_column in self.df.columns:
            image_path = row[self.image_column]
            if pd.notna(image_path):
                try:
                    # Construct full path if image_dir provided
                    if self.image_dir:
                        full_path = self.image_dir / image_path
                    else:
                        full_path = Path(image_path)

                    # Load and process image
                    if full_path.exists():
                        image = Image.open(full_path).convert('RGB')
                        image_tensor = self.image_preprocessor.preprocess(image)
                    else:
                        logger.warning(f"Image not found: {full_path}")
                        image_tensor = torch.zeros(3, 224, 224)  # Placeholder
                except Exception as e:
                    logger.warning(f"Error loading image at index {idx}: {e}")
                    image_tensor = torch.zeros(3, 224, 224)  # Placeholder

        if image_tensor is None and self.use_images:
            # Create placeholder if images are expected but not available
            image_tensor = torch.zeros(3, 224, 224)

        # 3. Process scalar features
        scalar_values = []
        for col in self.scalar_columns:
            value = row[col] if pd.notna(row[col]) else 0.0
            scalar_values.append(float(value))

        scalar_tensor = torch.tensor(scalar_values, dtype=torch.float32)

        # 4. Get labels
        label = None
        if self.label_column in self.df.columns:
            label = torch.tensor([row[self.label_column]], dtype=torch.long)

        velocity = None
        if self.velocity_column in self.df.columns:
            velocity = torch.tensor([row[self.velocity_column]], dtype=torch.float32)

        # Return sample
        sample = {
            'text': text_features,
            'scalars': scalar_tensor,
        }

        if image_tensor is not None:
            sample['image'] = image_tensor

        if label is not None:
            sample['label'] = label

        if velocity is not None:
            sample['velocity'] = velocity

        return sample


def collate_multimodal_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multimodal batches.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    batched = {}

    # Batch text features
    text_batch = {
        'input_ids': torch.stack([item['text']['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['text']['attention_mask'] for item in batch]),
    }
    if 'token_type_ids' in batch[0]['text']:
        text_batch['token_type_ids'] = torch.stack([item['text']['token_type_ids'] for item in batch])

    batched['text'] = text_batch

    # Batch images if present
    if 'image' in batch[0]:
        batched['image'] = torch.stack([item['image'] for item in batch])

    # Batch scalars
    batched['scalars'] = torch.stack([item['scalars'] for item in batch])

    # Batch labels if present
    if 'label' in batch[0]:
        batched['label'] = torch.stack([item['label'] for item in batch]).squeeze(1)

    if 'velocity' in batch[0]:
        batched['velocity'] = torch.stack([item['velocity'] for item in batch]).squeeze(1)

    return batched


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_column: Optional[str] = 'is_viral',
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.

    Args:
        df: Input DataFrame
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        stratify_column: Column to stratify on (for balanced splits)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    if stratify_column and stratify_column in df.columns:
        # Stratified split
        from sklearn.model_selection import train_test_split

        # First split: train + temp
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            stratify=df[stratify_column],
            random_state=random_seed
        )

        # Second split: val + test
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size_adjusted,
            stratify=temp_df[stratify_column],
            random_state=random_seed
        )
    else:
        # Random split
        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_seed
        )

        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size_adjusted,
            random_state=random_seed
        )

    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for ViralShortsDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ViralShortsDataset(
        train_df,
        augment_images=True,  # Augment training data
        **dataset_kwargs
    )

    val_dataset = ViralShortsDataset(
        val_df,
        augment_images=False,  # No augmentation for validation
        **dataset_kwargs
    )

    test_dataset = ViralShortsDataset(
        test_df,
        augment_images=False,  # No augmentation for test
        **dataset_kwargs
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multimodal_batch,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal_batch,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal_batch,
        pin_memory=True
    )

    logger.info(f"Created DataLoaders with batch_size={batch_size}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    logger.info("Testing ViralShortsDataset...")

    # Create sample data
    sample_df = pd.DataFrame({
        'video_id': [1, 2, 3, 4, 5],
        'title': [
            'Amazing AI breakthrough',
            'Funny cat video',
            'Cooking tutorial',
            'Travel vlog',
            'Tech review'
        ],
        'views_per_hour': [100.0, 500.0, 200.0, 50.0, 1000.0],
        'likes_per_hour': [10.0, 50.0, 20.0, 5.0, 100.0],
        'upload_day_of_week': [0, 1, 2, 3, 4],
        'upload_hour': [10, 14, 18, 8, 20],
        'is_weekend': [0, 0, 0, 0, 1],
        'age_hours': [24.0, 48.0, 12.0, 72.0, 6.0],
        'engagement_velocity': [100.0, 500.0, 200.0, 50.0, 1000.0],
        'is_viral': [0, 1, 0, 0, 1],
    })

    # Create dataset (text-only mode for testing)
    dataset = ViralShortsDataset(
        sample_df,
        text_column='title',
        use_images=False
    )

    # Test __getitem__
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Text input_ids shape: {sample['text']['input_ids'].shape}")
    print(f"  Scalars shape: {sample['scalars'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Velocity: {sample['velocity']}")

    # Test DataLoader
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_multimodal_batch
    )

    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Text input_ids shape: {batch['text']['input_ids'].shape}")
    print(f"  Scalars shape: {batch['scalars'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")

    logger.info("\nAll tests passed!")
