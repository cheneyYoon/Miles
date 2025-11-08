"""
Dataset adapter for HuggingFace YouTube Shorts dataset.
Maps actual dataset columns to our expected format.

Based on docs/dataset_column_mapping.md
"""

import logging
import pandas as pd
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Column mapping from actual dataset to our expected format
COLUMN_MAPPING = {
    'row_id': 'video_id',
    'publish_date_approx': 'upload_date',
    'hashtag': 'primary_hashtag',
}


def adapt_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt HuggingFace dataset columns to our expected format.

    Args:
        df: Raw dataset from HuggingFace

    Returns:
        DataFrame with renamed columns
    """
    logger.info("Adapting dataset columns to expected format...")

    df_adapted = df.copy()

    # Rename columns
    df_adapted = df_adapted.rename(columns=COLUMN_MAPPING)

    logger.info(f"Renamed columns: {list(COLUMN_MAPPING.keys())}")

    return df_adapted


def get_available_scalar_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of available scalar features from the dataset.

    Args:
        df: Dataset DataFrame

    Returns:
        List of scalar feature column names
    """
    # Core engagement metrics
    core_features = ['views', 'likes', 'comments', 'shares', 'saves']

    # Pre-calculated rates and metrics
    rate_features = [
        'engagement_rate',
        'completion_rate',
        'like_rate',
        'comment_ratio',
        'share_rate',
        'save_rate',
    ]

    # Temporal features
    temporal_features = [
        'upload_hour',
        'publish_dayofweek',
        'is_weekend',
    ]

    # Content features
    content_features = [
        'duration_sec',
        'title_length',
        'has_emoji',
    ]

    # Creator features
    creator_features = [
        'creator_avg_views',
    ]

    # Combine all potential features
    all_potential = (
        core_features +
        rate_features +
        temporal_features +
        content_features +
        creator_features
    )

    # Filter to only features that exist in the dataframe
    available = [col for col in all_potential if col in df.columns]

    logger.info(f"Found {len(available)} scalar features in dataset")

    return available


def validate_required_columns(df: pd.DataFrame) -> bool:
    """
    Validate that all required columns are present.

    Args:
        df: Dataset DataFrame

    Returns:
        True if all required columns present, raises ValueError otherwise
    """
    required_columns = [
        'video_id',  # After renaming from row_id
        'title',
        'views',
        'likes',
        'upload_date',  # After renaming from publish_date_approx
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info("✅ All required columns present")
    return True


def check_engagement_velocity(df: pd.DataFrame) -> bool:
    """
    Check if engagement_velocity is already calculated in the dataset.

    Args:
        df: Dataset DataFrame

    Returns:
        True if engagement_velocity exists and is valid
    """
    if 'engagement_velocity' not in df.columns:
        logger.warning("engagement_velocity column not found in dataset")
        return False

    # Check if it has valid values
    null_count = df['engagement_velocity'].isnull().sum()
    if null_count > 0:
        logger.warning(
            f"engagement_velocity has {null_count} null values "
            f"({100 * null_count / len(df):.1f}%)"
        )

    # Check if values are reasonable
    velocity_stats = df['engagement_velocity'].describe()
    logger.info(f"engagement_velocity statistics:")
    logger.info(f"  Mean: {velocity_stats['mean']:.2f}")
    logger.info(f"  Std: {velocity_stats['std']:.2f}")
    logger.info(f"  Min: {velocity_stats['min']:.2f}")
    logger.info(f"  Max: {velocity_stats['max']:.2f}")

    return True


def prepare_dataset_for_training(
    df: pd.DataFrame,
    text_column: str = 'title',
    create_viral_labels: bool = True,
    viral_threshold_percentile: float = 80.0
) -> pd.DataFrame:
    """
    Complete preparation of dataset for training.

    This function:
    1. Adapts column names
    2. Validates required columns
    3. Checks engagement_velocity
    4. Creates viral labels if needed
    5. Returns list of available scalar features

    Args:
        df: Raw dataset from HuggingFace
        text_column: Column to use for text features
        create_viral_labels: Whether to create is_viral column
        viral_threshold_percentile: Percentile for viral classification

    Returns:
        Prepared DataFrame ready for model training
    """
    logger.info("="*70)
    logger.info("Preparing dataset for training")
    logger.info("="*70)

    # Step 1: Adapt column names
    df_prepared = adapt_dataset_columns(df)

    # Step 2: Validate required columns
    validate_required_columns(df_prepared)

    # Step 3: Check engagement_velocity
    has_velocity = check_engagement_velocity(df_prepared)

    if not has_velocity:
        logger.warning("Will need to calculate engagement_velocity during feature engineering")

    # Step 4: Create viral labels if needed
    if create_viral_labels and 'is_viral' not in df_prepared.columns:
        if 'engagement_velocity' in df_prepared.columns:
            threshold = df_prepared['engagement_velocity'].quantile(viral_threshold_percentile / 100.0)
            df_prepared['is_viral'] = (df_prepared['engagement_velocity'] >= threshold).astype(int)

            viral_count = df_prepared['is_viral'].sum()
            total_count = len(df_prepared)
            logger.info(f"Created viral labels: {viral_count}/{total_count} viral ({100*viral_count/total_count:.1f}%)")
            logger.info(f"Viral threshold (engagement_velocity): {threshold:.2f}")
        else:
            logger.warning("Cannot create viral labels without engagement_velocity")

    # Step 5: Get available scalar features
    scalar_features = get_available_scalar_features(df_prepared)
    logger.info(f"Available scalar features ({len(scalar_features)}):")
    for feat in scalar_features:
        logger.info(f"  - {feat}")

    logger.info("="*70)
    logger.info("Dataset preparation complete!")
    logger.info(f"Shape: {df_prepared.shape}")
    logger.info(f"Text column: {text_column}")
    logger.info(f"Has viral labels: {'is_viral' in df_prepared.columns}")
    logger.info(f"Has engagement_velocity: {'engagement_velocity' in df_prepared.columns}")
    logger.info("="*70)

    return df_prepared


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive summary of the dataset.

    Args:
        df: Dataset DataFrame

    Returns:
        Dictionary with dataset statistics
    """
    summary = {
        'total_videos': len(df),
        'columns': list(df.columns),
        'platforms': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
        'languages': df['language'].value_counts().to_dict() if 'language' in df.columns else {},
        'genres': df['genre'].value_counts().to_dict() if 'genre' in df.columns else {},
    }

    if 'is_viral' in df.columns:
        summary['viral_count'] = int(df['is_viral'].sum())
        summary['viral_percentage'] = float(100 * df['is_viral'].mean())

    if 'engagement_velocity' in df.columns:
        summary['engagement_velocity_stats'] = df['engagement_velocity'].describe().to_dict()

    return summary


if __name__ == "__main__":
    # Test with dummy data
    logger.info("Testing dataset adapter...")

    # Create dummy data with actual column names
    df_test = pd.DataFrame({
        'row_id': ['vid1', 'vid2', 'vid3'],
        'title': ['Video 1', 'Video 2', 'Video 3'],
        'views': [1000, 5000, 10000],
        'likes': [100, 500, 1000],
        'comments': [10, 50, 100],
        'shares': [5, 25, 50],
        'saves': [2, 10, 20],
        'publish_date_approx': pd.date_range('2025-01-01', periods=3),
        'engagement_velocity': [100, 500, 1000],
        'upload_hour': [10, 14, 18],
        'publish_dayofweek': [0, 2, 4],
        'is_weekend': [0, 0, 0],
        'duration_sec': [30, 45, 60],
        'engagement_rate': [0.15, 0.12, 0.11],
        'completion_rate': [0.8, 0.85, 0.9],
        'creator_avg_views': [5000, 5000, 5000],
        'platform': ['YouTube', 'TikTok', 'YouTube'],
        'language': ['en', 'en', 'en'],
        'genre': ['Comedy', 'Education', 'Gaming'],
    })

    # Test preparation
    df_prepared = prepare_dataset_for_training(df_test)

    print("\nPrepared DataFrame columns:")
    print(df_prepared.columns.tolist())

    print("\nDataset summary:")
    summary = get_dataset_summary(df_prepared)
    import json
    print(json.dumps(summary, indent=2, default=str))

    logger.info("\n✅ All tests passed!")
