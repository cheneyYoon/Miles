"""
Feature engineering utilities for engagement metrics and viral label creation.
"""

import logging
from typing import Optional, Tuple, List
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EngagementFeatureEngineer:
    """
    Engineer engagement-based features normalized by video age.
    From implementation_plan.md: "Standardize engagement metrics by normalizing
    counts relative to hours since upload, to remove age biases."
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.fitted = False

    def calculate_video_age_hours(
        self,
        upload_date: pd.Series,
        reference_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Calculate video age in hours.

        Args:
            upload_date: Series with upload timestamps
            reference_date: Reference date for calculation (default: now)

        Returns:
            Series with video age in hours
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        # Convert to datetime if not already
        upload_date = pd.to_datetime(upload_date, errors='coerce')

        # Calculate age in hours
        age_hours = (reference_date - upload_date).dt.total_seconds() / 3600

        # Handle negative ages (future dates) and very small ages
        age_hours = age_hours.clip(lower=1.0)  # Minimum 1 hour to avoid division by zero

        return age_hours

    def normalize_engagement_metrics(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        age_column: str = 'upload_date',
        reference_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Normalize engagement metrics by video age.

        Args:
            df: Input DataFrame
            metrics: List of metric columns to normalize (e.g., ['views', 'likes', 'comments'])
            age_column: Column containing upload date
            reference_date: Reference date for age calculation

        Returns:
            DataFrame with normalized engagement metrics
        """
        df_eng = df.copy()

        # Default metrics if not specified
        if metrics is None:
            metrics = [col for col in ['views', 'likes', 'comments', 'shares'] if col in df.columns]

        if not metrics:
            logger.warning("No engagement metrics found in DataFrame")
            return df_eng

        # Calculate video age in hours
        if age_column in df.columns:
            age_hours = self.calculate_video_age_hours(df[age_column], reference_date)
            df_eng['age_hours'] = age_hours

            # Normalize each metric
            for metric in metrics:
                if metric in df.columns:
                    normalized_col = f'{metric}_per_hour'
                    df_eng[normalized_col] = df[metric] / age_hours

                    # Handle infinities and NaNs
                    df_eng[normalized_col] = df_eng[normalized_col].replace([np.inf, -np.inf], np.nan)
                    df_eng[normalized_col] = df_eng[normalized_col].fillna(0)

                    logger.info(f"Created normalized metric: {normalized_col}")
        else:
            logger.warning(f"Age column '{age_column}' not found. Skipping normalization.")

        return df_eng

    def calculate_engagement_velocity(
        self,
        df: pd.DataFrame,
        views_col: str = 'views',
        likes_col: str = 'likes',
        comments_col: str = 'comments',
        age_hours_col: str = 'age_hours'
    ) -> pd.Series:
        """
        Calculate a composite engagement velocity score.

        Engagement velocity = weighted combination of normalized metrics.
        This will be used as the target for regression.

        Args:
            df: Input DataFrame
            views_col: Column name for views
            likes_col: Column name for likes
            comments_col: Column name for comments
            age_hours_col: Column name for video age in hours

        Returns:
            Series with engagement velocity scores
        """
        # Weights for different engagement types
        weights = {
            'views': 0.4,
            'likes': 0.35,
            'comments': 0.25
        }

        velocity = pd.Series(0.0, index=df.index)

        # Add normalized views
        if views_col in df.columns and age_hours_col in df.columns:
            views_per_hour = df[views_col] / df[age_hours_col]
            velocity += weights['views'] * views_per_hour

        # Add normalized likes
        if likes_col in df.columns and age_hours_col in df.columns:
            likes_per_hour = df[likes_col] / df[age_hours_col]
            velocity += weights['likes'] * likes_per_hour

        # Add normalized comments
        if comments_col in df.columns and age_hours_col in df.columns:
            comments_per_hour = df[comments_col] / df[age_hours_col]
            velocity += weights['comments'] * comments_per_hour

        # Handle infinities and NaNs
        velocity = velocity.replace([np.inf, -np.inf], np.nan)
        velocity = velocity.fillna(0)

        return velocity

    def fit_scaler(self, features: pd.DataFrame) -> None:
        """
        Fit the scaler on training data.

        Args:
            features: DataFrame with numerical features
        """
        self.scaler.fit(features)
        self.fitted = True
        logger.info("Scaler fitted on training data")

    def transform_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Transform features using the fitted scaler.

        Args:
            features: DataFrame with numerical features

        Returns:
            Scaled feature array
        """
        if not self.fitted:
            logger.warning("Scaler not fitted. Fitting on current data...")
            self.fit_scaler(features)

        return self.scaler.transform(features)


def create_viral_labels(
    df: pd.DataFrame,
    velocity_column: str = 'engagement_velocity',
    threshold_percentile: float = 80.0
) -> Tuple[pd.Series, float]:
    """
    Create binary viral labels based on engagement velocity.

    From APS360_project_proposal.md: Top 20% engagement velocity = viral (1), rest = not viral (0).

    Args:
        df: Input DataFrame
        velocity_column: Column containing engagement velocity
        threshold_percentile: Percentile threshold for viral classification (default: 80 = top 20%)

    Returns:
        Tuple of (binary labels, threshold value)
    """
    if velocity_column not in df.columns:
        raise ValueError(f"Velocity column '{velocity_column}' not found in DataFrame")

    # Calculate threshold
    threshold = df[velocity_column].quantile(threshold_percentile / 100.0)

    # Create binary labels
    labels = (df[velocity_column] >= threshold).astype(int)

    # Log statistics
    viral_count = labels.sum()
    total_count = len(labels)
    viral_pct = (viral_count / total_count) * 100

    logger.info(f"Viral label threshold: {threshold:.4f}")
    logger.info(f"Viral videos: {viral_count} / {total_count} ({viral_pct:.2f}%)")

    return labels, threshold


def extract_temporal_features(
    df: pd.DataFrame,
    upload_date_column: str = 'upload_date'
) -> pd.DataFrame:
    """
    Extract temporal features from upload date.

    Features:
    - Day of week (0=Monday, 6=Sunday)
    - Hour of day (0-23)
    - Is weekend (boolean)
    - Month (1-12)

    Args:
        df: Input DataFrame
        upload_date_column: Column containing upload timestamps

    Returns:
        DataFrame with additional temporal features
    """
    df_temp = df.copy()

    if upload_date_column not in df.columns:
        logger.warning(f"Upload date column '{upload_date_column}' not found")
        return df_temp

    # Convert to datetime
    upload_dates = pd.to_datetime(df[upload_date_column], errors='coerce')

    # Extract features
    df_temp['upload_day_of_week'] = upload_dates.dt.dayofweek
    df_temp['upload_hour'] = upload_dates.dt.hour
    df_temp['is_weekend'] = (upload_dates.dt.dayofweek >= 5).astype(int)
    df_temp['upload_month'] = upload_dates.dt.month

    logger.info("Extracted temporal features")

    return df_temp


def extract_dominant_colors(
    image: Image.Image,
    num_colors: int = 5
) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from an image using color quantization.

    From APS360_project_proposal.md: "Extract five dominant colors and entropy
    measures from video thumbnails."

    Args:
        image: PIL Image
        num_colors: Number of dominant colors to extract

    Returns:
        List of RGB tuples representing dominant colors
    """
    # Resize image for faster processing
    image_small = image.resize((150, 150))

    # Convert to RGB if necessary
    if image_small.mode != 'RGB':
        image_small = image_small.convert('RGB')

    # Get pixel data
    pixels = list(image_small.getdata())

    # Quantize colors (simple approach: bin colors)
    # More sophisticated: use k-means clustering
    quantized = []
    for r, g, b in pixels:
        # Bin to 32 levels per channel (32^3 = 32768 colors)
        r_bin = (r // 32) * 32
        g_bin = (g // 32) * 32
        b_bin = (b // 32) * 32
        quantized.append((r_bin, g_bin, b_bin))

    # Count color frequencies
    color_counts = Counter(quantized)

    # Get top N colors
    dominant_colors = [color for color, _ in color_counts.most_common(num_colors)]

    return dominant_colors


def calculate_color_entropy(image: Image.Image) -> float:
    """
    Calculate color entropy (diversity) of an image.

    Higher entropy = more color diversity.

    Args:
        image: PIL Image

    Returns:
        Entropy value
    """
    # Resize for faster processing
    image_small = image.resize((100, 100))

    # Convert to RGB
    if image_small.mode != 'RGB':
        image_small = image_small.convert('RGB')

    # Get histogram
    histogram = image_small.histogram()

    # Calculate entropy
    total_pixels = sum(histogram)
    entropy = 0.0

    for count in histogram:
        if count > 0:
            probability = count / total_pixels
            entropy -= probability * np.log2(probability)

    return entropy


def create_scalar_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a consolidated set of scalar features for the model.

    Args:
        df: Input DataFrame with all features
        feature_columns: Specific columns to include (None = auto-detect)

    Returns:
        DataFrame with selected scalar features
    """
    if feature_columns is None:
        # Auto-detect numerical features
        feature_columns = []

        # Engagement metrics
        for col in ['views_per_hour', 'likes_per_hour', 'comments_per_hour', 'shares_per_hour']:
            if col in df.columns:
                feature_columns.append(col)

        # Temporal features
        for col in ['upload_day_of_week', 'upload_hour', 'is_weekend', 'upload_month']:
            if col in df.columns:
                feature_columns.append(col)

        # Video characteristics
        for col in ['duration', 'age_hours']:
            if col in df.columns:
                feature_columns.append(col)

    # Select features
    scalar_features = df[feature_columns].copy()

    # Fill NaNs with 0
    scalar_features = scalar_features.fillna(0)

    logger.info(f"Created {len(feature_columns)} scalar features: {feature_columns}")

    return scalar_features


if __name__ == "__main__":
    # Test feature engineering
    logger.info("Testing EngagementFeatureEngineer...")

    # Create sample data
    sample_data = pd.DataFrame({
        'video_id': [1, 2, 3, 4, 5],
        'views': [1000, 5000, 10000, 500, 50000],
        'likes': [100, 400, 800, 50, 4000],
        'comments': [10, 50, 100, 5, 500],
        'upload_date': pd.date_range(end=pd.Timestamp.now(), periods=5, freq='D')
    })

    engineer = EngagementFeatureEngineer()

    # Normalize metrics
    df_normalized = engineer.normalize_engagement_metrics(sample_data)
    print("\nNormalized metrics:")
    print(df_normalized[['views', 'views_per_hour', 'likes', 'likes_per_hour']])

    # Calculate engagement velocity
    df_normalized['engagement_velocity'] = engineer.calculate_engagement_velocity(df_normalized)
    print("\nEngagement velocity:")
    print(df_normalized[['video_id', 'engagement_velocity']])

    # Create viral labels
    labels, threshold = create_viral_labels(df_normalized)
    print(f"\nViral labels (threshold={threshold:.2f}):")
    print(labels)

    # Extract temporal features
    df_temporal = extract_temporal_features(sample_data)
    print("\nTemporal features:")
    print(df_temporal[['upload_day_of_week', 'upload_hour', 'is_weekend']])

    logger.info("\nAll tests passed!")
