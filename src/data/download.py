"""
Dataset download and caching utilities for HuggingFace datasets.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Downloads and caches the YouTube Shorts & TikTok Trends dataset from HuggingFace.
    """

    def __init__(
        self,
        dataset_name: str = "tarekmasryo/YouTube-Shorts-TikTok-Trends-2025",
        cache_dir: Optional[str] = None,
        output_dir: str = "data/raw"
    ):
        """
        Initialize the dataset downloader.

        Args:
            dataset_name: HuggingFace dataset identifier
            cache_dir: Directory for HuggingFace cache (None = default)
            output_dir: Directory to save processed parquet files
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get HuggingFace token from environment
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            logger.warning(
                "HF_TOKEN not found in environment. "
                "Dataset download may fail if authentication is required."
            )

    def download(
        self,
        split: Optional[str] = None,
        force_redownload: bool = False
    ) -> pd.DataFrame:
        """
        Download the dataset from HuggingFace.

        Args:
            split: Dataset split to download (None = all)
            force_redownload: If True, ignore cached data

        Returns:
            DataFrame containing the dataset
        """
        output_file = self.output_dir / f"youtube_shorts_raw{'_' + split if split else ''}.parquet"

        # Check if already cached
        if output_file.exists() and not force_redownload:
            logger.info(f"Loading cached dataset from {output_file}")
            return pd.read_parquet(output_file)

        logger.info(f"Downloading dataset: {self.dataset_name}")
        try:
            # Download from HuggingFace
            dataset = load_dataset(
                self.dataset_name,
                split=split,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )

            # Convert to pandas DataFrame
            df = dataset.to_pandas()

            logger.info(f"Dataset downloaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            # Save to parquet
            df.to_parquet(output_file, index=False)
            logger.info(f"Dataset saved to {output_file}")

            # Log basic statistics
            self._log_dataset_stats(df)

            return df

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise

    def _log_dataset_stats(self, df: pd.DataFrame) -> None:
        """Log basic dataset statistics."""
        logger.info("="*50)
        logger.info("DATASET STATISTICS")
        logger.info("="*50)
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.info("\nMissing values:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                logger.info(f"  {col}: {count:,} ({pct:.2f}%)")
        else:
            logger.info("\nNo missing values detected")

        # Data types
        logger.info("\nData types:")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col}: {dtype}")

        logger.info("="*50)


def download_dataset(
    dataset_name: str = "tarekmasryo/YouTube-Shorts-TikTok-Trends-2025",
    cache_dir: Optional[str] = None,
    output_dir: str = "data/raw",
    split: Optional[str] = None,
    force_redownload: bool = False
) -> pd.DataFrame:
    """
    Convenience function to download the dataset.

    Args:
        dataset_name: HuggingFace dataset identifier
        cache_dir: Directory for HuggingFace cache
        output_dir: Directory to save processed files
        split: Dataset split to download
        force_redownload: If True, ignore cached data

    Returns:
        DataFrame containing the dataset
    """
    downloader = DatasetDownloader(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    return downloader.download(split=split, force_redownload=force_redownload)


def load_cached_dataset(
    output_dir: str = "data/raw",
    split: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Load a previously cached dataset from disk.

    Args:
        output_dir: Directory where parquet files are saved
        split: Dataset split to load

    Returns:
        DataFrame if cached file exists, None otherwise
    """
    output_dir = Path(output_dir)
    filename = f"youtube_shorts_raw{'_' + split if split else ''}.parquet"
    filepath = output_dir / filename

    if filepath.exists():
        logger.info(f"Loading cached dataset from {filepath}")
        return pd.read_parquet(filepath)
    else:
        logger.warning(f"Cached dataset not found at {filepath}")
        return None


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataset.

    Args:
        df: Dataset DataFrame

    Returns:
        Dictionary with dataset information
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }

    # Add column-specific statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        info["numeric_stats"] = df[numeric_cols].describe().to_dict()

    return info


if __name__ == "__main__":
    # Example usage
    logger.info("Starting dataset download...")

    # Download the full dataset
    df = download_dataset()

    logger.info(f"\nSuccessfully downloaded dataset with {len(df):,} records")
    logger.info(f"First few rows:")
    print(df.head())
