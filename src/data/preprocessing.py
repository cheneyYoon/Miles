"""
Data preprocessing utilities for text, images, and general data cleaning.
"""

import re
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import BertTokenizer
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing and tokenization for BERT models.
    Based on implementation_plan.md lines 143-166.
    """

    def __init__(self, model_name: str = 'bert-base-uncased', vocab_size: int = 50000):
        """
        Initialize the text preprocessor with BERT tokenizer.

        Args:
            model_name: BERT model name for tokenizer
            vocab_size: Vocabulary size (unused, kept for compatibility)
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        logger.info(f"Initialized TextPreprocessor with {model_name}")

    def clean_text(self, text: str) -> str:
        """
        Remove URLs, emojis, and special characters from text.

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove special characters but keep hashtags (#) and mentions (@)
        text = re.sub(r'[^\w\s#@]', '', text)

        # Convert to lowercase and strip whitespace
        text = text.lower().strip()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    def tokenize(
        self,
        text: str,
        max_length: int = 128,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        WordPiece tokenization for BERT.

        Args:
            text: Input text string
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length', 'longest', or None)
            truncation: Whether to truncate to max_length
            return_tensors: Return type ('pt' for PyTorch tensors)

        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        # Clean text first
        clean = self.clean_text(text)

        # Tokenize
        encoded = self.tokenizer(
            clean,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )

        return encoded

    def batch_tokenize(
        self,
        texts: List[str],
        max_length: int = 128,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return type

        Returns:
            Dictionary with batched tensors
        """
        # Clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]

        # Batch tokenize
        encoded = self.tokenizer(
            cleaned_texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )

        return encoded


class ImagePreprocessor:
    """
    Image preprocessing for ResNet models.
    Handles thumbnail resizing, normalization, and augmentation.
    """

    def __init__(
        self,
        image_size: int = 224,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize image preprocessor.

        Args:
            image_size: Target image size (assumes square images)
            normalize: Whether to apply ImageNet normalization
            augment: Whether to apply data augmentation (for training)
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment

        # Build transformation pipeline
        transform_list = []

        # Resize
        transform_list.append(transforms.Resize((image_size, image_size)))

        # Augmentation (for training)
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
            ])

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize with ImageNet stats
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        self.transform = transforms.Compose(transform_list)

    def preprocess(self, image: Union[Image.Image, str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image: PIL Image, file path, or numpy array

        Returns:
            Preprocessed image tensor [3, H, W]
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        # Apply transforms
        return self.transform(image)

    def preprocess_batch(self, images: List[Union[Image.Image, str, np.ndarray]]) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of images

        Returns:
            Batched tensor [B, 3, H, W]
        """
        processed = [self.preprocess(img) for img in images]
        return torch.stack(processed)


def preprocess_dataset(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    drop_missing: bool = True,
    deduplicate: bool = True,
    filter_english: bool = True
) -> pd.DataFrame:
    """
    Preprocess the raw dataset according to the data cleaning plan.

    From APS360_project_proposal.md:
    - Deduplicate videos by unique video ID
    - Exclude non-English titles for simplicity
    - Handle missing values

    Args:
        df: Raw dataset DataFrame
        text_columns: Columns to preprocess (e.g., ['title', 'description'])
        drop_missing: Whether to drop rows with critical missing values
        deduplicate: Whether to deduplicate by video ID
        filter_english: Whether to filter for English-only content

    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Starting preprocessing. Initial shape: {df.shape}")

    df_clean = df.copy()

    # 1. Deduplicate by video ID
    if deduplicate:
        if 'video_id' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset='video_id', keep='first')
            after = len(df_clean)
            logger.info(f"Removed {before - after} duplicate videos")

    # 2. Filter for English content
    if filter_english:
        if 'language' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['language'].str.lower().isin(['en', 'english'])]
            after = len(df_clean)
            logger.info(f"Filtered to English content: {before} → {after} videos")
        elif 'title' in df_clean.columns:
            # Simple heuristic: check if title contains mostly ASCII characters
            def is_likely_english(text):
                if pd.isna(text) or not isinstance(text, str):
                    return False
                ascii_chars = sum(1 for c in text if ord(c) < 128)
                return ascii_chars / len(text) > 0.8 if len(text) > 0 else False

            before = len(df_clean)
            df_clean = df_clean[df_clean['title'].apply(is_likely_english)]
            after = len(df_clean)
            logger.info(f"Filtered to likely English titles: {before} → {after} videos")

    # 3. Handle missing values
    if drop_missing:
        # Identify critical columns (commonly found in video datasets)
        critical_cols = []
        for col in ['video_id', 'title', 'views', 'likes']:
            if col in df_clean.columns:
                critical_cols.append(col)

        if critical_cols:
            before = len(df_clean)
            df_clean = df_clean.dropna(subset=critical_cols)
            after = len(df_clean)
            logger.info(f"Dropped rows with missing critical values: {before} → {after} videos")

    # 4. Clean text columns
    if text_columns is None:
        text_columns = [col for col in ['title', 'description', 'hashtags'] if col in df_clean.columns]

    preprocessor = TextPreprocessor()
    for col in text_columns:
        if col in df_clean.columns:
            logger.info(f"Cleaning text column: {col}")
            df_clean[f'{col}_clean'] = df_clean[col].apply(
                lambda x: preprocessor.clean_text(str(x)) if pd.notna(x) else ""
            )

    logger.info(f"Preprocessing complete. Final shape: {df_clean.shape}")

    return df_clean


def create_text_features(
    df: pd.DataFrame,
    text_column: str = 'title',
    max_length: int = 128
) -> pd.DataFrame:
    """
    Create tokenized text features for the dataset.

    Args:
        df: Input DataFrame
        text_column: Column name containing text
        max_length: Maximum sequence length for tokenization

    Returns:
        DataFrame with added tokenization features
    """
    logger.info(f"Creating text features from column: {text_column}")

    preprocessor = TextPreprocessor()

    # Tokenize all texts
    texts = df[text_column].fillna("").astype(str).tolist()
    logger.info(f"Tokenizing {len(texts)} texts...")

    # Note: For large datasets, this should be done in batches
    # For now, we'll store the raw text and tokenize in the Dataset class
    df['text_clean'] = [preprocessor.clean_text(text) for text in texts]

    logger.info("Text features created successfully")
    return df


if __name__ == "__main__":
    # Test the preprocessors
    logger.info("Testing TextPreprocessor...")
    text_prep = TextPreprocessor()

    test_text = "Check out this AMAZING video! https://youtu.be/test #viral #trending 🔥🔥🔥"
    cleaned = text_prep.clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")

    tokens = text_prep.tokenize(test_text)
    print(f"Tokens shape: {tokens['input_ids'].shape}")

    logger.info("\nTesting ImagePreprocessor...")
    img_prep = ImagePreprocessor(image_size=224, augment=False)
    # Create a dummy image
    dummy_img = Image.new('RGB', (300, 300), color='red')
    processed = img_prep.preprocess(dummy_img)
    print(f"Processed image shape: {processed.shape}")

    logger.info("\nAll tests passed!")
