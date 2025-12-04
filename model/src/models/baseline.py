"""
Baseline logistic regression model using TF-IDF features.

From APS360_project_proposal.md: "A traditional baseline model will be constructed
using logistic regression with TF-IDF unigram features extracted from video titles
and hashtags. This transparent model sets a reference AUROC of approximately 0.65."
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Baseline logistic regression model with TF-IDF features.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 1),  # Unigrams only
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: str = 'balanced'
    ):
        """
        Initialize the baseline model.

        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF (default: unigrams only)
            max_iter: Maximum iterations for logistic regression
            random_state: Random seed
            class_weight: Class weight strategy ('balanced' or None)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight

        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency (ignore very common words)
        )

        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
            solver='lbfgs'
        )

        self.is_fitted = False

        logger.info(f"Initialized BaselineModel with max_features={max_features}, ngram_range={ngram_range}")

    def _prepare_text(self, df: pd.DataFrame, text_columns: list = ['title', 'hashtags']) -> pd.Series:
        """
        Combine text columns into a single text field.

        Args:
            df: Input DataFrame
            text_columns: Columns to combine

        Returns:
            Series with combined text
        """
        combined_texts = []

        for idx, row in df.iterrows():
            text_parts = []

            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))

            combined_text = ' '.join(text_parts)
            combined_texts.append(combined_text)

        return pd.Series(combined_texts)

    def fit(
        self,
        train_df: pd.DataFrame,
        label_column: str = 'is_viral',
        text_columns: list = ['title', 'hashtags']
    ) -> 'BaselineModel':
        """
        Fit the baseline model on training data.

        Args:
            train_df: Training DataFrame
            label_column: Column with binary labels
            text_columns: Text columns to use for features

        Returns:
            self
        """
        logger.info("Fitting baseline model...")

        # Prepare text
        texts = self._prepare_text(train_df, text_columns)
        labels = train_df[label_column].values

        # Fit vectorizer and transform text
        logger.info(f"Vectorizing {len(texts)} training samples...")
        X_train = self.vectorizer.fit_transform(texts)

        logger.info(f"TF-IDF feature matrix shape: {X_train.shape}")
        logger.info(f"Non-zero entries: {X_train.nnz} ({100 * X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.2f}%)")

        # Fit logistic regression
        logger.info("Training logistic regression...")
        self.model.fit(X_train, labels)

        self.is_fitted = True
        logger.info("Baseline model training complete!")

        # Log feature importance (top features)
        self._log_top_features(n=20)

        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        text_columns: list = ['title', 'hashtags']
    ) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            test_df: Test DataFrame
            text_columns: Text columns to use

        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        texts = self._prepare_text(test_df, text_columns)
        X_test = self.vectorizer.transform(texts)

        return self.model.predict(X_test)

    def predict_proba(
        self,
        test_df: pd.DataFrame,
        text_columns: list = ['title', 'hashtags']
    ) -> np.ndarray:
        """
        Predict probability scores.

        Args:
            test_df: Test DataFrame
            text_columns: Text columns to use

        Returns:
            Array of predicted probabilities [N, 2]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        texts = self._prepare_text(test_df, text_columns)
        X_test = self.vectorizer.transform(texts)

        return self.model.predict_proba(X_test)

    def evaluate(
        self,
        test_df: pd.DataFrame,
        label_column: str = 'is_viral',
        text_columns: list = ['title', 'hashtags']
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_df: Test DataFrame
            label_column: Column with true labels
            text_columns: Text columns to use

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating baseline model...")

        # Get predictions
        y_true = test_df[label_column].values
        y_pred = self.predict(test_df, text_columns)
        y_proba = self.predict_proba(test_df, text_columns)[:, 1]

        # Calculate metrics
        auroc = roc_auc_score(y_true, y_proba)

        metrics = {
            'auroc': auroc,
            'accuracy': (y_pred == y_true).mean(),
        }

        # Log results
        logger.info(f"AUROC: {auroc:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        logger.info("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Not Viral', 'Viral']))

        logger.info("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        return metrics

    def _log_top_features(self, n: int = 20):
        """Log top N most important features (highest coefficients)."""
        if not self.is_fitted:
            return

        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]

        # Get top positive and negative coefficients
        top_positive_idx = np.argsort(coefficients)[-n:][::-1]
        top_negative_idx = np.argsort(coefficients)[:n]

        logger.info(f"\nTop {n} features predicting VIRAL:")
        for idx in top_positive_idx:
            logger.info(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

        logger.info(f"\nTop {n} features predicting NOT VIRAL:")
        for idx in top_negative_idx:
            logger.info(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

    def save(self, filepath: str):
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BaselineModel':
        """
        Load a model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded BaselineModel instance
        """
        model_data = joblib.load(filepath)

        # Create instance
        instance = cls(
            max_features=model_data['max_features'],
            ngram_range=model_data['ngram_range']
        )

        # Restore fitted components
        instance.vectorizer = model_data['vectorizer']
        instance.model = model_data['model']
        instance.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

        return instance


if __name__ == "__main__":
    # Test the baseline model
    logger.info("Testing BaselineModel...")

    # Create sample data
    sample_data = pd.DataFrame({
        'title': [
            'Amazing viral video #trending',
            'Boring content nothing special',
            'Incredible breakthrough #viral #amazing',
            'Regular everyday video',
            'Must watch this goes viral #trending',
        ],
        'hashtags': [
            '#viral #trending',
            '',
            '#viral #amazing #breakthrough',
            '#daily',
            '#trending #mustsee',
        ],
        'is_viral': [1, 0, 1, 0, 1],
    })

    # Create train/test split
    train_df = sample_data.iloc[:3]
    test_df = sample_data.iloc[3:]

    # Train model
    model = BaselineModel(max_features=100)
    model.fit(train_df)

    # Evaluate
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)

    print(f"\nPredictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")

    logger.info("\nAll tests passed!")
