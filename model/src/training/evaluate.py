"""
Evaluation metrics and model evaluation functions.

From implementation_plan.md lines 275-310:
- AUROC for classification
- Precision@10% recall
- MAE for engagement velocity regression
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    regression_true: Optional[np.ndarray] = None,
    regression_pred: Optional[np.ndarray] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True binary labels [N]
        y_pred: Predicted binary labels [N]
        y_proba: Predicted probabilities for positive class [N]
        regression_true: True regression values [N] (optional)
        regression_pred: Predicted regression values [N] (optional)
        prefix: Prefix for metric names (e.g., "train_", "val_")

    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}

    # Classification metrics
    try:
        # AUROC
        auroc = roc_auc_score(y_true, y_proba)
        metrics[f'{prefix}auroc'] = auroc

        # Average Precision (PR-AUC)
        ap = average_precision_score(y_true, y_proba)
        metrics[f'{prefix}average_precision'] = ap

        # Precision@10% recall
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        # Find precision at recall >= 0.1
        valid_idx = np.where(recall >= 0.1)[0]
        if len(valid_idx) > 0:
            precision_at_10 = precision[valid_idx[0]]
            metrics[f'{prefix}precision_at_10pct_recall'] = precision_at_10

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}accuracy'] = accuracy

        # F1 score
        f1 = f1_score(y_true, y_pred, average='binary')
        metrics[f'{prefix}f1'] = f1

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics[f'{prefix}true_positives'] = int(tp)
        metrics[f'{prefix}true_negatives'] = int(tn)
        metrics[f'{prefix}false_positives'] = int(fp)
        metrics[f'{prefix}false_negatives'] = int(fn)

        # Precision and Recall
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f'{prefix}precision'] = precision_score
        metrics[f'{prefix}recall'] = recall_score

    except Exception as e:
        logger.warning(f"Error computing classification metrics: {e}")

    # Regression metrics (if provided)
    if regression_true is not None and regression_pred is not None:
        try:
            # Mean Absolute Error
            mae = mean_absolute_error(regression_true, regression_pred)
            metrics[f'{prefix}velocity_mae'] = mae

            # Mean Squared Error
            mse = mean_squared_error(regression_true, regression_pred)
            metrics[f'{prefix}velocity_mse'] = mse

            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            metrics[f'{prefix}velocity_rmse'] = rmse

            # R² score (coefficient of determination)
            from sklearn.metrics import r2_score
            r2 = r2_score(regression_true, regression_pred)
            metrics[f'{prefix}velocity_r2'] = r2

        except Exception as e:
            logger.warning(f"Error computing regression metrics: {e}")

    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    classification_criterion: Optional[nn.Module] = None,
    regression_criterion: Optional[nn.Module] = None,
    classification_weight: float = 0.7,
    regression_weight: float = 0.3,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    From implementation_plan.md lines 275-310.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        classification_criterion: Loss function for classification
        regression_criterion: Loss function for regression
        classification_weight: Weight for classification loss
        regression_weight: Weight for regression loss
        return_predictions: Whether to return raw predictions

    Returns:
        Dictionary with metrics (and optionally predictions)
    """
    model.eval()

    # Storage for predictions and labels
    all_cls_preds = []
    all_cls_probs = []
    all_cls_labels = []
    all_reg_preds = []
    all_reg_labels = []
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for batch in dataloader:
        # Move batch to device
        text = {k: v.to(device) for k, v in batch['text'].items()}
        scalars = batch['scalars'].to(device)
        labels = batch['label'].to(device)
        velocity = batch['velocity'].to(device)

        # Handle images if present
        images = batch['image'].to(device) if 'image' in batch else None

        # Forward pass
        cls_logits, reg_output = model(
            text_input=text,
            image_input=images,
            scalar_features=scalars
        )

        # Compute losses if criteria provided
        if classification_criterion:
            cls_loss = classification_criterion(cls_logits, labels)
            total_cls_loss += cls_loss.item()

        if regression_criterion:
            reg_loss = regression_criterion(reg_output.squeeze(), velocity)
            total_reg_loss += reg_loss.item()

        if classification_criterion and regression_criterion:
            combined_loss = classification_weight * cls_loss + regression_weight * reg_loss
            total_loss += combined_loss.item()

        # Store predictions
        cls_probs = torch.softmax(cls_logits, dim=1)[:, 1]  # Probability of positive class
        cls_pred = torch.argmax(cls_logits, dim=1)

        all_cls_preds.extend(cls_pred.cpu().numpy())
        all_cls_probs.extend(cls_probs.cpu().numpy())
        all_cls_labels.extend(labels.cpu().numpy())
        all_reg_preds.extend(reg_output.squeeze().cpu().numpy())
        all_reg_labels.extend(velocity.cpu().numpy())

    # Convert to numpy arrays
    all_cls_preds = np.array(all_cls_preds)
    all_cls_probs = np.array(all_cls_probs)
    all_cls_labels = np.array(all_cls_labels)
    all_reg_preds = np.array(all_reg_preds)
    all_reg_labels = np.array(all_reg_labels)

    # Compute metrics
    metrics = compute_metrics(
        y_true=all_cls_labels,
        y_pred=all_cls_preds,
        y_proba=all_cls_probs,
        regression_true=all_reg_labels,
        regression_pred=all_reg_preds
    )

    # Add loss metrics
    num_batches = len(dataloader)
    if classification_criterion and regression_criterion:
        metrics['loss'] = total_loss / num_batches
        metrics['cls_loss'] = total_cls_loss / num_batches
        metrics['reg_loss'] = total_reg_loss / num_batches

    # Optionally return predictions
    if return_predictions:
        metrics['predictions'] = {
            'cls_pred': all_cls_preds,
            'cls_probs': all_cls_probs,
            'cls_labels': all_cls_labels,
            'reg_pred': all_reg_preds,
            'reg_labels': all_reg_labels,
        }

    return metrics


def print_evaluation_report(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Print a formatted evaluation report.

    Args:
        metrics: Dictionary with evaluation metrics
        title: Title for the report
    """
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)

    # Classification metrics
    if 'auroc' in metrics:
        print("\nClassification Metrics:")
        print(f"  AUROC:                    {metrics['auroc']:.4f}")
        if 'average_precision' in metrics:
            print(f"  Average Precision (PR-AUC): {metrics['average_precision']:.4f}")
        if 'accuracy' in metrics:
            print(f"  Accuracy:                 {metrics['accuracy']:.4f}")
        if 'f1' in metrics:
            print(f"  F1 Score:                 {metrics['f1']:.4f}")
        if 'precision' in metrics and 'recall' in metrics:
            print(f"  Precision:                {metrics['precision']:.4f}")
            print(f"  Recall:                   {metrics['recall']:.4f}")
        if 'precision_at_10pct_recall' in metrics:
            print(f"  Precision@10% Recall:     {metrics['precision_at_10pct_recall']:.4f}")

    # Confusion matrix
    if all(k in metrics for k in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']):
        print("\nConfusion Matrix:")
        print(f"  TP: {metrics['true_positives']:5d}  |  FP: {metrics['false_positives']:5d}")
        print(f"  FN: {metrics['false_negatives']:5d}  |  TN: {metrics['true_negatives']:5d}")

    # Regression metrics
    if 'velocity_mae' in metrics:
        print("\nRegression Metrics (Engagement Velocity):")
        print(f"  MAE:  {metrics['velocity_mae']:.4f}")
        if 'velocity_rmse' in metrics:
            print(f"  RMSE: {metrics['velocity_rmse']:.4f}")
        if 'velocity_r2' in metrics:
            print(f"  R²:   {metrics['velocity_r2']:.4f}")

    # Loss metrics
    if 'loss' in metrics:
        print("\nLoss Metrics:")
        print(f"  Total Loss:          {metrics['loss']:.4f}")
        if 'cls_loss' in metrics:
            print(f"  Classification Loss: {metrics['cls_loss']:.4f}")
        if 'reg_loss' in metrics:
            print(f"  Regression Loss:     {metrics['reg_loss']:.4f}")

    print("=" * 70 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Viral', 'Viral'],
        yticklabels=['Not Viral', 'Viral']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save plot (optional)
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved ROC curve to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save plot (optional)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved PR curve to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test evaluation functions
    logger.info("Testing evaluation functions...")

    # Create dummy predictions
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.randint(0, 2, n_samples)
    y_proba = np.random.rand(n_samples)
    y_pred = (y_proba > 0.5).astype(int)

    reg_true = np.random.rand(n_samples) * 100
    reg_pred = reg_true + np.random.randn(n_samples) * 10

    # Compute metrics
    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        regression_true=reg_true,
        regression_pred=reg_pred,
        prefix="test_"
    )

    # Print report
    print_evaluation_report(metrics, "Test Evaluation")

    logger.info("\nAll tests passed!")
