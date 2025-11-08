"""
Training module for the Miles project.
Handles training loops, evaluation, and training utilities.
"""

from .train import train_model, train_epoch
from .evaluate import evaluate_model, compute_metrics
from .utils import (
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    get_optimizer,
    get_scheduler,
)

__all__ = [
    'train_model',
    'train_epoch',
    'evaluate_model',
    'compute_metrics',
    'save_checkpoint',
    'load_checkpoint',
    'EarlyStopping',
    'get_optimizer',
    'get_scheduler',
]
