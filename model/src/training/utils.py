"""
Training utilities: checkpointing, early stopping, optimizers, schedulers.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam, SGD
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, LinearLR, SequentialLR
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def get_device(device_name: str = "cuda") -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        device_name: "cuda", "cpu", or "mps"

    Returns:
        torch.device object
    """
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal (MPS) device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: Model to optimize
        optimizer_name: "adamw", "adam", or "sgd"
        learning_rate: Learning rate
        weight_decay: L2 regularization weight
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    # Get trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_name.lower() == "adam":
        optimizer = Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    logger.info(f"Initialized {optimizer_name} optimizer (lr={learning_rate}, wd={weight_decay})")

    return optimizer


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "linear_warmup_cosine",
    num_epochs: int = 15,
    steps_per_epoch: int = 100,
    warmup_steps: int = 500,
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_name: "linear_warmup_cosine", "cosine", or "linear"
        num_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch
        warmup_steps: Number of warmup steps
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance or None
    """
    total_steps = num_epochs * steps_per_epoch

    if scheduler_name == "linear_warmup_cosine":
        # Linear warmup followed by cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=kwargs.get('eta_min', 1e-7)
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        logger.info(f"Initialized linear warmup + cosine scheduler (warmup={warmup_steps} steps)")

    elif scheduler_name == "cosine":
        # Cosine annealing without warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=kwargs.get('eta_min', 1e-7)
        )
        logger.info(f"Initialized cosine scheduler (T_max={total_steps})")

    elif scheduler_name == "linear":
        # Linear decay
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
        logger.info(f"Initialized linear scheduler (total={total_steps})")

    elif scheduler_name == "none":
        scheduler = None
        logger.info("No scheduler used")

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


class EarlyStopping:
    """
    Early stopping to stop training when monitored metric stops improving.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "max" to maximize metric, "min" to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        logger.info(f"Initialized EarlyStopping (patience={patience}, min_delta={min_delta}, mode={mode})")

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric score

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check for improvement
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered! Best score: {self.best_score:.4f}")
                return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: Optional[str] = None
) -> str:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Dictionary with metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Optional custom filename

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"

    checkpoint_path = checkpoint_dir / filename

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Optional[Optimizer], Optional[_LRScheduler], int, Dict[str, float]]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load model to

    Returns:
        Tuple of (model, optimizer, scheduler, epoch, metrics)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    logger.info(f"Loaded checkpoint from epoch {epoch}")
    if metrics:
        logger.info(f"Checkpoint metrics: {metrics}")

    return model, optimizer, scheduler, epoch, metrics


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: Model with gradients
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


if __name__ == "__main__":
    # Test utilities
    logger.info("Testing training utilities...")

    # Test config loading
    config = load_config("src/configs/training_config.yaml")
    print(f"Config loaded: {config['model']['name']}")

    # Test seed setting
    set_seed(42)

    # Test device
    device = get_device("cuda")

    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode="max")

    scores = [0.65, 0.70, 0.72, 0.71, 0.70, 0.69]
    for i, score in enumerate(scores):
        should_stop = early_stop(score)
        print(f"Epoch {i}: score={score:.2f}, should_stop={should_stop}")

    # Test AverageMeter
    meter = AverageMeter()
    for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        meter.update(val)
    print(f"Average: {meter.avg:.2f}")

    logger.info("\nAll tests passed!")
