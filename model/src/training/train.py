"""
Main training loop for multimodal virality prediction model.

Includes:
- Mixed precision training (AMP)
- MLflow experiment tracking
- Early stopping
- Gradient clipping
- Learning rate scheduling
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .utils import (
    load_config,
    set_seed,
    get_device,
    get_optimizer,
    get_scheduler,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    clip_gradients,
    AverageMeter,
    count_parameters
)
from .evaluate import evaluate_model, print_evaluation_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("MLflow not available. Experiment tracking disabled.")
    MLFLOW_AVAILABLE = False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    classification_criterion: nn.Module,
    regression_criterion: nn.Module,
    device: torch.device,
    classification_weight: float = 0.7,
    regression_weight: float = 0.3,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
    log_interval: int = 50
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training DataLoader
        optimizer: Optimizer
        classification_criterion: Loss for classification
        regression_criterion: Loss for regression
        device: Training device
        classification_weight: Weight for classification loss
        regression_weight: Weight for regression loss
        max_grad_norm: Maximum gradient norm for clipping
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP
        log_interval: Log every N batches

    Returns:
        Dictionary with epoch metrics
    """
    model.train()

    # Metrics tracking
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        text = {k: v.to(device) for k, v in batch['text'].items()}
        scalars = batch['scalars'].to(device)
        labels = batch['label'].to(device)
        velocity = batch['velocity'].to(device)

        # Handle images if present
        images = batch['image'].to(device) if 'image' in batch else None

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with AMP
        if use_amp and scaler is not None:
            with autocast():
                cls_logits, reg_output = model(
                    text_input=text,
                    image_input=images,
                    scalar_features=scalars
                )

                # Compute losses
                cls_loss = classification_criterion(cls_logits, labels)
                reg_loss = regression_criterion(reg_output.squeeze(), velocity)
                loss = classification_weight * cls_loss + regression_weight * reg_loss

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model, max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

        else:
            # Standard training without AMP
            cls_logits, reg_output = model(
                text_input=text,
                image_input=images,
                scalar_features=scalars
            )

            # Compute losses
            cls_loss = classification_criterion(cls_logits, labels)
            reg_loss = regression_criterion(reg_output.squeeze(), velocity)
            loss = classification_weight * cls_loss + regression_weight * reg_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = clip_gradients(model, max_grad_norm)

            # Optimizer step
            optimizer.step()

        # Update metrics
        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        cls_loss_meter.update(cls_loss.item(), batch_size)
        reg_loss_meter.update(reg_loss.item(), batch_size)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls': f'{cls_loss_meter.avg:.4f}',
            'reg': f'{reg_loss_meter.avg:.4f}'
        })

        # Log at intervals
        if (batch_idx + 1) % log_interval == 0:
            logger.debug(
                f"Batch {batch_idx + 1}/{len(dataloader)}: "
                f"loss={loss_meter.avg:.4f}, "
                f"cls_loss={cls_loss_meter.avg:.4f}, "
                f"reg_loss={reg_loss_meter.avg:.4f}, "
                f"grad_norm={grad_norm:.4f}"
            )

    return {
        'train_loss': loss_meter.avg,
        'train_cls_loss': cls_loss_meter.avg,
        'train_reg_loss': reg_loss_meter.avg,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    resume_from: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Complete training loop with validation and checkpointing.

    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Training configuration dictionary
        device: Training device
        resume_from: Path to checkpoint to resume from (optional)

    Returns:
        Tuple of (trained model, best metrics)
    """
    logger.info("Starting training...")

    # Extract config
    train_config = config['training']
    epochs = train_config['epochs']
    lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    classification_weight = train_config['classification_weight']
    regression_weight = train_config['regression_weight']
    max_grad_norm = train_config['max_grad_norm']
    patience = train_config['patience']
    min_delta = train_config['min_delta']
    checkpoint_dir = train_config['checkpoint_dir']
    use_amp = config['hardware']['mixed_precision']

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Log model parameters
    params = count_parameters(model)
    logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Setup optimizer
    optimizer = get_optimizer(
        model,
        optimizer_name=train_config['optimizer'],
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=train_config.get('betas', (0.9, 0.999)),
        eps=train_config.get('eps', 1e-8)
    )

    # Setup scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=train_config.get('scheduler', 'linear_warmup_cosine'),
        num_epochs=epochs,
        steps_per_epoch=len(train_loader),
        warmup_steps=train_config.get('warmup_steps', 500)
    )

    # Setup loss functions
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    # Setup AMP
    scaler = GradScaler() if use_amp else None

    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode='max'  # Maximize AUROC
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    best_auroc = 0.0

    if resume_from:
        model, optimizer, scheduler, start_epoch, checkpoint_metrics = load_checkpoint(
            resume_from, model, optimizer, scheduler, device
        )
        best_auroc = checkpoint_metrics.get('val_auroc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best AUROC: {best_auroc:.4f}")

    # Setup MLflow
    if MLFLOW_AVAILABLE and config.get('logging', {}).get('mlflow_tracking_uri'):
        mlflow.set_tracking_uri(config['logging']['mlflow_tracking_uri'])
        mlflow.set_experiment(config['logging']['experiment_name'])

        with mlflow.start_run(run_name=config['logging']['run_name']):
            # Log parameters
            mlflow.log_params({
                'learning_rate': lr,
                'batch_size': config['data']['batch_size'],
                'epochs': epochs,
                'optimizer': train_config['optimizer'],
                'classification_weight': classification_weight,
                'regression_weight': regression_weight,
            })

            # Training loop
            best_auroc, model = _training_loop(
                model, train_loader, val_loader,
                optimizer, scheduler,
                classification_criterion, regression_criterion,
                device, early_stopping,
                start_epoch, epochs,
                classification_weight, regression_weight,
                max_grad_norm, use_amp, scaler,
                checkpoint_dir, best_auroc,
                log_interval=train_config.get('log_every_n_steps', 50)
            )

    else:
        # Training loop without MLflow
        best_auroc, model = _training_loop(
            model, train_loader, val_loader,
            optimizer, scheduler,
            classification_criterion, regression_criterion,
            device, early_stopping,
            start_epoch, epochs,
            classification_weight, regression_weight,
            max_grad_norm, use_amp, scaler,
            checkpoint_dir, best_auroc,
            log_interval=train_config.get('log_every_n_steps', 50)
        )

    logger.info(f"Training complete! Best validation AUROC: {best_auroc:.4f}")

    return model, {'best_auroc': best_auroc}


def _training_loop(
    model, train_loader, val_loader,
    optimizer, scheduler,
    classification_criterion, regression_criterion,
    device, early_stopping,
    start_epoch, epochs,
    classification_weight, regression_weight,
    max_grad_norm, use_amp, scaler,
    checkpoint_dir, best_auroc,
    log_interval
):
    """Internal training loop (separated for MLflow context management)."""

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*70}")

        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader,
            optimizer,
            classification_criterion, regression_criterion,
            device,
            classification_weight, regression_weight,
            max_grad_norm, use_amp, scaler,
            log_interval
        )

        # Validate
        logger.info("Validating...")
        val_metrics = evaluate_model(
            model, val_loader, device,
            classification_criterion, regression_criterion,
            classification_weight, regression_weight
        )

        # Add prefix to validation metrics
        val_metrics = {f"val_{k}" if not k.startswith('val_') else k: v
                      for k, v in val_metrics.items()}

        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}

        # Update learning rate
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            epoch_metrics['learning_rate'] = current_lr

        # Log metrics
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time

        logger.info(
            f"Epoch {epoch + 1} Summary: "
            f"train_loss={train_metrics['train_loss']:.4f}, "
            f"val_auroc={val_metrics['val_auroc']:.4f}, "
            f"val_mae={val_metrics.get('val_velocity_mae', 0):.4f}, "
            f"time={epoch_time:.2f}s"
        )

        # Log to MLflow
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics(epoch_metrics, step=epoch)

        # Check for improvement
        val_auroc = val_metrics['val_auroc']

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            logger.info(f"New best AUROC: {best_auroc:.4f} - Saving checkpoint")

            # Save best checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                epoch_metrics, checkpoint_dir,
                filename='best_model.pt'
            )

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            epoch_metrics, checkpoint_dir,
            filename='latest_model.pt'
        )

        # Early stopping check
        if early_stopping(val_auroc):
            logger.info("Early stopping triggered!")
            break

    # Load best model
    best_checkpoint = Path(checkpoint_dir) / 'best_model.pt'
    if best_checkpoint.exists():
        model, _, _, _, _ = load_checkpoint(str(best_checkpoint), model, device=device)
        logger.info("Loaded best model for final evaluation")

    return best_auroc, model


if __name__ == "__main__":
    # Test training setup
    logger.info("Testing training setup...")

    # Load config
    config = load_config("src/configs/training_config.yaml")

    # Set seed
    set_seed(config['seed'])

    # Get device
    device = get_device(config['hardware']['device'])

    logger.info("Training setup test complete!")
