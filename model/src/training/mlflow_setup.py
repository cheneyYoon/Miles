"""
MLflow experiment tracking setup and utilities.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow(
    tracking_uri: str = "experiments/mlruns",
    experiment_name: str = "viral_shorts_prediction"
) -> None:
    """
    Initialize MLflow tracking.

    Args:
        tracking_uri: Path to MLflow tracking directory
        experiment_name: Name of the experiment
    """
    # Create tracking directory
    Path(tracking_uri).mkdir(parents=True, exist_ok=True)

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

        mlflow.set_experiment(experiment_name)

    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        raise


def log_model_info(model, model_name: str = "multimodal_predictor"):
    """
    Log model architecture and parameters to MLflow.

    Args:
        model: PyTorch model
        model_name: Name for the model artifact
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Cannot log model info.")
        return

    try:
        # Count parameters
        from ..training.utils import count_parameters
        params = count_parameters(model)

        # Log parameter counts
        mlflow.log_params({
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'frozen_parameters': params['frozen'],
        })

        # Log model architecture as text
        model_summary = str(model)
        mlflow.log_text(model_summary, "model_architecture.txt")

        logger.info(f"Logged model info for {model_name}")

    except Exception as e:
        logger.warning(f"Error logging model info: {e}")


def log_config(config: Dict):
    """
    Log configuration to MLflow.

    Args:
        config: Configuration dictionary
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Cannot log config.")
        return

    try:
        # Flatten nested config
        flat_config = {}

        def flatten_dict(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, new_key)
                else:
                    flat_config[new_key] = v

        flatten_dict(config)

        # Log parameters
        mlflow.log_params(flat_config)

        logger.info("Logged configuration to MLflow")

    except Exception as e:
        logger.warning(f"Error logging config: {e}")


def log_dataset_info(train_size: int, val_size: int, test_size: int):
    """
    Log dataset split sizes to MLflow.

    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Cannot log dataset info.")
        return

    mlflow.log_params({
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'total_size': train_size + val_size + test_size,
    })

    logger.info(f"Logged dataset info: train={train_size}, val={val_size}, test={test_size}")


def log_artifacts(artifact_dir: str):
    """
    Log directory of artifacts to MLflow.

    Args:
        artifact_dir: Path to directory containing artifacts
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Cannot log artifacts.")
        return

    try:
        mlflow.log_artifacts(artifact_dir)
        logger.info(f"Logged artifacts from {artifact_dir}")
    except Exception as e:
        logger.warning(f"Error logging artifacts: {e}")


if __name__ == "__main__":
    # Test MLflow setup
    logger.info("Testing MLflow setup...")

    # Initialize MLflow
    setup_mlflow()

    # Start a test run
    with mlflow.start_run(run_name="test_run"):
        # Log some test parameters
        mlflow.log_params({
            'test_param_1': 0.001,
            'test_param_2': 'test_value'
        })

        # Log some test metrics
        for i in range(5):
            mlflow.log_metrics({
                'test_loss': 1.0 / (i + 1),
                'test_accuracy': 0.5 + i * 0.1
            }, step=i)

        logger.info("Test run completed successfully")

    logger.info(f"\nView results with: mlflow ui --backend-store-uri experiments/mlruns")
