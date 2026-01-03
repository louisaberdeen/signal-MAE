"""
Training utilities: losses, trainer, callbacks.

Components:
- losses: InfoNCE, uniformity, reconstruction losses
- trainer: Training loop abstraction
- callbacks: MLflow, checkpointing
"""

from src.training.losses import info_nce_loss, uniformity_loss, get_embedding
# from src.training.trainer import Trainer
# from src.training.callbacks import MLflowCallback, CheckpointCallback

__all__ = [
    "info_nce_loss",
    "uniformity_loss",
    "get_embedding",
]
