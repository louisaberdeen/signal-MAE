"""
Model plugins for audio/signal representation learning.

Available models:
- audiomae++: AudioMAE++ with Macaron blocks, SwiGLU, RoPE
- baseline: Standard ViT-MAE for comparison
- signalmae: Masked autoencoder for RF spectrograms
- signalmae-small: Smaller variant for faster training

Usage:
    from src import model_registry
    from src.config import Config, create_rf_config

    # Audio model
    config = Config()
    model = model_registry.create("audiomae++", config)

    # RF signal model
    config = create_rf_config("base")
    model = model_registry.create("signalmae", config)
"""

from src.registry import model_registry

# Import classes to trigger registration
from src.models.audiomae import AudioMAEPlusPlus
from src.models.baseline import BaselineMAE
from src.models.classifier import AudioMAEClassifier
from src.models.signalmae import (
    SignalMAE,
    SignalMAESmall,
    SignalMAEClassifier,
    create_signalmae,
)

__all__ = [
    "model_registry",
    # Audio models
    "AudioMAEPlusPlus",
    "BaselineMAE",
    "AudioMAEClassifier",
    # RF signal models
    "SignalMAE",
    "SignalMAESmall",
    "SignalMAEClassifier",
    "create_signalmae",
]
