"""
Model plugins for audio/signal representation learning.

Available models:
- audiomae++: AudioMAE++ with Macaron blocks, SwiGLU, RoPE
- baseline: Standard ViT-MAE for comparison
- signalmae: Baseline MAE for RF spectrograms (simple)
- signalmae++: Advanced MAE for RF with Macaron, SwiGLU, RoPE
- signalmae-small: Smaller variant for faster training

Usage:
    from src import model_registry
    from src.config import Config, create_rf_config

    # Audio model
    config = Config()
    model = model_registry.create("audiomae++", config)

    # RF signal model (baseline)
    config = create_rf_config("base")
    model = model_registry.create("signalmae", config)

    # RF signal model (advanced with all features)
    config = create_rf_config("base", advanced=True)
    model = model_registry.create("signalmae++", config)
"""

from src.registry import model_registry

# Import classes to trigger registration
from src.models.audiomae import AudioMAEPlusPlus
from src.models.baseline import BaselineMAE
from src.models.classifier import AudioMAEClassifier
from src.models.signalmae import (
    SignalMAE,
    SignalMAEPlusPlus,
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
    "SignalMAEPlusPlus",
    "SignalMAESmall",
    "SignalMAEClassifier",
    "create_signalmae",
]
