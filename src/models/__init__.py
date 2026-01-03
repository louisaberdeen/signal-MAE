"""
Model plugins for audio/signal representation learning.

Available models:
- audiomae++: AudioMAE++ with Macaron blocks, SwiGLU, RoPE
- baseline: Standard ViT-MAE for comparison

Usage:
    from src import model_registry
    from src.config import Config

    config = Config()
    model = model_registry.create("audiomae++", config)
"""

from src.registry import model_registry

# Import classes to trigger registration
from src.models.audiomae import AudioMAEPlusPlus
from src.models.baseline import BaselineMAE
from src.models.classifier import AudioMAEClassifier

__all__ = [
    "model_registry",
    "AudioMAEPlusPlus",
    "BaselineMAE",
    "AudioMAEClassifier",
]
