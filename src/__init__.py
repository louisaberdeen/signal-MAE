"""
AudioMAE++ Framework - Modular audio/signal ML training framework.

This package provides:
- Plugin-based model, data loader, and transform registration
- Self-contained notebook generation for cloud training
- Automatic test generation for plugin compatibility

Usage:
    from src import model_registry, data_loader_registry, transform_registry
    from src.config import Config

    # Create a model
    model = model_registry.create("audiomae++", config)

    # Create a data loader
    loader = data_loader_registry.create("esc50", data_root)
"""

__version__ = "2.0.0"

from src.registry import (
    PluginRegistry,
    model_registry,
    data_loader_registry,
    transform_registry,
    loss_registry,
)

# Import submodules to trigger plugin registration
from src import models  # noqa: F401
from src import data  # noqa: F401
from src import transforms  # noqa: F401

__all__ = [
    "PluginRegistry",
    "model_registry",
    "data_loader_registry",
    "transform_registry",
    "loss_registry",
    "__version__",
]
