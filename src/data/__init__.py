"""
Data loader plugins for various audio/signal datasets.

Available loaders:
- esc50: ESC-50 environmental sound dataset
- radioml: RadioML RF signal dataset
- custom: Generic CSV-based loader

Usage:
    from src import data_loader_registry

    loader = data_loader_registry.create("esc50", data_root=Path("data/ESC-50-master"))
    metadata = loader.load_metadata()
"""

from src.registry import data_loader_registry

# Import classes to trigger registration
from src.data.esc50 import ESC50DataLoader
from src.data.custom import CustomAudioDataLoader, RFSignalDataLoader

__all__ = [
    "data_loader_registry",
    "ESC50DataLoader",
    "CustomAudioDataLoader",
    "RFSignalDataLoader",
]
