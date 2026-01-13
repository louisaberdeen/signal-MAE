"""
Data loader plugins for various audio/signal datasets.

Available loaders:
- esc50: ESC-50 environmental sound dataset
- custom: Generic CSV-based loader
- rf: Basic RF signal data loader
- torchsig: TorchSig-generated RF datasets
- torchsig_iq: TorchSig datasets (IQ data as primary)
- torchsig_png: TorchSig datasets (PNG spectrograms)

Usage:
    from src import data_loader_registry

    # Audio datasets
    loader = data_loader_registry.create("esc50", data_root=Path("data/ESC-50-master"))
    metadata = loader.load_metadata()

    # RF datasets (generated with TorchSig)
    loader = data_loader_registry.create("torchsig", data_root=Path("data/rf_datasets/my_dataset"))
    metadata = loader.load_metadata()

RF Dataset Generation:
    from src.data.torchsig_config import TorchSigConfig
    from src.data.torchsig_generator import TorchSigGenerator

    config = TorchSigConfig.classification_preset(num_samples=10000)
    generator = TorchSigGenerator(config)
    generator.generate()
"""

from src.registry import data_loader_registry

# Import classes to trigger registration
from src.data.esc50 import ESC50DataLoader
from src.data.custom import CustomAudioDataLoader, RFSignalDataLoader
from src.data.torchsig import (
    TorchSigDataLoader,
    TorchSigIQDataLoader,
    TorchSigPNGDataLoader,
)
from src.data.torchsig_config import (
    TorchSigConfig,
    ImpairmentLevel,
    ModulationFamily,
    SignalConfig,
    NoiseConfig,
    MODULATION_TYPES,
    get_all_modulations,
    get_modulations_by_family,
)
from src.data.torchsig_generator import (
    TorchSigGenerator,
    generate_rf_dataset,
)

__all__ = [
    # Registry
    "data_loader_registry",
    # Audio loaders
    "ESC50DataLoader",
    "CustomAudioDataLoader",
    "RFSignalDataLoader",
    # TorchSig loaders
    "TorchSigDataLoader",
    "TorchSigIQDataLoader",
    "TorchSigPNGDataLoader",
    # TorchSig configuration
    "TorchSigConfig",
    "ImpairmentLevel",
    "ModulationFamily",
    "SignalConfig",
    "NoiseConfig",
    "MODULATION_TYPES",
    "get_all_modulations",
    "get_modulations_by_family",
    # TorchSig generator
    "TorchSigGenerator",
    "generate_rf_dataset",
]
