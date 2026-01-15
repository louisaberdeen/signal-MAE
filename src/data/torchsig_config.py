"""
Configuration classes for TorchSig RF dataset generation.

Provides comprehensive configuration for generating synthetic RF datasets
with control over modulation types, signal parameters, noise, and impairments.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json


class ImpairmentLevel(Enum):
    """Channel impairment levels for RF signal generation."""
    PERFECT = 0      # No impairments - clean signals
    CABLED = 1       # Cable channel - minor impairments
    WIRELESS = 2     # Wireless channel - realistic impairments


class ModulationFamily(Enum):
    """Modulation families available in TorchSig."""
    ASK = "ask"      # Amplitude Shift Keying
    PSK = "psk"      # Phase Shift Keying
    QAM = "qam"      # Quadrature Amplitude Modulation
    FSK = "fsk"      # Frequency Shift Keying
    OFDM = "ofdm"    # Orthogonal Frequency Division Multiplexing
    ANALOG = "analog"  # Analog modulations (AM, FM)


# Comprehensive modulation type definitions
MODULATION_TYPES: Dict[str, List[str]] = {
    "ask": [
        "ook",       # On-Off Keying
        "4ask",
        "8ask",
        "16ask",
        "32ask",
        "64ask",
    ],
    "psk": [
        "bpsk",      # Binary PSK
        "qpsk",      # Quadrature PSK
        "8psk",
        "16psk",
        "32psk",
        "64psk",
    ],
    "qam": [
        "16qam",
        "32qam",
        "32qam_cross",
        "64qam",
        "128qam_cross",
        "256qam",
        "512qam_cross",
        "1024qam",
    ],
    "fsk": [
        "2fsk",
        "4fsk",
        "8fsk",
        "16fsk",
        "2gfsk",     # Gaussian FSK
        "4gfsk",
        "8gfsk",
        "16gfsk",
        "2msk",      # Minimum Shift Keying
        "4msk",
        "2gmsk",     # Gaussian MSK
        "4gmsk",
    ],
    "ofdm": [
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ],
    "analog": [
        "am-dsb",    # AM Double Sideband
        "am-dsb-sc", # AM DSB Suppressed Carrier
        "am-lsb",    # AM Lower Sideband
        "am-usb",    # AM Upper Sideband
        "fm",        # Frequency Modulation
    ],
}


def get_all_modulations() -> List[str]:
    """Get flat list of all available modulation types."""
    all_mods = []
    for family_mods in MODULATION_TYPES.values():
        all_mods.extend(family_mods)
    return all_mods


def get_modulations_by_family(families: List[str]) -> List[str]:
    """Get modulations for specified families."""
    mods = []
    for family in families:
        family_lower = family.lower()
        if family_lower in MODULATION_TYPES:
            mods.extend(MODULATION_TYPES[family_lower])
    return mods


@dataclass
class SignalConfig:
    """
    Configuration for individual signal generation parameters.

    Controls signal-level characteristics like bandwidth, duration,
    and center frequency ranges.
    """
    # Center frequency range (normalized, -0.5 to 0.5 of sample rate)
    center_freq_min: Optional[float] = None
    center_freq_max: Optional[float] = None

    # Bandwidth range (normalized, 0 to 1 of sample rate)
    bandwidth_min: Optional[float] = None
    bandwidth_max: Optional[float] = None

    # Duration range (0 to 1, fraction of total sample length)
    duration_min: Optional[float] = None
    duration_max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for torchsig."""
        d = {}
        if self.center_freq_min is not None:
            d['signal_center_freq_min'] = self.center_freq_min
        if self.center_freq_max is not None:
            d['signal_center_freq_max'] = self.center_freq_max
        if self.bandwidth_min is not None:
            d['signal_bandwidth_min'] = self.bandwidth_min
        if self.bandwidth_max is not None:
            d['signal_bandwidth_max'] = self.bandwidth_max
        if self.duration_min is not None:
            d['signal_duration_min'] = self.duration_min
        if self.duration_max is not None:
            d['signal_duration_max'] = self.duration_max
        return d


@dataclass
class NoiseConfig:
    """
    Configuration for noise and SNR parameters.

    Controls signal-to-noise ratio ranges for realistic signal simulation.
    """
    # SNR range in dB
    snr_db_min: float = 0.0
    snr_db_max: float = 30.0

    # Whether to use fixed SNR (use snr_db_min as fixed value)
    fixed_snr: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for torchsig."""
        if self.fixed_snr:
            return {
                'snr_db_min': self.snr_db_min,
                'snr_db_max': self.snr_db_min,  # Same value = fixed
            }
        return {
            'snr_db_min': self.snr_db_min,
            'snr_db_max': self.snr_db_max,
        }


@dataclass
class TorchSigConfig:
    """
    Comprehensive configuration for TorchSig RF dataset generation.

    This configuration class provides full control over all aspects of
    synthetic RF dataset generation including:
    - Sample parameters (rate, length, count)
    - Modulation types and distribution
    - Signal characteristics (frequency, bandwidth, duration)
    - Noise and SNR parameters
    - Channel impairments
    - Multi-signal detection scenarios

    Example:
        config = TorchSigConfig(
            name="my_rf_dataset",
            num_samples=10000,
            modulations=["bpsk", "qpsk", "16qam", "64qam"],
            sample_rate=10e6,
            snr_db_min=0,
            snr_db_max=30,
            impairment_level=ImpairmentLevel.WIRELESS,
        )

        # Or use preset
        config = TorchSigConfig.classification_preset()

    Args:
        name: Dataset name (used for directory naming)
        output_dir: Root directory for generated datasets
        num_samples: Total number of samples to generate
        num_iq_samples: Number of I/Q samples per signal (length)
        sample_rate: Sample rate in Hz (default: 10 MHz)
        fft_size: FFT size for spectrogram generation
        modulations: List of modulation types to include
        modulation_families: Alternative to modulations - specify families
        class_distribution: Optional probability distribution for classes
        impairment_level: Channel impairment level (perfect/cabled/wireless)
        snr_db_min: Minimum SNR in dB
        snr_db_max: Maximum SNR in dB
        num_signals_min: Min signals per sample (1 for classification)
        num_signals_max: Max signals per sample (>1 for detection)
        signal_config: Detailed signal parameter configuration
        cochannel_interference_prob: Probability of co-channel interference
        random_seed: Random seed for reproducibility
        num_workers: Number of parallel workers for generation
        batch_size: Batch size for writing to disk
    """
    # Dataset identification
    name: str = "torchsig_rf"
    output_dir: Path = field(default_factory=lambda: Path("data/rf_datasets"))

    # Sample parameters
    num_samples: int = 10000
    num_iq_samples: int = 4096
    sample_rate: float = 10e6  # 10 MHz default
    fft_size: int = 1024

    # Modulation configuration
    modulations: Optional[List[str]] = None
    modulation_families: Optional[List[str]] = None
    class_distribution: Optional[List[float]] = None

    # Channel and impairment configuration
    impairment_level: ImpairmentLevel = ImpairmentLevel.WIRELESS

    # SNR configuration
    snr_db_min: float = 0.0
    snr_db_max: float = 30.0

    # Multi-signal configuration (for detection tasks)
    num_signals_min: int = 1
    num_signals_max: int = 1  # Set >1 for detection tasks
    num_signals_distribution: Optional[List[float]] = None

    # Signal parameters
    signal_config: Optional[SignalConfig] = None

    # Interference
    cochannel_interference_prob: float = 0.0

    # Generation parameters
    random_seed: Optional[int] = None
    num_workers: int = 4
    batch_size: int = 100

    # Output format options
    save_spectrograms: bool = True
    save_iq_data: bool = True
    spectrogram_size: int = 224

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        self.output_dir = Path(self.output_dir)

        # Set default modulations if none specified
        if self.modulations is None and self.modulation_families is None:
            # Default: common digital modulations
            self.modulations = [
                "bpsk", "qpsk", "8psk",
                "16qam", "64qam", "256qam",
                "2fsk", "4fsk",
            ]
        elif self.modulations is None and self.modulation_families is not None:
            # Generate from families
            self.modulations = get_modulations_by_family(self.modulation_families)

        # Validate modulations
        all_valid = get_all_modulations()
        invalid = [m for m in self.modulations if m not in all_valid]
        if invalid:
            raise ValueError(
                f"Invalid modulation types: {invalid}. "
                f"Valid types: {all_valid}"
            )

        # Validate class distribution if provided
        if self.class_distribution is not None:
            if len(self.class_distribution) != len(self.modulations):
                raise ValueError(
                    f"class_distribution length ({len(self.class_distribution)}) "
                    f"must match modulations length ({len(self.modulations)})"
                )
            if abs(sum(self.class_distribution) - 1.0) > 1e-6:
                raise ValueError("class_distribution must sum to 1.0")

        # Validate SNR range
        if self.snr_db_min > self.snr_db_max:
            raise ValueError("snr_db_min must be <= snr_db_max")

        # Validate signal counts
        if self.num_signals_min > self.num_signals_max:
            raise ValueError("num_signals_min must be <= num_signals_max")
        if self.num_signals_min < 1:
            raise ValueError("num_signals_min must be >= 1")

    @property
    def dataset_path(self) -> Path:
        """Get full path to dataset directory."""
        return self.output_dir / self.name

    @property
    def is_classification_task(self) -> bool:
        """Check if this is a classification (single signal) task."""
        return self.num_signals_max == 1

    @property
    def is_detection_task(self) -> bool:
        """Check if this is a detection (multi-signal) task."""
        return self.num_signals_max > 1

    @property
    def num_classes(self) -> int:
        """Get number of modulation classes."""
        return len(self.modulations)

    def to_torchsig_metadata(self) -> Dict[str, Any]:
        """
        Convert to TorchSig DatasetMetadata parameters.

        Returns:
            Dictionary of parameters for DatasetMetadata constructor
        """
        params = {
            'num_iq_samples_dataset': self.num_iq_samples,
            'num_samples': self.num_samples,
            'fft_size': self.fft_size,
            'sample_rate': self.sample_rate,
            'impairment_level': self.impairment_level.value,
            'class_list': self.modulations,
            'snr_db_min': self.snr_db_min,
            'snr_db_max': self.snr_db_max,
            'num_signals_min': self.num_signals_min,
            'num_signals_max': self.num_signals_max,
            'cochannel_overlap_probability': self.cochannel_interference_prob,
        }

        # Add class distribution if specified
        if self.class_distribution is not None:
            params['class_distribution'] = self.class_distribution

        # Add signal distribution if specified
        if self.num_signals_distribution is not None:
            params['num_signals_distribution'] = self.num_signals_distribution

        # Add signal config parameters
        if self.signal_config is not None:
            params.update(self.signal_config.to_dict())

        return params

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'output_dir': str(self.output_dir),
            'num_samples': self.num_samples,
            'num_iq_samples': self.num_iq_samples,
            'sample_rate': self.sample_rate,
            'fft_size': self.fft_size,
            'modulations': self.modulations,
            'modulation_families': self.modulation_families,
            'class_distribution': self.class_distribution,
            'impairment_level': self.impairment_level.name,
            'snr_db_min': self.snr_db_min,
            'snr_db_max': self.snr_db_max,
            'num_signals_min': self.num_signals_min,
            'num_signals_max': self.num_signals_max,
            'num_signals_distribution': self.num_signals_distribution,
            'signal_config': self.signal_config.to_dict() if self.signal_config else None,
            'cochannel_interference_prob': self.cochannel_interference_prob,
            'random_seed': self.random_seed,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'save_spectrograms': self.save_spectrograms,
            'save_iq_data': self.save_iq_data,
            'spectrogram_size': self.spectrogram_size,
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        """
        Convert to JSON string and optionally save to file.

        Args:
            path: Optional path to save JSON file

        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)
        return json_str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TorchSigConfig':
        """Create configuration from dictionary."""
        # Handle impairment level conversion
        if 'impairment_level' in d and isinstance(d['impairment_level'], str):
            d['impairment_level'] = ImpairmentLevel[d['impairment_level']]

        # Handle signal config
        if 'signal_config' in d and d['signal_config'] is not None:
            d['signal_config'] = SignalConfig(**d['signal_config'])

        # Handle Path conversion
        if 'output_dir' in d:
            d['output_dir'] = Path(d['output_dir'])

        return cls(**d)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'TorchSigConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        d = json.loads(path.read_text())
        return cls.from_dict(d)

    @classmethod
    def classification_preset(
        cls,
        name: str = "rf_classification",
        num_samples: int = 10000,
        difficulty: str = "medium",
        **kwargs
    ) -> 'TorchSigConfig':
        """
        Create preset for single-signal classification tasks.

        Args:
            name: Dataset name
            num_samples: Number of samples
            difficulty: 'easy', 'medium', or 'hard'
            **kwargs: Override any default parameters

        Returns:
            Configured TorchSigConfig for classification
        """
        presets = {
            'easy': {
                'snr_db_min': 10.0,
                'snr_db_max': 30.0,
                'impairment_level': ImpairmentLevel.PERFECT,
            },
            'medium': {
                'snr_db_min': 0.0,
                'snr_db_max': 30.0,
                'impairment_level': ImpairmentLevel.CABLED,
            },
            'hard': {
                'snr_db_min': -5.0,
                'snr_db_max': 20.0,
                'impairment_level': ImpairmentLevel.WIRELESS,
            },
        }

        if difficulty not in presets:
            raise ValueError(f"difficulty must be one of: {list(presets.keys())}")

        config_params = {
            'name': name,
            'num_samples': num_samples,
            'num_signals_max': 1,
            'modulations': [
                "bpsk", "qpsk", "8psk", "16psk",
                "16qam", "64qam", "256qam",
                "2fsk", "4fsk", "8fsk",
                "ofdm-64", "ofdm-128",
            ],
            **presets[difficulty],
            **kwargs,
        }

        return cls(**config_params)

    @classmethod
    def detection_preset(
        cls,
        name: str = "rf_detection",
        num_samples: int = 5000,
        max_signals: int = 5,
        **kwargs
    ) -> 'TorchSigConfig':
        """
        Create preset for multi-signal detection tasks.

        Args:
            name: Dataset name
            num_samples: Number of samples
            max_signals: Maximum signals per sample
            **kwargs: Override any default parameters

        Returns:
            Configured TorchSigConfig for detection
        """
        return cls(
            name=name,
            num_samples=num_samples,
            num_signals_min=1,
            num_signals_max=max_signals,
            impairment_level=ImpairmentLevel.WIRELESS,
            snr_db_min=-5.0,
            snr_db_max=25.0,
            cochannel_interference_prob=0.2,
            modulations=[
                "bpsk", "qpsk", "8psk",
                "16qam", "64qam",
                "2fsk", "4fsk",
                "ofdm-64", "ofdm-256",
            ],
            **kwargs,
        )

    @classmethod
    def minimal_test_preset(
        cls,
        name: str = "rf_test",
        num_samples: int = 100,
        **kwargs
    ) -> 'TorchSigConfig':
        """
        Create minimal preset for testing and development.

        Args:
            name: Dataset name
            num_samples: Number of samples (small for testing)
            **kwargs: Override any default parameters

        Returns:
            Configured TorchSigConfig for quick testing
        """
        return cls(
            name=name,
            num_samples=num_samples,
            num_iq_samples=1024,
            modulations=["bpsk", "qpsk", "16qam"],
            impairment_level=ImpairmentLevel.PERFECT,
            snr_db_min=10.0,
            snr_db_max=20.0,
            **kwargs,
        )

    def summary(self) -> str:
        """Generate human-readable configuration summary."""
        lines = [
            f"TorchSig RF Dataset Configuration",
            f"=" * 40,
            f"Name: {self.name}",
            f"Output: {self.dataset_path}",
            f"",
            f"Sample Parameters:",
            f"  Samples: {self.num_samples:,}",
            f"  IQ Length: {self.num_iq_samples:,}",
            f"  Sample Rate: {self.sample_rate/1e6:.1f} MHz",
            f"  FFT Size: {self.fft_size}",
            f"",
            f"Modulations ({self.num_classes} classes):",
            f"  {', '.join(self.modulations)}",
            f"",
            f"Channel Configuration:",
            f"  Impairment: {self.impairment_level.name}",
            f"  SNR Range: {self.snr_db_min:.1f} to {self.snr_db_max:.1f} dB",
            f"",
            f"Task Type: {'Classification' if self.is_classification_task else 'Detection'}",
        ]

        if self.is_detection_task:
            lines.append(f"  Signals per sample: {self.num_signals_min}-{self.num_signals_max}")
            lines.append(f"  Co-channel interference: {self.cochannel_interference_prob*100:.0f}%")

        return "\n".join(lines)
