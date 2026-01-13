"""
TorchSig RF dataset generator.

Generates synthetic RF datasets using TorchSig library with comprehensive
configuration support for modulation types, noise, and channel impairments.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.torchsig_config import TorchSigConfig, ImpairmentLevel

logger = logging.getLogger(__name__)


class TorchSigGenerator:
    """
    Generate synthetic RF datasets using TorchSig.

    This class wraps TorchSig's dataset generation capabilities and provides:
    - Configuration-driven dataset creation
    - Progress tracking and logging
    - Automatic spectrogram generation
    - Metadata CSV creation for data loader compatibility
    - Resumable generation for large datasets

    Example:
        from src.data.torchsig_config import TorchSigConfig
        from src.data.torchsig_generator import TorchSigGenerator

        config = TorchSigConfig.classification_preset(
            name="my_dataset",
            num_samples=10000,
            difficulty="medium"
        )

        generator = TorchSigGenerator(config)
        generator.generate()

    Args:
        config: TorchSigConfig instance with generation parameters
        verbose: Enable verbose logging
    """

    def __init__(self, config: TorchSigConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._torchsig_available = None

    def _check_torchsig(self) -> bool:
        """Check if torchsig is available."""
        if self._torchsig_available is None:
            try:
                import torchsig
                self._torchsig_available = True
                logger.info(f"TorchSig version: {getattr(torchsig, '__version__', 'unknown')}")
            except ImportError:
                self._torchsig_available = False
                logger.warning(
                    "TorchSig not installed. Install with: pip install torchsig"
                )
        return self._torchsig_available

    def _setup_directories(self) -> Dict[str, Path]:
        """Create output directory structure."""
        base_dir = self.config.dataset_path
        dirs = {
            'base': base_dir,
            'iq': base_dir / 'iq',
            'spectrograms': base_dir / 'spectrograms',
            'spectrograms_png': base_dir / 'spectrograms_png',
            'metadata': base_dir,
        }

        for name, path in dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logger.info(f"Created directory: {path}")

        return dirs

    def _save_config(self, dirs: Dict[str, Path]) -> None:
        """Save configuration to dataset directory."""
        config_path = dirs['base'] / 'config.json'
        self.config.to_json(config_path)
        logger.info(f"Saved configuration to: {config_path}")

    def _create_torchsig_dataset(self) -> Any:
        """Create TorchSig iterable dataset."""
        from torchsig.datasets.datasets import TorchSigIterableDataset
        from torchsig.datasets.dataset_metadata import DatasetMetadata

        # Get parameters from config
        params = self.config.to_torchsig_metadata()

        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info("Creating TorchSig DatasetMetadata...")
        metadata = DatasetMetadata(**params)

        logger.info("Creating TorchSigIterableDataset...")
        dataset = TorchSigIterableDataset(metadata)

        return dataset

    def _generate_spectrogram(
        self,
        iq_data: np.ndarray,
        nperseg: int = 64,
        noverlap: int = 32,
    ) -> np.ndarray:
        """
        Generate spectrogram from IQ data.

        Args:
            iq_data: Complex IQ samples or [2, N] array
            nperseg: Samples per FFT segment
            noverlap: Overlap between segments

        Returns:
            Spectrogram as 2D numpy array (normalized to 0-1)
        """
        from scipy import signal as scipy_signal

        # Convert to complex if needed
        if iq_data.ndim == 2 and iq_data.shape[0] == 2:
            complex_data = iq_data[0] + 1j * iq_data[1]
        elif np.iscomplexobj(iq_data):
            complex_data = iq_data
        else:
            complex_data = iq_data

        # Compute spectrogram
        _, _, Sxx = scipy_signal.spectrogram(
            complex_data,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False,
        )

        # Convert to dB and normalize
        Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
        Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-8)

        return Sxx_norm

    def _resize_spectrogram(
        self,
        spectrogram: np.ndarray,
        size: int = 224
    ) -> np.ndarray:
        """Resize spectrogram to target size."""
        from PIL import Image

        img = Image.fromarray((spectrogram * 255).astype(np.uint8))
        img = img.resize((size, size), Image.BILINEAR)
        return np.array(img) / 255.0

    def _iterate_dataset(
        self,
        dataset: Any,
        num_samples: int
    ) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Iterate through TorchSig dataset yielding samples.

        Args:
            dataset: TorchSigIterableDataset instance
            num_samples: Number of samples to generate

        Yields:
            Tuple of (iq_data, metadata_dict)
        """
        count = 0
        for sample in dataset:
            if count >= num_samples:
                break

            # Extract IQ data and metadata
            # TorchSig returns (data, target) or more complex structures
            if isinstance(sample, tuple):
                iq_data = sample[0]
                if len(sample) > 1:
                    target = sample[1]
                else:
                    target = None
            else:
                iq_data = sample
                target = None

            # Convert to numpy if tensor
            if hasattr(iq_data, 'numpy'):
                iq_data = iq_data.numpy()

            # Build metadata
            meta = {
                'index': count,
                'target': target,
            }

            yield iq_data, meta
            count += 1

    def generate(self, resume: bool = False) -> Path:
        """
        Generate the RF dataset.

        Args:
            resume: If True, resume from last checkpoint

        Returns:
            Path to generated dataset directory

        Raises:
            ImportError: If TorchSig is not installed
            RuntimeError: If generation fails
        """
        if not self._check_torchsig():
            raise ImportError(
                "TorchSig is required for dataset generation. "
                "Install with: pip install torchsig"
            )

        logger.info(f"\n{self.config.summary()}\n")

        # Setup directories
        dirs = self._setup_directories()
        self._save_config(dirs)

        # Create dataset
        dataset = self._create_torchsig_dataset()

        # Track metadata for CSV
        metadata_records = []

        # Generate samples with progress bar
        logger.info(f"Generating {self.config.num_samples:,} samples...")

        pbar = tqdm(
            self._iterate_dataset(dataset, self.config.num_samples),
            total=self.config.num_samples,
            desc="Generating",
            disable=not self.verbose
        )

        for iq_data, meta in pbar:
            idx = meta['index']
            target = meta['target']

            # Determine label
            if isinstance(target, (int, np.integer)):
                label = self.config.modulations[target]
                target_idx = int(target)
            elif isinstance(target, str):
                label = target
                target_idx = self.config.modulations.index(target) if target in self.config.modulations else -1
            else:
                label = "unknown"
                target_idx = -1

            # Generate filename
            filename = f"sample_{idx:06d}"

            # Save IQ data
            if self.config.save_iq_data:
                iq_path = dirs['iq'] / f"{filename}.npy"
                np.save(iq_path, iq_data)

            # Generate and save spectrogram
            if self.config.save_spectrograms:
                spectrogram = self._generate_spectrogram(iq_data)
                spectrogram_resized = self._resize_spectrogram(
                    spectrogram,
                    self.config.spectrogram_size
                )

                # Save as .npy
                spec_path = dirs['spectrograms'] / f"{filename}.npy"
                np.save(spec_path, spectrogram_resized)

                # Save as PNG for visualization
                from PIL import Image
                png_path = dirs['spectrograms_png'] / f"{filename}.png"
                img = Image.fromarray((spectrogram_resized * 255).astype(np.uint8))
                img.save(png_path)

            # Record metadata
            metadata_records.append({
                'filename': f"{filename}.npy",
                'label': label,
                'target': target_idx,
                'modulation': label,
                'snr_db': meta.get('snr_db', np.nan),
                'sample_rate': self.config.sample_rate,
                'num_iq_samples': self.config.num_iq_samples,
            })

            # Update progress bar description
            if idx % 100 == 0:
                pbar.set_postfix({'label': label})

        # Save metadata CSV
        metadata_df = pd.DataFrame(metadata_records)
        metadata_path = dirs['metadata'] / 'metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to: {metadata_path}")

        # Save generation info
        info = {
            'generated_at': datetime.now().isoformat(),
            'num_samples': len(metadata_records),
            'num_classes': len(self.config.modulations),
            'modulations': self.config.modulations,
            'config': self.config.to_dict(),
        }
        info_path = dirs['base'] / 'generation_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        logger.info(f"\nDataset generated successfully!")
        logger.info(f"Location: {dirs['base']}")
        logger.info(f"Samples: {len(metadata_records):,}")
        logger.info(f"Classes: {len(self.config.modulations)}")

        return dirs['base']

    def generate_without_torchsig(self) -> Path:
        """
        Generate a synthetic dataset without TorchSig (fallback mode).

        Creates a simple dataset with basic modulated signals for testing
        when TorchSig is not available.

        Returns:
            Path to generated dataset directory
        """
        logger.warning(
            "TorchSig not available. Generating basic synthetic signals."
        )

        dirs = self._setup_directories()
        self._save_config(dirs)

        metadata_records = []

        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        pbar = tqdm(
            range(self.config.num_samples),
            desc="Generating (fallback)",
            disable=not self.verbose
        )

        for idx in pbar:
            # Random modulation selection
            mod_idx = np.random.randint(len(self.config.modulations))
            label = self.config.modulations[mod_idx]

            # Generate basic IQ signal (simplified synthetic data)
            iq_data = self._generate_basic_signal(
                label,
                self.config.num_iq_samples,
                self.config.snr_db_min,
                self.config.snr_db_max,
            )

            filename = f"sample_{idx:06d}"

            # Save IQ data
            if self.config.save_iq_data:
                iq_path = dirs['iq'] / f"{filename}.npy"
                np.save(iq_path, iq_data)

            # Generate and save spectrogram
            if self.config.save_spectrograms:
                spectrogram = self._generate_spectrogram(iq_data)
                spectrogram_resized = self._resize_spectrogram(
                    spectrogram,
                    self.config.spectrogram_size
                )

                spec_path = dirs['spectrograms'] / f"{filename}.npy"
                np.save(spec_path, spectrogram_resized)

                from PIL import Image
                png_path = dirs['spectrograms_png'] / f"{filename}.png"
                img = Image.fromarray((spectrogram_resized * 255).astype(np.uint8))
                img.save(png_path)

            metadata_records.append({
                'filename': f"{filename}.npy",
                'label': label,
                'target': mod_idx,
                'modulation': label,
                'snr_db': np.random.uniform(self.config.snr_db_min, self.config.snr_db_max),
                'sample_rate': self.config.sample_rate,
                'num_iq_samples': self.config.num_iq_samples,
            })

        # Save metadata
        metadata_df = pd.DataFrame(metadata_records)
        metadata_path = dirs['metadata'] / 'metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)

        # Save generation info
        info = {
            'generated_at': datetime.now().isoformat(),
            'num_samples': len(metadata_records),
            'num_classes': len(self.config.modulations),
            'modulations': self.config.modulations,
            'fallback_mode': True,
            'config': self.config.to_dict(),
        }
        info_path = dirs['base'] / 'generation_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        logger.info(f"\nDataset generated (fallback mode)!")
        logger.info(f"Location: {dirs['base']}")

        return dirs['base']

    def _generate_basic_signal(
        self,
        modulation: str,
        num_samples: int,
        snr_min: float,
        snr_max: float,
    ) -> np.ndarray:
        """
        Generate basic synthetic signal for fallback mode.

        This is a simplified version that doesn't require TorchSig.
        """
        t = np.arange(num_samples) / self.config.sample_rate

        # Generate carrier frequency
        fc = np.random.uniform(0.1, 0.4) * self.config.sample_rate

        # Generate base signal based on modulation type
        if 'psk' in modulation.lower() or 'bpsk' in modulation.lower():
            # Simple PSK-like signal
            symbols = np.random.choice([-1, 1], size=num_samples // 100 + 1)
            symbols = np.repeat(symbols, 100)[:num_samples]
            signal = symbols * np.exp(1j * 2 * np.pi * fc * t)

        elif 'qam' in modulation.lower():
            # Simple QAM-like signal
            I = np.random.choice([-3, -1, 1, 3], size=num_samples // 100 + 1)
            Q = np.random.choice([-3, -1, 1, 3], size=num_samples // 100 + 1)
            I = np.repeat(I, 100)[:num_samples]
            Q = np.repeat(Q, 100)[:num_samples]
            signal = (I + 1j * Q) * np.exp(1j * 2 * np.pi * fc * t)

        elif 'fsk' in modulation.lower():
            # Simple FSK-like signal
            freq_dev = 0.1 * self.config.sample_rate
            symbols = np.random.choice([-1, 1], size=num_samples // 100 + 1)
            symbols = np.repeat(symbols, 100)[:num_samples]
            phase = np.cumsum(symbols) * freq_dev / self.config.sample_rate
            signal = np.exp(1j * 2 * np.pi * (fc * t + phase))

        elif 'ofdm' in modulation.lower():
            # Simple OFDM-like signal (sum of carriers)
            num_carriers = int(modulation.split('-')[-1]) if '-' in modulation else 64
            num_carriers = min(num_carriers, 64)  # Limit for simplicity
            signal = np.zeros(num_samples, dtype=complex)
            for k in range(num_carriers):
                f_k = fc + (k - num_carriers // 2) * 1000
                signal += np.exp(1j * 2 * np.pi * f_k * t + 1j * np.random.uniform(0, 2*np.pi))
            signal /= num_carriers

        else:
            # Default: simple sinusoid with random phase
            signal = np.exp(1j * 2 * np.pi * fc * t + 1j * np.random.uniform(0, 2*np.pi))

        # Add noise based on SNR
        snr_db = np.random.uniform(snr_min, snr_max)
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        )

        noisy_signal = signal + noise

        # Return as [2, N] array (I and Q channels)
        return np.stack([noisy_signal.real, noisy_signal.imag], axis=0).astype(np.float32)

    def validate_dataset(self, dataset_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validate a generated dataset.

        Args:
            dataset_path: Path to dataset (defaults to config path)

        Returns:
            Validation results dictionary
        """
        path = dataset_path or self.config.dataset_path

        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {},
        }

        # Check directories exist
        required_dirs = ['iq', 'spectrograms', 'spectrograms_png']
        for dir_name in required_dirs:
            dir_path = path / dir_name
            if not dir_path.exists():
                if dir_name == 'iq' and not self.config.save_iq_data:
                    continue
                if 'spectrogram' in dir_name and not self.config.save_spectrograms:
                    continue
                results['errors'].append(f"Missing directory: {dir_name}")
                results['valid'] = False

        # Check metadata
        metadata_path = path / 'metadata.csv'
        if not metadata_path.exists():
            results['errors'].append("Missing metadata.csv")
            results['valid'] = False
        else:
            df = pd.read_csv(metadata_path)
            results['stats']['num_samples'] = len(df)
            results['stats']['num_classes'] = df['label'].nunique()
            results['stats']['class_distribution'] = df['label'].value_counts().to_dict()

            # Check required columns
            required_cols = ['filename', 'label', 'target']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                results['errors'].append(f"Missing columns: {missing_cols}")
                results['valid'] = False

        # Check config
        config_path = path / 'config.json'
        if not config_path.exists():
            results['warnings'].append("Missing config.json")

        return results


def generate_rf_dataset(
    config: Optional[TorchSigConfig] = None,
    preset: Optional[str] = None,
    **kwargs
) -> Path:
    """
    Convenience function to generate RF dataset.

    Args:
        config: TorchSigConfig instance (optional)
        preset: Preset name ('classification', 'detection', 'test')
        **kwargs: Additional arguments passed to config

    Returns:
        Path to generated dataset

    Example:
        # Using preset
        path = generate_rf_dataset(preset='classification', num_samples=5000)

        # Using config
        config = TorchSigConfig(name="custom", modulations=["bpsk", "qpsk"])
        path = generate_rf_dataset(config=config)
    """
    if config is None:
        if preset == 'classification':
            config = TorchSigConfig.classification_preset(**kwargs)
        elif preset == 'detection':
            config = TorchSigConfig.detection_preset(**kwargs)
        elif preset == 'test':
            config = TorchSigConfig.minimal_test_preset(**kwargs)
        else:
            config = TorchSigConfig(**kwargs)

    generator = TorchSigGenerator(config)

    # Try TorchSig first, fall back to basic generation
    try:
        return generator.generate()
    except ImportError:
        logger.warning("TorchSig not available, using fallback generator")
        return generator.generate_without_torchsig()
