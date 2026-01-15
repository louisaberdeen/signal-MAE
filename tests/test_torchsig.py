"""
Tests for TorchSig RF dataset generation and loading.

These tests verify the torchsig integration including:
- Configuration validation
- Dataset generation (fallback mode without torchsig)
- Data loader functionality
- Plugin registration
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.torchsig_config import (
    TorchSigConfig,
    ImpairmentLevel,
    SignalConfig,
    NoiseConfig,
    MODULATION_TYPES,
    get_all_modulations,
    get_modulations_by_family,
)
from src.data.torchsig_generator import TorchSigGenerator
from src.data.torchsig import (
    TorchSigDataLoader,
    TorchSigIQDataLoader,
    TorchSigPNGDataLoader,
)
from src.registry import data_loader_registry


class TestTorchSigConfig:
    """Tests for TorchSigConfig configuration class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = TorchSigConfig()

        assert config.name == "torchsig_rf"
        assert config.num_samples == 10000
        assert config.num_iq_samples == 4096
        assert config.sample_rate == 10e6
        assert config.impairment_level == ImpairmentLevel.WIRELESS
        assert config.num_signals_max == 1  # Classification task
        assert len(config.modulations) > 0

    def test_custom_modulations(self):
        """Test custom modulation list."""
        config = TorchSigConfig(
            modulations=["bpsk", "qpsk", "16qam"]
        )

        assert config.modulations == ["bpsk", "qpsk", "16qam"]
        assert config.num_classes == 3

    def test_modulation_families(self):
        """Test modulation family selection."""
        config = TorchSigConfig(
            modulation_families=["psk", "qam"]
        )

        # Should include all PSK and QAM modulations
        for mod in config.modulations:
            assert mod in MODULATION_TYPES["psk"] or mod in MODULATION_TYPES["qam"]

    def test_invalid_modulation_raises(self):
        """Test that invalid modulations raise error."""
        with pytest.raises(ValueError, match="Invalid modulation"):
            TorchSigConfig(modulations=["invalid_mod"])

    def test_snr_validation(self):
        """Test SNR range validation."""
        with pytest.raises(ValueError, match="snr_db_min must be"):
            TorchSigConfig(snr_db_min=30, snr_db_max=10)

    def test_signal_count_validation(self):
        """Test signal count validation."""
        with pytest.raises(ValueError, match="num_signals_min must be"):
            TorchSigConfig(num_signals_min=5, num_signals_max=2)

    def test_classification_preset(self):
        """Test classification preset creation."""
        config = TorchSigConfig.classification_preset(
            name="test_cls",
            num_samples=1000,
            difficulty="easy"
        )

        assert config.name == "test_cls"
        assert config.num_samples == 1000
        assert config.num_signals_max == 1
        assert config.impairment_level == ImpairmentLevel.PERFECT
        assert config.snr_db_min >= 10  # Easy = higher SNR

    def test_detection_preset(self):
        """Test detection preset creation."""
        config = TorchSigConfig.detection_preset(
            name="test_det",
            max_signals=5
        )

        assert config.num_signals_max == 5
        assert config.is_detection_task

    def test_minimal_test_preset(self):
        """Test minimal preset for quick testing."""
        config = TorchSigConfig.minimal_test_preset()

        assert config.num_samples == 100
        assert config.num_iq_samples == 1024
        assert len(config.modulations) <= 5

    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        config = TorchSigConfig(
            name="roundtrip_test",
            modulations=["bpsk", "qpsk"],
            snr_db_min=5,
            snr_db_max=25,
        )

        d = config.to_dict()
        config2 = TorchSigConfig.from_dict(d)

        assert config2.name == config.name
        assert config2.modulations == config.modulations
        assert config2.snr_db_min == config.snr_db_min
        assert config2.snr_db_max == config.snr_db_max

    def test_to_json_file(self):
        """Test JSON file save/load."""
        config = TorchSigConfig(name="json_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            assert json_path.exists()

            config2 = TorchSigConfig.from_json(json_path)
            assert config2.name == config.name

    def test_to_torchsig_metadata(self):
        """Test conversion to TorchSig metadata parameters."""
        config = TorchSigConfig(
            modulations=["bpsk", "qpsk"],
            snr_db_min=0,
            snr_db_max=30,
            impairment_level=ImpairmentLevel.CABLED,
        )

        params = config.to_torchsig_metadata()

        assert params['class_list'] == ["bpsk", "qpsk"]
        assert params['snr_db_min'] == 0
        assert params['snr_db_max'] == 30
        assert params['impairment_level'] == 1  # CABLED = 1

    def test_summary(self):
        """Test human-readable summary generation."""
        config = TorchSigConfig(name="summary_test")
        summary = config.summary()

        assert "summary_test" in summary
        assert "Sample Parameters" in summary
        assert "Modulations" in summary


class TestModulationHelpers:
    """Tests for modulation helper functions."""

    def test_get_all_modulations(self):
        """Test getting all modulations."""
        all_mods = get_all_modulations()

        assert len(all_mods) > 50  # Should have 70+ modulations
        assert "bpsk" in all_mods
        assert "16qam" in all_mods
        assert "ofdm-64" in all_mods

    def test_get_modulations_by_family(self):
        """Test getting modulations by family."""
        psk_mods = get_modulations_by_family(["psk"])

        assert "bpsk" in psk_mods
        assert "qpsk" in psk_mods
        assert "16qam" not in psk_mods

    def test_modulation_types_structure(self):
        """Test MODULATION_TYPES structure."""
        assert "ask" in MODULATION_TYPES
        assert "psk" in MODULATION_TYPES
        assert "qam" in MODULATION_TYPES
        assert "fsk" in MODULATION_TYPES
        assert "ofdm" in MODULATION_TYPES


class TestSignalConfig:
    """Tests for SignalConfig class."""

    def test_signal_config_to_dict(self):
        """Test SignalConfig conversion."""
        sig_config = SignalConfig(
            center_freq_min=-0.3,
            center_freq_max=0.3,
            bandwidth_min=0.1,
        )

        d = sig_config.to_dict()

        assert d['signal_center_freq_min'] == -0.3
        assert d['signal_center_freq_max'] == 0.3
        assert d['signal_bandwidth_min'] == 0.1
        assert 'signal_bandwidth_max' not in d  # None values excluded


class TestNoiseConfig:
    """Tests for NoiseConfig class."""

    def test_noise_config_range(self):
        """Test noise config with range."""
        noise = NoiseConfig(snr_db_min=0, snr_db_max=30)
        d = noise.to_dict()

        assert d['snr_db_min'] == 0
        assert d['snr_db_max'] == 30

    def test_noise_config_fixed(self):
        """Test noise config with fixed SNR."""
        noise = NoiseConfig(snr_db_min=15, fixed_snr=True)
        d = noise.to_dict()

        assert d['snr_db_min'] == 15
        assert d['snr_db_max'] == 15


class TestTorchSigGenerator:
    """Tests for TorchSigGenerator class."""

    def test_generator_creation(self):
        """Test generator instantiation."""
        config = TorchSigConfig.minimal_test_preset()
        generator = TorchSigGenerator(config, verbose=False)

        assert generator.config == config

    def test_fallback_generation(self):
        """Test fallback generation without TorchSig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TorchSigConfig.minimal_test_preset(
                name="fallback_test",
                num_samples=10,
                output_dir=Path(tmpdir),
            )

            generator = TorchSigGenerator(config, verbose=False)
            dataset_path = generator.generate_without_torchsig()

            # Check directory structure
            assert dataset_path.exists()
            assert (dataset_path / "metadata.csv").exists()
            assert (dataset_path / "config.json").exists()
            assert (dataset_path / "iq").exists()
            assert (dataset_path / "spectrograms").exists()

            # Check metadata
            df = pd.read_csv(dataset_path / "metadata.csv")
            assert len(df) == 10
            assert "filename" in df.columns
            assert "label" in df.columns
            assert "target" in df.columns

            # Check IQ files
            iq_files = list((dataset_path / "iq").glob("*.npy"))
            assert len(iq_files) == 10

            # Check spectrogram files
            spec_files = list((dataset_path / "spectrograms").glob("*.npy"))
            assert len(spec_files) == 10

    def test_basic_signal_generation(self):
        """Test basic signal generation for different modulations."""
        config = TorchSigConfig.minimal_test_preset()
        generator = TorchSigGenerator(config, verbose=False)

        for mod in ["bpsk", "qpsk", "16qam", "2fsk", "ofdm-64"]:
            signal = generator._generate_basic_signal(
                modulation=mod,
                num_samples=1024,
                snr_min=10,
                snr_max=20,
            )

            assert signal.shape == (2, 1024)  # I/Q channels
            assert signal.dtype == np.float32

    def test_spectrogram_generation(self):
        """Test spectrogram generation from IQ data."""
        config = TorchSigConfig.minimal_test_preset()
        generator = TorchSigGenerator(config, verbose=False)

        # Generate test IQ data
        iq_data = np.random.randn(2, 1024).astype(np.float32)

        spectrogram = generator._generate_spectrogram(iq_data)

        assert spectrogram.ndim == 2
        assert spectrogram.min() >= 0
        assert spectrogram.max() <= 1

    def test_spectrogram_resize(self):
        """Test spectrogram resizing."""
        config = TorchSigConfig.minimal_test_preset()
        generator = TorchSigGenerator(config, verbose=False)

        spec = np.random.rand(64, 32)
        resized = generator._resize_spectrogram(spec, size=224)

        assert resized.shape == (224, 224)

    def test_dataset_validation(self):
        """Test dataset validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TorchSigConfig.minimal_test_preset(
                name="validation_test",
                num_samples=5,
                output_dir=Path(tmpdir),
            )

            generator = TorchSigGenerator(config, verbose=False)
            dataset_path = generator.generate_without_torchsig()

            validation = generator.validate_dataset(dataset_path)

            assert validation['valid']
            assert len(validation['errors']) == 0
            assert validation['stats']['num_samples'] == 5


class TestTorchSigDataLoader:
    """Tests for TorchSigDataLoader class."""

    @pytest.fixture
    def generated_dataset(self):
        """Create a temporary generated dataset for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TorchSigConfig.minimal_test_preset(
                name="loader_test",
                num_samples=10,
                output_dir=Path(tmpdir),
            )

            generator = TorchSigGenerator(config, verbose=False)
            dataset_path = generator.generate_without_torchsig()

            yield dataset_path

    def test_loader_creation(self, generated_dataset):
        """Test data loader creation."""
        loader = TorchSigDataLoader(generated_dataset)

        assert loader.data_root == generated_dataset
        assert loader.generation_config is not None

    def test_load_metadata(self, generated_dataset):
        """Test metadata loading."""
        loader = TorchSigDataLoader(generated_dataset)
        df = loader.load_metadata()

        assert len(df) == 10
        assert "filepath" in df.columns
        assert "label" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns

    def test_get_sample_paths(self, generated_dataset):
        """Test sample path retrieval."""
        loader = TorchSigDataLoader(generated_dataset)
        paths = loader.get_sample_paths()

        assert len(paths) == 10
        assert all(p.exists() for p in paths)

    def test_get_iq_paths(self, generated_dataset):
        """Test IQ path retrieval."""
        loader = TorchSigDataLoader(generated_dataset)
        paths = loader.get_iq_paths()

        assert len(paths) == 10
        assert all(p.suffix == ".npy" for p in paths)

    def test_load_sample(self, generated_dataset):
        """Test single sample loading."""
        loader = TorchSigDataLoader(generated_dataset)
        sample = loader.load_sample("sample_000000.npy")

        assert "iq" in sample
        assert "spectrogram" in sample
        assert "metadata" in sample
        assert sample["iq"].shape[0] == 2  # I/Q channels

    def test_get_class_distribution(self, generated_dataset):
        """Test class distribution retrieval."""
        loader = TorchSigDataLoader(generated_dataset)
        dist = loader.get_class_distribution()

        assert isinstance(dist, dict)
        assert sum(dist.values()) == 10

    def test_get_info(self, generated_dataset):
        """Test dataset info retrieval."""
        loader = TorchSigDataLoader(generated_dataset)
        info = loader.get_info()

        assert "modulations" in info
        assert "sample_rate" in info
        assert "has_iq_data" in info
        assert info["has_iq_data"]

    def test_validation(self, generated_dataset):
        """Test loader validation."""
        loader = TorchSigDataLoader(generated_dataset)
        is_valid, errors = loader.validate()

        assert is_valid
        assert len(errors) == 0

    def test_iq_loader_variant(self, generated_dataset):
        """Test IQ-focused loader variant."""
        loader = TorchSigIQDataLoader(generated_dataset)

        assert not loader.use_spectrograms
        paths = loader.get_sample_paths()
        assert all(p.parent.name == "iq" for p in paths)

    def test_png_loader_variant(self, generated_dataset):
        """Test PNG-focused loader variant."""
        loader = TorchSigPNGDataLoader(generated_dataset)

        assert loader.use_spectrograms
        assert loader.spectrogram_format == "png"


class TestPluginRegistration:
    """Tests for plugin registration."""

    def test_torchsig_loader_registered(self):
        """Test that TorchSig loader is registered."""
        assert "torchsig" in data_loader_registry

    def test_torchsig_iq_loader_registered(self):
        """Test that TorchSig IQ loader is registered."""
        assert "torchsig_iq" in data_loader_registry

    def test_torchsig_png_loader_registered(self):
        """Test that TorchSig PNG loader is registered."""
        assert "torchsig_png" in data_loader_registry

    def test_registry_metadata(self):
        """Test registry metadata."""
        metadata = data_loader_registry.get_metadata("torchsig")

        assert metadata["class_name"] == "TorchSigDataLoader"
        assert metadata["version"] == "1.0"

    def test_create_loader_from_registry(self):
        """Test creating loader through registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal dataset structure
            dataset_path = Path(tmpdir) / "test_dataset"
            (dataset_path / "iq").mkdir(parents=True)
            (dataset_path / "spectrograms").mkdir()
            (dataset_path / "spectrograms_png").mkdir()

            # Create minimal metadata
            df = pd.DataFrame({
                "filename": ["sample_000000.npy"],
                "label": ["bpsk"],
                "target": [0],
            })
            df.to_csv(dataset_path / "metadata.csv", index=False)

            # Create through registry
            loader = data_loader_registry.create("torchsig", data_root=dataset_path)
            assert isinstance(loader, TorchSigDataLoader)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test complete generation and loading pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create configuration
            config = TorchSigConfig(
                name="integration_test",
                output_dir=Path(tmpdir),
                num_samples=20,
                num_iq_samples=512,
                modulations=["bpsk", "qpsk", "16qam", "64qam"],
                snr_db_min=5,
                snr_db_max=25,
                impairment_level=ImpairmentLevel.PERFECT,
            )

            # 2. Generate dataset
            generator = TorchSigGenerator(config, verbose=False)
            dataset_path = generator.generate_without_torchsig()

            # 3. Load with data loader
            loader = TorchSigDataLoader(dataset_path)

            # 4. Verify
            df = loader.load_metadata()
            assert len(df) == 20
            assert set(df["label"].unique()) == {"bpsk", "qpsk", "16qam", "64qam"}

            # 5. Load and verify samples
            for idx in range(3):
                sample = loader.load_sample(f"sample_{idx:06d}.npy")
                assert sample["iq"].shape == (2, 512)
                assert sample["spectrogram"].shape == (224, 224)
