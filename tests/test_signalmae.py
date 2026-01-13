"""
Comprehensive tests for SignalMAE model and training pipeline.

Tests cover:
- Model instantiation and configuration
- Forward pass and output shapes
- Masking behavior
- Embedding extraction
- Classifier wrapper
- Training pipeline
- Integration with TorchSig data
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config import Config, create_rf_config, create_baseline_config
from src.registry import model_registry


class TestRFConfig:
    """Tests for RF-specific configuration."""

    def test_create_rf_config_tiny(self):
        """Test tiny RF config creation."""
        config = create_rf_config("tiny")

        assert config.embed_dim == 256
        assert config.encoder_depth == 4
        assert config.decoder_depth == 2
        assert config.use_macaron is False
        assert config.use_swiglu is False
        assert config.use_rope is False

    def test_create_rf_config_small(self):
        """Test small RF config creation."""
        config = create_rf_config("small")

        assert config.embed_dim == 384
        assert config.encoder_depth == 6
        assert config.decoder_depth == 4
        assert config.batch_size == 32

    def test_create_rf_config_base(self):
        """Test base RF config creation."""
        config = create_rf_config("base")

        assert config.embed_dim == 768
        assert config.encoder_depth == 12
        assert config.decoder_depth == 8
        assert config.batch_size == 16

    def test_create_rf_config_invalid_size(self):
        """Test invalid size raises error."""
        with pytest.raises(ValueError, match="Unknown size"):
            create_rf_config("xlarge")

    def test_rf_config_disables_advanced_features(self):
        """Test RF config disables advanced features."""
        config = create_rf_config("base")

        assert config.use_macaron is False
        assert config.use_swiglu is False
        assert config.use_rope is False
        assert config.use_contrastive_loss is False
        assert config.use_uniformity_loss is False

    def test_rf_config_uses_mean_pooling(self):
        """Test RF config uses mean pooling by default."""
        config = create_rf_config("base")
        assert config.pooling_mode == "mean"


class TestSignalMAE:
    """Tests for SignalMAE model."""

    @pytest.fixture
    def tiny_config(self):
        """Create tiny config for fast testing."""
        return create_rf_config("tiny")

    @pytest.fixture
    def model(self, tiny_config):
        """Create SignalMAE model."""
        from src.models.signalmae import SignalMAE
        return SignalMAE(tiny_config)

    def test_model_registration(self):
        """Test SignalMAE is registered in model registry."""
        assert "signalmae" in model_registry
        assert "signalmae-small" in model_registry

    def test_model_creation_via_registry(self, tiny_config):
        """Test creating model through registry."""
        model = model_registry.create("signalmae", tiny_config)
        assert model is not None

    def test_model_instantiation(self, model):
        """Test model can be instantiated."""
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'get_embedding')

    def test_model_forward_pretraining(self, model):
        """Test forward pass with masking (pre-training)."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        loss, pred, mask = model(x, mask_ratio=0.75)

        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

        # Check predictions shape
        # pred should be [B, num_patches, patch_size^2 * 3]
        assert pred.shape[0] == batch_size
        assert pred.shape[1] == model.num_patches  # 196

        # Check mask shape
        assert mask.shape == (batch_size, model.num_patches)

    def test_model_forward_inference(self, model):
        """Test forward pass without masking (inference)."""
        x = torch.randn(4, 3, 224, 224)

        loss, pred, mask = model(x, mask_ratio=0.0)

        # With no masking, mask should be all zeros
        assert mask.sum() == 0
        # Loss should still be computable
        assert isinstance(loss, torch.Tensor)

    def test_forward_encoder(self, model):
        """Test encoder forward pass."""
        x = torch.randn(4, 3, 224, 224)

        latent, mask, ids_restore = model.forward_encoder(x, mask_ratio=0.0)

        # latent: [B, num_patches+1, embed_dim]
        assert latent.shape == (4, 197, model.embed_dim)

    def test_forward_encoder_with_masking(self, model):
        """Test encoder with masking."""
        x = torch.randn(4, 3, 224, 224)
        mask_ratio = 0.75

        latent, mask, ids_restore = model.forward_encoder(x, mask_ratio=mask_ratio)

        # With 75% masking, we keep 25% of patches = 49
        expected_visible = int(196 * (1 - mask_ratio))
        assert latent.shape[1] == expected_visible + 1  # +1 for CLS token

    def test_get_embedding_cls(self, model):
        """Test CLS token embedding extraction."""
        x = torch.randn(4, 3, 224, 224)

        emb = model.get_embedding(x, pooling_mode="cls")

        assert emb.shape == (4, model.embed_dim)

    def test_get_embedding_mean(self, model):
        """Test mean pooling embedding extraction."""
        x = torch.randn(4, 3, 224, 224)

        emb = model.get_embedding(x, pooling_mode="mean")

        assert emb.shape == (4, model.embed_dim)

    def test_get_embedding_cls_mean(self, model):
        """Test CLS+mean concatenated embedding."""
        x = torch.randn(4, 3, 224, 224)

        emb = model.get_embedding(x, pooling_mode="cls+mean")

        # Should be double the embed_dim
        assert emb.shape == (4, model.embed_dim * 2)

    def test_model_info(self, model):
        """Test get_model_info method."""
        info = model.get_model_info()

        assert info["model_name"] == "SignalMAE"
        assert info["version"] == "1.0"
        assert info["signal_type"] == "rf_spectrogram"
        assert info["uses_macaron"] is False
        assert info["uses_swiglu"] is False
        assert "num_parameters" in info

    def test_signal_type_property(self, model):
        """Test signal_type property."""
        assert model.signal_type == "rf_spectrogram"

    def test_num_patches_property(self, model):
        """Test num_patches property."""
        # 224 / 16 = 14, 14^2 = 196
        assert model.num_patches == 196

    def test_embed_dim_property(self, model, tiny_config):
        """Test embed_dim property."""
        assert model.embed_dim == tiny_config.embed_dim


class TestSignalMAESmall:
    """Tests for SignalMAESmall variant."""

    def test_small_model_dimensions(self):
        """Test small model has correct dimensions."""
        from src.models.signalmae import SignalMAESmall

        config = create_rf_config("small")
        model = SignalMAESmall(config)

        assert model.embed_dim == 384

    def test_small_model_registry(self):
        """Test small model is registered."""
        config = create_rf_config("small")
        model = model_registry.create("signalmae-small", config)

        assert model is not None
        assert model.embed_dim == 384


class TestSignalMAEClassifier:
    """Tests for SignalMAE classification wrapper."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for classifier."""
        from src.models.signalmae import SignalMAE
        config = create_rf_config("tiny")
        return SignalMAE(config)

    @pytest.fixture
    def classifier(self, encoder):
        """Create classifier."""
        from src.models.signalmae import SignalMAEClassifier
        return SignalMAEClassifier(
            encoder=encoder,
            num_classes=12,
            freeze_encoder=True,
        )

    def test_classifier_creation(self, classifier):
        """Test classifier can be created."""
        assert classifier is not None
        assert classifier.num_classes == 12

    def test_classifier_forward(self, classifier):
        """Test classifier forward pass."""
        x = torch.randn(4, 3, 224, 224)

        logits = classifier(x)

        assert logits.shape == (4, 12)

    def test_classifier_freeze_encoder(self, classifier):
        """Test encoder is frozen by default."""
        for param in classifier.encoder.parameters():
            assert param.requires_grad is False

        # Head should be trainable
        for param in classifier.head.parameters():
            assert param.requires_grad is True

    def test_classifier_unfreeze(self, classifier):
        """Test unfreezing encoder."""
        classifier.unfreeze_encoder()

        for param in classifier.encoder.parameters():
            assert param.requires_grad is True

    def test_classifier_get_embeddings(self, classifier):
        """Test embedding extraction through classifier."""
        x = torch.randn(4, 3, 224, 224)

        emb = classifier.get_embeddings(x)

        assert emb.shape[0] == 4
        assert emb.ndim == 2

    def test_classifier_with_cls_mean_pooling(self, encoder):
        """Test classifier with cls+mean pooling."""
        from src.models.signalmae import SignalMAEClassifier

        classifier = SignalMAEClassifier(
            encoder=encoder,
            num_classes=12,
            pooling_mode="cls+mean",
        )

        x = torch.randn(4, 3, 224, 224)
        logits = classifier(x)

        assert logits.shape == (4, 12)


class TestCreateSignalMAE:
    """Tests for create_signalmae factory function."""

    def test_create_base_model(self):
        """Test creating base model."""
        from src.models.signalmae import create_signalmae

        model = create_signalmae(size="base")
        assert model.embed_dim == 768

    def test_create_small_model(self):
        """Test creating small model."""
        from src.models.signalmae import create_signalmae

        model = create_signalmae(size="small")
        assert model.embed_dim == 384

    def test_create_classifier(self):
        """Test creating classifier directly."""
        from src.models.signalmae import create_signalmae, SignalMAEClassifier

        classifier = create_signalmae(size="tiny", num_classes=10)

        assert isinstance(classifier, SignalMAEClassifier)
        assert classifier.num_classes == 10


class TestRFSpectrogramDataset:
    """Tests for RF spectrogram dataset."""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset with spectrograms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create spectrogram files
            for i in range(10):
                spec = np.random.rand(224, 224).astype(np.float32)
                np.save(tmpdir / f"sample_{i:04d}.npy", spec)

            yield tmpdir

    def test_dataset_loading(self, temp_dataset):
        """Test loading dataset from directory."""
        from src.training.rf_trainer import RFSpectrogramDataset

        dataset = RFSpectrogramDataset(temp_dataset)

        assert len(dataset) == 10

    def test_dataset_getitem(self, temp_dataset):
        """Test getting item from dataset."""
        from src.training.rf_trainer import RFSpectrogramDataset

        dataset = RFSpectrogramDataset(temp_dataset)

        item = dataset[0]

        # Returns tuple with tensor
        assert isinstance(item, tuple)
        assert item[0].shape == (3, 224, 224)

    def test_dataset_with_metadata(self, temp_dataset):
        """Test loading dataset with metadata."""
        import pandas as pd
        from src.training.rf_trainer import RFSpectrogramDataset

        # Create metadata CSV
        df = pd.DataFrame({
            'filename': [f"sample_{i:04d}.npy" for i in range(10)],
            'label': ['bpsk'] * 5 + ['qpsk'] * 5,
        })
        metadata_path = temp_dataset / "metadata.csv"
        df.to_csv(metadata_path, index=False)

        dataset = RFSpectrogramDataset(
            temp_dataset,
            metadata_csv=metadata_path,
            return_labels=True,
        )

        assert dataset.num_classes == 2

        tensor, label = dataset[0]
        assert tensor.shape == (3, 224, 224)
        assert isinstance(label, int)


class TestSignalMAETrainer:
    """Tests for SignalMAE training pipeline."""

    @pytest.fixture
    def model(self):
        """Create tiny model for training tests."""
        from src.models.signalmae import SignalMAE
        config = create_rf_config("tiny")
        return SignalMAE(config)

    @pytest.fixture
    def train_loader(self):
        """Create dummy training data loader."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create random data
        data = torch.randn(32, 3, 224, 224)
        dataset = TensorDataset(data)

        return DataLoader(dataset, batch_size=8, shuffle=True)

    def test_trainer_creation(self, model):
        """Test trainer can be created."""
        from src.training.rf_trainer import SignalMAETrainer

        config = create_rf_config("tiny")
        trainer = SignalMAETrainer(model, config, device="cpu")

        assert trainer is not None

    def test_pretrain_epoch(self, model, train_loader):
        """Test single pre-training epoch."""
        from src.training.rf_trainer import SignalMAETrainer

        config = create_rf_config("tiny")
        trainer = SignalMAETrainer(model, config, device="cpu")

        metrics = trainer.pretrain_epoch(train_loader, epoch=0)

        assert 'loss' in metrics
        assert 'lr' in metrics
        assert metrics['loss'] > 0

    def test_validation(self, model, train_loader):
        """Test validation."""
        from src.training.rf_trainer import SignalMAETrainer

        config = create_rf_config("tiny")
        trainer = SignalMAETrainer(model, config, device="cpu")

        metrics = trainer.validate(train_loader)

        assert 'val_loss' in metrics


class TestSignalMAEClassifierTrainer:
    """Tests for classifier fine-tuning."""

    @pytest.fixture
    def classifier(self):
        """Create classifier for training."""
        from src.models.signalmae import SignalMAE, SignalMAEClassifier

        config = create_rf_config("tiny")
        encoder = SignalMAE(config)

        return SignalMAEClassifier(
            encoder=encoder,
            num_classes=4,
            freeze_encoder=False,
        )

    @pytest.fixture
    def train_loader(self):
        """Create labeled training data."""
        from torch.utils.data import DataLoader, TensorDataset

        data = torch.randn(32, 3, 224, 224)
        labels = torch.randint(0, 4, (32,))
        dataset = TensorDataset(data, labels)

        return DataLoader(dataset, batch_size=8, shuffle=True)

    def test_classifier_trainer_creation(self, classifier):
        """Test classifier trainer creation."""
        from src.training.rf_trainer import SignalMAEClassifierTrainer

        config = create_rf_config("tiny")
        trainer = SignalMAEClassifierTrainer(classifier, config, device="cpu")

        assert trainer is not None

    def test_finetune_epoch(self, classifier, train_loader):
        """Test single fine-tuning epoch."""
        from src.training.rf_trainer import SignalMAEClassifierTrainer

        config = create_rf_config("tiny")
        trainer = SignalMAEClassifierTrainer(classifier, config, device="cpu")

        metrics = trainer.finetune_epoch(train_loader, epoch=0)

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_validate_classification(self, classifier, train_loader):
        """Test classification validation."""
        from src.training.rf_trainer import SignalMAEClassifierTrainer

        config = create_rf_config("tiny")
        trainer = SignalMAEClassifierTrainer(classifier, config, device="cpu")

        metrics = trainer.validate_classification(train_loader)

        assert 'val_loss' in metrics
        assert 'val_accuracy' in metrics


class TestIntegration:
    """Integration tests for SignalMAE with TorchSig."""

    def test_full_pipeline_synthetic(self):
        """Test full pipeline with synthetic data."""
        from src.models.signalmae import SignalMAE, SignalMAEClassifier
        from src.training.rf_trainer import SignalMAETrainer
        from torch.utils.data import DataLoader, TensorDataset

        # 1. Create model
        config = create_rf_config("tiny")
        model = SignalMAE(config)

        # 2. Create synthetic data
        train_data = torch.randn(16, 3, 224, 224)
        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=8)

        # 3. Pre-train for 1 epoch
        config.epochs = 1
        trainer = SignalMAETrainer(model, config, device="cpu")
        metrics = trainer.pretrain_epoch(train_loader, epoch=0)

        assert metrics['loss'] > 0

        # 4. Create classifier
        classifier = SignalMAEClassifier(
            encoder=model,
            num_classes=4,
            freeze_encoder=True,
        )

        # 5. Test classification forward
        logits = classifier(train_data[:4])
        assert logits.shape == (4, 4)

    def test_model_save_load(self):
        """Test saving and loading model."""
        from src.models.signalmae import SignalMAE

        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_rf_config("tiny")
            model = SignalMAE(config)

            # Save
            save_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), save_path)

            # Load into new model
            model2 = SignalMAE(config)
            model2.load_state_dict(torch.load(save_path))

            # Compare outputs
            x = torch.randn(2, 3, 224, 224)
            emb1 = model.get_embedding(x, pooling_mode="mean")
            emb2 = model2.get_embedding(x, pooling_mode="mean")

            assert torch.allclose(emb1, emb2)
