"""
SignalMAE - Masked Autoencoder for RF Signal Spectrograms.

A simple baseline model for RF signal representation learning using
masked autoencoder pre-training on spectrograms. This model reuses
the standard ViT-MAE architecture from BaselineMAE for simplicity.

The model processes RF spectrograms (generated from IQ data via TorchSig)
and learns representations through masked patch reconstruction.

References:
    - MAE: https://arxiv.org/abs/2111.06377
    - AudioMAE: https://arxiv.org/abs/2203.16691
    - MSM-MAE: https://github.com/nttcslab/msm-mae
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from src.registry import model_registry
from src.config import Config
from src.models.baseline import BaselineMAE
from src.models.audiomae import AudioMAEPlusPlus
from src.models.base import BaseClassifier


@model_registry.register("signalmae", version="1.0")
class SignalMAE(BaselineMAE):
    """
    Masked Autoencoder for RF signal spectrograms.

    This model uses the standard ViT-MAE architecture (no Macaron/SwiGLU/RoPE)
    to learn representations from RF spectrograms through masked reconstruction.
    It serves as a simple baseline for RF signal processing tasks.

    The architecture is intentionally simple:
    - Standard transformer blocks with GELU activation
    - Learned position embeddings
    - Asymmetric encoder-decoder (larger encoder, smaller decoder)
    - 75% masking ratio during pre-training

    Input: RF spectrograms [B, 3, 224, 224]
        - Generated from IQ data using TorchSig + IQToSpectrogram transform
        - 3-channel RGB spectrograms (replicated for ViT compatibility)

    Output (pre-training): Reconstruction loss, predictions, mask
    Output (inference): Embeddings [B, 768] or [B, 1536] for cls+mean

    Example:
        from src.models.signalmae import SignalMAE
        from src.config import create_rf_config

        # Create model
        config = create_rf_config("base")
        model = SignalMAE(config)

        # Pre-training
        spectrograms = torch.randn(16, 3, 224, 224)
        loss, pred, mask = model(spectrograms, mask_ratio=0.75)

        # Inference
        embeddings = model.get_embedding(spectrograms, pooling_mode="mean")

    Args:
        config: Configuration object with model hyperparameters
    """

    def __init__(self, config: Config):
        """
        Initialize SignalMAE.

        Args:
            config: Configuration object. Advanced features (Macaron, SwiGLU, RoPE)
                    will be automatically disabled by the parent BaselineMAE class.
        """
        super().__init__(config)

        # Store RF-specific info
        self._signal_type = "rf_spectrogram"

    @property
    def signal_type(self) -> str:
        """Type of signal this model processes."""
        return self._signal_type

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging/debugging.

        Returns:
            Dictionary with model configuration and statistics
        """
        return {
            "model_name": "SignalMAE",
            "version": "1.0",
            "signal_type": self.signal_type,
            "embed_dim": self.embed_dim,
            "num_patches": self.num_patches,
            "encoder_depth": self.config.encoder_depth,
            "decoder_depth": self.config.decoder_depth,
            "mask_ratio": self.config.mask_ratio,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "num_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "uses_macaron": False,
            "uses_swiglu": False,
            "uses_rope": False,
        }


@model_registry.register("signalmae++", version="1.0")
class SignalMAEPlusPlus(AudioMAEPlusPlus):
    """
    Advanced Masked Autoencoder for RF signal spectrograms.

    This model uses the full AudioMAE++ architecture with all advanced features:
    - Macaron-style transformer blocks (FFN → Attention → FFN sandwich)
    - SwiGLU activation (gated linear unit with Swish)
    - RoPE (Rotary Position Embeddings) for better generalization

    Use this when you want maximum model capacity and have sufficient
    compute resources. For a simpler baseline, use SignalMAE instead.

    Input: RF spectrograms [B, 3, 224, 224]
        - Generated from IQ data using TorchSig + IQToSpectrogram transform

    Example:
        from src.models.signalmae import SignalMAEPlusPlus
        from src.config import create_rf_config

        # Create model with advanced features
        config = create_rf_config("base", advanced=True)
        model = SignalMAEPlusPlus(config)

        # Pre-training
        loss, pred, mask = model(spectrograms, mask_ratio=0.75)

        # Embedding extraction
        embeddings = model.get_embedding(spectrograms, pooling_mode="mean")

    Args:
        config: Configuration object with model hyperparameters.
                Advanced features should be enabled in config.
    """

    def __init__(self, config: Config):
        """
        Initialize SignalMAE++.

        Args:
            config: Configuration object. For full advanced features, use
                    create_rf_config(size, advanced=True).
        """
        # Ensure advanced features are enabled
        advanced_config = Config(
            # Copy all settings from input config
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            audio_duration=config.audio_duration,
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_heads=config.encoder_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_heads=config.decoder_heads,
            mlp_ratio=config.mlp_ratio,
            mask_ratio=config.mask_ratio,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            pooling_mode=config.pooling_mode,
            # Enable ALL advanced features
            use_macaron=True,
            use_swiglu=True,
            use_rope=True,
            # Enable contrastive losses for better embeddings
            use_contrastive_loss=config.use_contrastive_loss,
            use_uniformity_loss=config.use_uniformity_loss,
            contrastive_weight=config.contrastive_weight,
            contrastive_temperature=config.contrastive_temperature,
            uniformity_weight=config.uniformity_weight,
            uniformity_t=config.uniformity_t,
        )

        super().__init__(advanced_config)
        self._signal_type = "rf_spectrogram"

    @property
    def signal_type(self) -> str:
        """Type of signal this model processes."""
        return self._signal_type

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging/debugging.

        Returns:
            Dictionary with model configuration and statistics
        """
        return {
            "model_name": "SignalMAE++",
            "version": "1.0",
            "signal_type": self.signal_type,
            "embed_dim": self.embed_dim,
            "num_patches": self.num_patches,
            "encoder_depth": self.config.encoder_depth,
            "decoder_depth": self.config.decoder_depth,
            "mask_ratio": self.config.mask_ratio,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "num_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "uses_macaron": True,
            "uses_swiglu": True,
            "uses_rope": True,
        }


@model_registry.register("signalmae-small", version="1.0")
class SignalMAESmall(SignalMAE):
    """
    Small variant of SignalMAE for faster training and lower memory usage.

    Reduced dimensions:
    - Encoder: 384-dim, 6 layers, 6 heads
    - Decoder: 256-dim, 4 layers, 8 heads

    Suitable for:
    - Quick experiments and prototyping
    - Limited GPU memory
    - Smaller datasets

    Example:
        from src.models.signalmae import SignalMAESmall
        from src.config import create_rf_config

        config = create_rf_config("small")
        model = SignalMAESmall(config)
    """

    def __init__(self, config: Config):
        # Override to small dimensions
        small_config = Config(
            # Keep RF/spectrogram settings
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            audio_duration=config.audio_duration,
            img_size=config.img_size,
            patch_size=config.patch_size,
            # Small architecture
            embed_dim=384,
            encoder_depth=6,
            encoder_heads=6,
            decoder_embed_dim=256,
            decoder_depth=4,
            decoder_heads=8,
            mlp_ratio=config.mlp_ratio,
            mask_ratio=config.mask_ratio,
            # Training
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            pooling_mode=config.pooling_mode,
            # Disable advanced features
            use_macaron=False,
            use_swiglu=False,
            use_rope=False,
            # Disable extra losses for simplicity
            use_contrastive_loss=False,
            use_uniformity_loss=False,
        )
        # Call grandparent (AudioMAEPlusPlus) directly with small config
        from src.models.audiomae import AudioMAEPlusPlus
        AudioMAEPlusPlus.__init__(self, small_config)
        self._signal_type = "rf_spectrogram"


class SignalMAEClassifier(BaseClassifier):
    """
    Classification wrapper for SignalMAE.

    Wraps a pre-trained SignalMAE encoder for fine-tuning on
    modulation classification or other RF signal classification tasks.

    The classifier adds a linear head on top of the encoder embeddings
    and supports both frozen encoder (linear probing) and full fine-tuning.

    Example:
        from src.models.signalmae import SignalMAE, SignalMAEClassifier
        from src.config import create_rf_config

        # Load pre-trained encoder
        config = create_rf_config("base")
        encoder = SignalMAE(config)
        encoder.load_state_dict(torch.load("pretrained.pt"))

        # Create classifier for 12 modulation types
        classifier = SignalMAEClassifier(
            encoder=encoder,
            num_classes=12,
            freeze_encoder=True  # Linear probing
        )

        # Fine-tune
        logits = classifier(spectrograms)  # [B, 12]

        # Later: unfreeze for full fine-tuning
        classifier.unfreeze_encoder()

    Args:
        encoder: Pre-trained SignalMAE model
        num_classes: Number of classification classes (e.g., modulation types)
        freeze_encoder: If True, freeze encoder weights (linear probing)
        pooling_mode: How to pool encoder outputs ("cls", "mean", "cls+mean")
        dropout: Dropout probability before classification head
    """

    def __init__(
        self,
        encoder: SignalMAE,
        num_classes: int,
        freeze_encoder: bool = True,
        pooling_mode: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = encoder
        self._num_classes = num_classes
        self._pooling_mode = pooling_mode

        # Determine input dimension based on pooling mode
        embed_dim = encoder.embed_dim
        if pooling_mode == "cls+mean":
            input_dim = embed_dim * 2
        else:
            input_dim = embed_dim

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

        # Optionally freeze encoder
        if freeze_encoder:
            self.freeze_encoder()

    @property
    def num_classes(self) -> int:
        """Number of classification classes."""
        return self._num_classes

    @property
    def pooling_mode(self) -> str:
        """Embedding pooling mode."""
        return self._pooling_mode

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters (for linear probing)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters (for fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input spectrograms [B, C, H, W]
            mask_ratio: Masking ratio (typically 0 for inference)

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Get embeddings from encoder
        embeddings = self.encoder.get_embedding(x, pooling_mode=self._pooling_mode)

        # Classification
        logits = self.head(embeddings)

        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings without classification.

        Args:
            x: Input spectrograms [B, C, H, W]

        Returns:
            embeddings: [B, embed_dim] or [B, 2*embed_dim] for cls+mean
        """
        return self.encoder.get_embedding(x, pooling_mode=self._pooling_mode)


def create_signalmae(
    config: Optional[Config] = None,
    size: str = "base",
    num_classes: Optional[int] = None,
    pretrained_path: Optional[str] = None,
) -> SignalMAE:
    """
    Factory function to create SignalMAE models.

    Args:
        config: Optional configuration (if None, uses default for size)
        size: Model size ("small" or "base")
        num_classes: If provided, returns classifier wrapper
        pretrained_path: Path to pretrained weights

    Returns:
        SignalMAE model or SignalMAEClassifier

    Example:
        # Base model for pre-training
        model = create_signalmae(size="base")

        # Small model for classification
        classifier = create_signalmae(size="small", num_classes=12)

        # Load pretrained
        model = create_signalmae(pretrained_path="checkpoints/signalmae.pt")
    """
    from src.config import create_rf_config

    # Get config
    if config is None:
        config = create_rf_config(size)

    # Create model
    if size == "small":
        model = SignalMAESmall(config)
    else:
        model = SignalMAE(config)

    # Load pretrained weights if provided
    if pretrained_path is not None:
        import torch
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict)

    # Wrap with classifier if num_classes provided
    if num_classes is not None:
        model = SignalMAEClassifier(
            encoder=model,
            num_classes=num_classes,
            freeze_encoder=True,
        )

    return model
