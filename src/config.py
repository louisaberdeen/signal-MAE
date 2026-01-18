"""
Configuration classes for AudioMAE++ models.

The Config class centralizes all hyperparameters for:
- Audio processing (sample rate, mel bins, etc.)
- Model architecture (layers, dimensions, heads)
- Training (learning rate, batch size, losses)
- Feature flags (Macaron, SwiGLU, RoPE)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class Config:
    """
    Configuration for AudioMAE models.

    This dataclass defines all hyperparameters for model architecture,
    audio processing, and training. Feature flags allow toggling
    between AudioMAE++ (advanced) and baseline (standard) variants.

    Attributes:
        sample_rate: Audio resampling rate in Hz
        n_mels: Number of mel spectrogram bins
        img_size: Spectrogram image size (square)
        patch_size: ViT patch size
        embed_dim: Encoder embedding dimension
        encoder_depth: Number of encoder transformer layers
        decoder_depth: Number of decoder transformer layers
        mask_ratio: Fraction of patches to mask during training
        use_macaron: Use Macaron-style transformer blocks
        use_swiglu: Use SwiGLU activation in FFN
        use_rope: Use Rotary Position Embeddings
    """

    # Audio processing
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    audio_duration: int = 5  # seconds

    # Spectrogram image size
    img_size: int = 224

    # Patch settings
    patch_size: int = 16

    # Model architecture - Encoder
    embed_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12

    # Model architecture - Decoder
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_heads: int = 16

    # MLP settings
    mlp_ratio: float = 4.0

    # Training
    mask_ratio: float = 0.75  # Mask 75% of patches
    batch_size: int = 16
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    epochs: int = 50
    warmup_epochs: int = 5
    checkpoint_interval: int = 5  # Save checkpoint every N epochs

    # AudioMAE++ feature flags
    use_macaron: bool = True
    use_swiglu: bool = True
    use_rope: bool = True

    # Embedding extraction
    pooling_mode: str = "mean"  # "cls", "mean", or "cls+mean"

    # Contrastive loss (improves semantic clustering)
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.01
    contrastive_temperature: float = 0.07

    # Uniformity loss (prevents embedding collapse)
    use_uniformity_loss: bool = True
    uniformity_weight: float = 0.1
    uniformity_t: float = 2.0

    @property
    def num_patches(self) -> int:
        """Calculate number of patches from image and patch size."""
        return (self.img_size // self.patch_size) ** 2

    @property
    def num_samples(self) -> int:
        """Calculate number of audio samples from duration and sample rate."""
        return self.sample_rate * self.audio_duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> 'Config':
        """Create config from JSON string."""
        return cls.from_dict(json.loads(s))

    def get_experiment_name(self) -> str:
        """Generate experiment name from key parameters."""
        features = []
        if self.use_macaron:
            features.append("macaron")
        if self.use_swiglu:
            features.append("swiglu")
        if self.use_rope:
            features.append("rope")

        feature_str = "-".join(features) if features else "baseline"
        return f"audiomae_{feature_str}_d{self.encoder_depth}_h{self.embed_dim}"

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )

        if self.embed_dim % self.encoder_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"encoder_heads ({self.encoder_heads})"
            )

        if self.decoder_embed_dim % self.decoder_heads != 0:
            raise ValueError(
                f"decoder_embed_dim ({self.decoder_embed_dim}) must be divisible by "
                f"decoder_heads ({self.decoder_heads})"
            )

        if not 0 <= self.mask_ratio < 1:
            raise ValueError(f"mask_ratio must be in [0, 1), got {self.mask_ratio}")

        if self.pooling_mode not in ("cls", "mean", "cls+mean"):
            raise ValueError(
                f"pooling_mode must be 'cls', 'mean', or 'cls+mean', "
                f"got '{self.pooling_mode}'"
            )

    def __post_init__(self):
        """Validate config after initialization."""
        self.validate()


def create_baseline_config() -> Config:
    """
    Create a baseline config (standard ViT-MAE, no advanced features).

    Returns:
        Config with Macaron, SwiGLU, RoPE disabled
    """
    return Config(
        use_macaron=False,
        use_swiglu=False,
        use_rope=False,
        use_contrastive_loss=False,
        use_uniformity_loss=False,
    )


def create_small_config() -> Config:
    """
    Create a small config for faster training/testing.

    Returns:
        Config with reduced model size
    """
    return Config(
        embed_dim=384,
        encoder_depth=6,
        encoder_heads=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_heads=8,
    )


def create_rf_config(size: str = "base", advanced: bool = False) -> Config:
    """
    Create configuration optimized for RF signal spectrograms.

    RF spectrograms from TorchSig are typically 224x224 images with
    time on one axis and frequency on the other.

    Args:
        size: Model size - "small", "base", or "tiny"
            - tiny: Minimal model for testing (256-dim, 4 layers)
            - small: Faster training (384-dim, 6 layers)
            - base: Full model (768-dim, 12 layers)
        advanced: If True, enable advanced features (Macaron, SwiGLU, RoPE)
            - False: Standard ViT-MAE (SignalMAE baseline)
            - True: Full AudioMAE++ features (SignalMAE++)

    Returns:
        Config optimized for RF signal processing

    Example:
        from src.config import create_rf_config

        # Baseline model (simple, for comparison)
        config = create_rf_config("base")

        # Advanced model with all features
        config = create_rf_config("base", advanced=True)

        # Small advanced model
        config = create_rf_config("small", advanced=True)
    """
    # Common RF settings
    rf_common = {
        "img_size": 224,
        "patch_size": 16,
        "mask_ratio": 0.75,
        # Advanced features controlled by parameter
        "use_macaron": advanced,
        "use_swiglu": advanced,
        "use_rope": advanced,
        # Contrastive losses (can help with clustering)
        "use_contrastive_loss": advanced,
        "use_uniformity_loss": advanced,
        # Use mean pooling for embeddings
        "pooling_mode": "mean",
    }

    sizes = {
        "tiny": Config(
            embed_dim=256,
            encoder_depth=4,
            encoder_heads=4,
            decoder_embed_dim=128,
            decoder_depth=2,
            decoder_heads=4,
            mlp_ratio=4.0,
            batch_size=32,
            learning_rate=1e-4,
            epochs=20,
            warmup_epochs=2,
            **rf_common,
        ),
        "small": Config(
            embed_dim=384,
            encoder_depth=6,
            encoder_heads=6,
            decoder_embed_dim=256,
            decoder_depth=4,
            decoder_heads=8,
            mlp_ratio=4.0,
            batch_size=32,
            learning_rate=1.5e-4,
            epochs=50,
            warmup_epochs=5,
            **rf_common,
        ),
        "base": Config(
            embed_dim=768,
            encoder_depth=12,
            encoder_heads=12,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_heads=16,
            mlp_ratio=4.0,
            batch_size=16,
            learning_rate=1.5e-4,
            epochs=100,
            warmup_epochs=10,
            **rf_common,
        ),
    }

    if size not in sizes:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(sizes.keys())}")

    return sizes[size]
