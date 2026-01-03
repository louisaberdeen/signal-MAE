"""
Baseline MAE - Standard Masked Autoencoder for comparison.

This model uses standard transformer blocks without the advanced
features of AudioMAE++ (no Macaron, no SwiGLU, no RoPE).
"""

import torch
from typing import Tuple, Optional

from src.registry import model_registry
from src.config import Config
from src.models.audiomae import AudioMAEPlusPlus


@model_registry.register("baseline", version="2.0")
class BaselineMAE(AudioMAEPlusPlus):
    """
    Baseline Masked Autoencoder with standard transformer blocks.

    This is a wrapper around AudioMAEPlusPlus that disables all
    advanced features (Macaron, SwiGLU, RoPE) for comparison.

    Args:
        config: Configuration object (advanced features will be disabled)
    """

    def __init__(self, config: Config):
        # Create a modified config with baseline settings
        baseline_config = Config(
            # Copy core architecture from input config
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
            # Disable advanced features
            use_macaron=False,
            use_swiglu=False,
            use_rope=False,
            # Optionally disable extra losses for fair comparison
            use_contrastive_loss=config.use_contrastive_loss,
            use_uniformity_loss=config.use_uniformity_loss,
            contrastive_weight=config.contrastive_weight,
            contrastive_temperature=config.contrastive_temperature,
            uniformity_weight=config.uniformity_weight,
            uniformity_t=config.uniformity_t,
        )

        super().__init__(baseline_config)
        self._original_config = config  # Keep reference to original

    @property
    def original_config(self) -> Config:
        """Return the original (non-modified) config."""
        return self._original_config
