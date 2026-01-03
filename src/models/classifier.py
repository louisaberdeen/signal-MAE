"""
Classification wrapper for fine-tuning pretrained encoders.

AudioMAEClassifier wraps a pretrained encoder with a classification
head for downstream tasks like ESC-50 classification.
"""

import torch
import torch.nn as nn
from typing import Optional

from src.config import Config
from src.models.base import BaseClassifier
from src.models.audiomae import AudioMAEPlusPlus
from src.models.baseline import BaselineMAE


# Note: Classifier is NOT registered in model_registry because it has a
# different interface - it wraps pretrained models rather than being
# instantiated from a Config like autoencoders.
class AudioMAEClassifier(BaseClassifier):
    """
    Classification head on top of AudioMAE encoder.

    Can be used with either AudioMAE++ or BaselineMAE pretrained models.
    Supports freezing the encoder for linear probing or full fine-tuning.

    Args:
        pretrained_model: Pretrained AudioMAE model (encoder will be extracted)
        num_classes: Number of output classes
        freeze_encoder: If True, freeze encoder weights (linear probing)
    """

    def __init__(
        self,
        pretrained_model: AudioMAEPlusPlus,
        num_classes: int,
        freeze_encoder: bool = False
    ):
        super().__init__()

        # Handle BaselineMAE wrapper
        if isinstance(pretrained_model, BaselineMAE):
            # BaselineMAE is a subclass, use it directly
            pass

        self.config = pretrained_model.config
        self._num_classes = num_classes

        # Copy encoder components
        self.patch_embed = pretrained_model.patch_embed
        self.cls_token = pretrained_model.cls_token
        self.pos_embed = pretrained_model.pos_embed
        self.encoder_blocks = pretrained_model.encoder_blocks
        self.encoder_norm = pretrained_model.encoder_norm

        # Freeze encoder if specified
        if freeze_encoder:
            self.freeze_encoder()

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, num_classes)
        )

    @property
    def num_classes(self) -> int:
        """Return number of output classes."""
        return self._num_classes

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for linear probing."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.encoder_blocks.parameters():
            param.requires_grad = False
        for param in self.encoder_norm.parameters():
            param.requires_grad = False
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for full fine-tuning."""
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        for param in self.encoder_blocks.parameters():
            param.requires_grad = True
        for param in self.encoder_norm.parameters():
            param.requires_grad = True
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = True
        self.cls_token.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Classification forward pass.

        Args:
            x: Input images [B, 3, H, W]
            mask_ratio: Optional masking for regularization during training

        Returns:
            Logits [B, num_classes]
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Add position embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:, :]

        # Optional masking during fine-tuning (regularization)
        if mask_ratio > 0 and self.training:
            B, N, D = x.shape
            len_keep = int(N * (1 - mask_ratio))
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x = torch.gather(
                x, dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
            )

        # Add CLS token
        cls_token = self.cls_token
        if self.pos_embed is not None:
            cls_token = cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        # Classification from CLS token
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits


def create_classifier(
    pretrained_model: AudioMAEPlusPlus,
    num_classes: int,
    freeze_encoder: bool = False
) -> AudioMAEClassifier:
    """
    Create a classifier from a pretrained model.

    Args:
        pretrained_model: Pretrained encoder model
        num_classes: Number of output classes
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        AudioMAEClassifier instance
    """
    return AudioMAEClassifier(pretrained_model, num_classes, freeze_encoder)
