"""
Abstract base classes for model plugins.

This module defines the interfaces that all model plugins must implement:
- BaseModel: Core encoder interface
- BaseAutoencoder: Masked autoencoder interface
- BaseClassifier: Classification wrapper interface

All models registered with model_registry should inherit from one of these.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all model plugins.

    Defines the core encoder interface that all models must implement.
    This is the minimal interface required for embedding extraction.

    Subclasses must implement:
        - forward_encoder: Encode input with optional masking
        - get_embedding: Extract embedding vector
        - embed_dim: Property returning embedding dimension
        - num_patches: Property returning number of patches
    """

    @abstractmethod
    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input with optional masking.

        Args:
            x: Input tensor [B, C, H, W]
            mask_ratio: Fraction of patches to mask (0.0 for inference)

        Returns:
            latent: Encoder output [B, N+1, D] (includes CLS token)
            mask: Binary mask [B, N] (1=masked, 0=visible)
            ids_restore: Indices to restore original order [B, N]
        """
        pass

    @abstractmethod
    def get_embedding(
        self,
        x: torch.Tensor,
        pooling_mode: str = "mean"
    ) -> torch.Tensor:
        """
        Extract embedding vector from input.

        Args:
            x: Input tensor [B, C, H, W]
            pooling_mode: "cls" (CLS token), "mean" (mean of patches),
                         or "cls+mean" (concatenation)

        Returns:
            Embedding [B, D] or [B, 2*D] for cls+mean
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Return embedding dimension."""
        pass

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """Return number of patches."""
        pass

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count model parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class BaseAutoencoder(BaseModel):
    """
    Abstract base class for masked autoencoder models.

    Extends BaseModel with decoder and reconstruction loss interfaces.
    Used for self-supervised pre-training with masked reconstruction.

    Subclasses must implement:
        - All methods from BaseModel
        - forward_decoder: Decode encoded representation
        - forward_loss: Compute reconstruction loss
        - patchify: Convert images to patches
        - unpatchify: Convert patches back to images
    """

    @abstractmethod
    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode encoded representation.

        Args:
            x: Encoder output [B, N_vis+1, D] (visible patches + CLS)
            ids_restore: Indices to restore original patch order [B, N]

        Returns:
            Reconstruction [B, N, patch_pixels]
        """
        pass

    @abstractmethod
    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss on masked patches.

        Args:
            imgs: Original images [B, C, H, W]
            pred: Predicted patch pixels [B, N, patch_pixels]
            mask: Binary mask [B, N] (1=masked, 0=visible)

        Returns:
            Scalar loss value
        """
        pass

    @abstractmethod
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.

        Args:
            imgs: Images [B, C, H, W]

        Returns:
            Patches [B, N, patch_size^2 * C]
        """
        pass

    @abstractmethod
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images.

        Args:
            x: Patches [B, N, patch_size^2 * C]

        Returns:
            Images [B, C, H, W]
        """
        pass

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: Optional[float] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass with masking, encoding, decoding, and loss.

        Args:
            imgs: Input images [B, C, H, W]
            mask_ratio: Fraction to mask (uses config default if None)
            labels: Optional class labels for contrastive loss

        Returns:
            loss: Reconstruction loss (scalar)
            pred: Predicted patches [B, N, patch_pixels]
            mask: Binary mask [B, N]
        """
        # Default implementation - subclasses may override
        raise NotImplementedError("Subclasses must implement forward()")


class BaseClassifier(nn.Module, ABC):
    """
    Abstract base class for classification wrapper models.

    Wraps an encoder model with a classification head for
    fine-tuning on downstream tasks.

    Subclasses must implement:
        - forward: Classification forward pass
        - num_classes: Property returning number of output classes
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Classification forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            mask_ratio: Optional masking for regularization during training

        Returns:
            Logits [B, num_classes]
        """
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return number of output classes."""
        pass

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for linear probing."""
        raise NotImplementedError("Subclasses must implement freeze_encoder()")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for full fine-tuning."""
        raise NotImplementedError("Subclasses must implement unfreeze_encoder()")
