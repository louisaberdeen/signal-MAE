"""
AudioMAE++ - Masked Autoencoder with modern transformer features.

This is the main model implementation with:
- Macaron-style transformer blocks
- SwiGLU activation
- Optional Rotary Position Embeddings (RoPE)
- Asymmetric encoder-decoder architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from src.registry import model_registry
from src.config import Config
from src.models.base import BaseAutoencoder
from src.models.blocks.transformer import (
    StandardTransformerBlock,
    MacaronTransformerBlock,
    PatchEmbed
)
from src.training.losses import get_embedding, info_nce_loss, uniformity_loss


@model_registry.register("audiomae++", version="2.0")
class AudioMAEPlusPlus(BaseAutoencoder):
    """
    AudioMAE++ - Masked Autoencoder with modern features.

    Architecture:
    - Patch embedding with configurable patch size
    - Optional learned or rotary position embeddings
    - Encoder with Macaron or Standard transformer blocks
    - Asymmetric decoder (smaller than encoder)
    - Reconstruction head for masked patches

    Training:
    - Self-supervised with masked reconstruction (75% masking)
    - Optional contrastive and uniformity losses for better embeddings

    Args:
        config: Configuration object with model hyperparameters
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._embed_dim = config.embed_dim
        self._num_patches = config.num_patches

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.embed_dim
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Position embeddings (if not using RoPE)
        if not config.use_rope:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, config.num_patches + 1, config.embed_dim)
            )
        else:
            self.pos_embed = None

        # Mask token for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))

        # Select transformer block type
        BlockClass = (
            MacaronTransformerBlock if config.use_macaron
            else StandardTransformerBlock
        )

        # Encoder
        dim_head = config.embed_dim // config.encoder_heads
        self.encoder_blocks = nn.ModuleList([
            BlockClass(
                dim=config.embed_dim,
                heads=config.encoder_heads,
                dim_head=dim_head,
                mlp_ratio=config.mlp_ratio,
                use_swiglu=config.use_swiglu,
                use_rope=config.use_rope,
                max_seq_len=config.num_patches + 1
            )
            for _ in range(config.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)

        # Encoder to decoder projection
        self.encoder_to_decoder = nn.Linear(
            config.embed_dim, config.decoder_embed_dim, bias=True
        )

        # Decoder position embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.decoder_embed_dim)
        )

        # Decoder
        decoder_dim_head = config.decoder_embed_dim // config.decoder_heads
        self.decoder_blocks = nn.ModuleList([
            BlockClass(
                dim=config.decoder_embed_dim,
                heads=config.decoder_heads,
                dim_head=decoder_dim_head,
                mlp_ratio=config.mlp_ratio,
                use_swiglu=config.use_swiglu,
                use_rope=config.use_rope,
                max_seq_len=config.num_patches + 1
            )
            for _ in range(config.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)

        # Prediction head (reconstruct patches)
        self.pred_head = nn.Linear(
            config.decoder_embed_dim,
            config.patch_size ** 2 * 3,  # Reconstruct RGB patches
            bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Initialize tokens
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Initialize linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @property
    def embed_dim(self) -> int:
        """Return encoder embedding dimension."""
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        """Return number of patches."""
        return self._num_patches

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking by shuffling.

        Args:
            x: Patches [B, N, D]
            mask_ratio: Fraction of patches to mask

        Returns:
            x_masked: Visible patches [B, N_vis, D]
            mask: Binary mask [B, N] (1=masked, 0=visible)
            ids_restore: Indices to restore original order [B, N]
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep only unmasked tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Generate binary mask: 0 = keep, 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode only unmasked patches.

        Args:
            x: Images [B, 3, H, W]
            mask_ratio: Fraction to mask (0.0 for inference)

        Returns:
            latent: Encoder output [B, N_vis+1, D]
            mask: Binary mask [B, N]
            ids_restore: Restoration indices [B, N]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Add position embeddings (before masking)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:, :]  # Skip CLS position

        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Append CLS token
        cls_token = self.cls_token
        if self.pos_embed is not None:
            cls_token = cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode and reconstruct masked patches.

        Args:
            x: Encoder output [B, N_vis+1, D]
            ids_restore: Restoration indices [B, N]

        Returns:
            Predicted patches [B, N, patch_pixels]
        """
        # Project to decoder dimension
        x = self.encoder_to_decoder(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Skip CLS, add masks

        # Unshuffle to restore original order
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )

        # Add back CLS token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # Add decoder position embeddings
        x = x + self.decoder_pos_embed

        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predict patches (skip CLS)
        x = self.pred_head(x[:, 1:, :])

        return x

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss on masked patches only.

        Args:
            imgs: Original images [B, 3, H, W]
            pred: Predicted patches [B, N, patch_pixels]
            mask: Binary mask [B, N] (1=masked, 0=visible)

        Returns:
            Scalar reconstruction loss
        """
        # Patchify target images
        target = self.patchify(imgs)

        # Normalize target (per-patch)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        # MSE loss on masked patches only
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch pixels
        loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches

        return loss

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches."""
        p = self.config.patch_size
        h = w = self.config.img_size // p

        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, h, w, p, p, 3]
        x = x.reshape(imgs.shape[0], h * w, p * p * 3)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images."""
        p = self.config.patch_size
        h = w = self.config.img_size // p

        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, 3, h, p, w, p]
        x = x.reshape(x.shape[0], 3, h * p, w * p)
        return x

    def get_embedding(
        self,
        imgs: torch.Tensor,
        pooling_mode: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract embeddings from encoder without masking.

        Args:
            imgs: Images [B, 3, H, W]
            pooling_mode: "cls", "mean", or "cls+mean" (defaults to config)

        Returns:
            Embeddings [B, D] or [B, 2*D] for cls+mean
        """
        if pooling_mode is None:
            pooling_mode = self.config.pooling_mode

        # Forward encoder with no masking
        latent, _, _ = self.forward_encoder(imgs, mask_ratio=0.0)

        # Apply pooling
        return get_embedding(latent, mode=pooling_mode)

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: Optional[float] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Full forward pass with optional contrastive and uniformity losses.

        Args:
            imgs: Images [B, 3, H, W]
            mask_ratio: Masking ratio (defaults to config)
            labels: Class labels for contrastive loss (optional)

        Returns:
            If labels provided and losses enabled:
                (total_loss, pred, mask, loss_dict)
            Otherwise:
                (reconstruction_loss, pred, mask)
        """
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio

        # Encode
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        # Decode
        pred = self.forward_decoder(latent, ids_restore)

        # Reconstruction loss
        recon_loss = self.forward_loss(imgs, pred, mask)

        # If no labels or no extra losses enabled, return simple output
        use_extra_losses = (
            labels is not None and
            (self.config.use_contrastive_loss or self.config.use_uniformity_loss)
        )

        if not use_extra_losses:
            return recon_loss, pred, mask

        # Compute embeddings for contrastive/uniformity losses
        embeddings = get_embedding(latent, mode=self.config.pooling_mode)

        # Initialize loss dict
        loss_dict = {
            'reconstruction': recon_loss.item(),
            'contrastive': 0.0,
            'uniformity': 0.0,
            'total': recon_loss.item()
        }

        total_loss = recon_loss

        # Contrastive loss
        if self.config.use_contrastive_loss and labels is not None:
            c_loss = info_nce_loss(
                embeddings, labels,
                temperature=self.config.contrastive_temperature
            )
            total_loss = total_loss + self.config.contrastive_weight * c_loss
            loss_dict['contrastive'] = c_loss.item()

        # Uniformity loss
        if self.config.use_uniformity_loss:
            u_loss = uniformity_loss(embeddings, t=self.config.uniformity_t)
            total_loss = total_loss + self.config.uniformity_weight * u_loss
            loss_dict['uniformity'] = u_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, pred, mask, loss_dict
