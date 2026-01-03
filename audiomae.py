"""
AudioMAE++ and Baseline Transformer Models

This module provides:
- AudioMAE++: Masked Autoencoder with Macaron blocks, SwiGLU, and RoPE
- BaselineMAE: Standard Masked Autoencoder for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


# ============================================
# Configuration
# ============================================

class Config:
    """Configuration for AudioMAE models."""
    # Audio processing
    sample_rate = 22050
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    audio_duration = 5  # seconds

    # Spectrogram image size
    img_size = 224

    # Patch settings
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2  # 196 patches

    # Model architecture
    embed_dim = 768
    encoder_depth = 12
    encoder_heads = 12
    decoder_embed_dim = 512
    decoder_depth = 8
    decoder_heads = 16
    mlp_ratio = 4.0

    # Training
    mask_ratio = 0.75  # Mask 75% of patches (changed from 0.8)
    batch_size = 16
    learning_rate = 1.5e-4
    weight_decay = 0.05
    epochs = 50
    warmup_epochs = 5

    # AudioMAE++ features
    use_macaron = True
    use_swiglu = True
    use_rope = True

    # Embedding extraction
    pooling_mode = "mean"  # "cls", "mean", or "cls+mean"

    # Contrastive loss (improves semantic clustering)
    use_contrastive_loss = True
    contrastive_weight = 0.01  # Weight for contrastive loss
    contrastive_temperature = 0.07  # Temperature for InfoNCE

    # Uniformity loss (prevents embedding collapse)
    use_uniformity_loss = True
    uniformity_weight = 0.1  # Weight for uniformity loss
    uniformity_t = 2.0  # Temperature parameter for uniformity


# ============================================
# Loss Functions for Embedding Quality
# ============================================

def info_nce_loss(embeddings, labels, temperature=0.07):
    """
    InfoNCE contrastive loss for same-class positive pairs.

    Encourages embeddings of the same class to be closer together
    while pushing different classes apart.

    Args:
        embeddings: [B, D] embeddings (will be normalized)
        labels: [B] class labels
        temperature: softmax temperature (lower = harder negatives)

    Returns:
        Scalar loss value
    """
    # Normalize embeddings to unit sphere
    embeddings = F.normalize(embeddings, dim=1)
    batch_size = embeddings.shape[0]

    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Create positive mask: same class = positive pair
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)  # Exclude self-similarity

    # Check if we have any positive pairs
    num_positives = pos_mask.sum(dim=1)
    valid_samples = num_positives > 0

    if not valid_samples.any():
        # No positive pairs in batch, return zero loss
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Compute log softmax
    # Mask out self-similarity by setting diagonal to large negative
    logits_mask = torch.ones_like(sim_matrix)
    logits_mask.fill_diagonal_(0)

    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

    # Loss is negative log-likelihood (only for samples with positives)
    loss = -mean_log_prob_pos[valid_samples].mean()

    return loss


def uniformity_loss(embeddings, t=2.0):
    """
    Uniformity loss from "Understanding Contrastive Representation Learning".

    Encourages embeddings to be uniformly distributed on the unit hypersphere,
    preventing dimensional collapse where embeddings cluster too tightly.

    Args:
        embeddings: [B, D] embeddings (will be normalized)
        t: temperature parameter (higher = stronger push apart)

    Returns:
        Scalar loss value (lower is more uniform)
    """
    # Normalize embeddings to unit sphere
    embeddings = F.normalize(embeddings, dim=1)

    # Compute pairwise squared distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 2 - 2*a.b (for unit vectors)
    sq_pdist = torch.pdist(embeddings, p=2).pow(2)

    # Uniformity loss: log of average Gaussian kernel
    loss = sq_pdist.mul(-t).exp().mean().log()

    return loss


def get_embedding(latent, mode="mean"):
    """
    Extract embedding from encoder output using specified pooling.

    Args:
        latent: [B, N+1, D] encoder output (CLS token at position 0)
        mode: "cls" (CLS token only), "mean" (mean of patches),
              or "cls+mean" (concatenation)

    Returns:
        [B, D] or [B, 2*D] embedding
    """
    if mode == "cls":
        return latent[:, 0, :]
    elif mode == "mean":
        # Mean of patch tokens (exclude CLS)
        return latent[:, 1:, :].mean(dim=1)
    elif mode == "cls+mean":
        cls_embed = latent[:, 0, :]
        mean_embed = latent[:, 1:, :].mean(dim=1)
        return torch.cat([cls_embed, mean_embed], dim=1)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}. Use 'cls', 'mean', or 'cls+mean'")


# ============================================
# SwiGLU Feed-Forward Network
# ============================================

class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(x @ W_gate) * (x @ W_value)"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_value = nn.Linear(dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish activation
        value = self.w_value(x)
        x = gate * value  # Element-wise gating
        x = self.dropout(x)
        x = self.w_out(x)
        return x


class StandardFFN(nn.Module):
    """Standard FFN with GELU activation."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================
# Rotary Position Embeddings (RoPE)
# ============================================

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding for 2D (image patches)."""

    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        self.dim = dim

        # Compute frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute position encodings
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys."""
    # Expand cos and sin to match q and k dimensions
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================
# Attention with optional RoPE
# ============================================

class Attention(nn.Module):
    """Multi-head attention with optional RoPE."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_rope=False, max_seq_len=256):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        if use_rope:
            self.rope = RotaryPositionEmbedding(dim_head, max_seq_len)

    def forward(self, x):
        b, n, _ = x.shape

        # Compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), qkv)

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rope(x, seq_len=n)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        q = rearrange(q, 'b n h d -> b h n d')
        k = rearrange(k, 'b n h d -> b h n d')
        v = rearrange(v, 'b n h d -> b h n d')

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ============================================
# Transformer Blocks: Standard vs Macaron
# ============================================

class StandardTransformerBlock(nn.Module):
    """Standard transformer block: Attention -> FFN"""

    def __init__(self, dim, heads, dim_head, mlp_ratio=4.0, dropout=0.0,
                 use_swiglu=False, use_rope=False, max_seq_len=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout, use_rope, max_seq_len)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        if use_swiglu:
            self.ffn = SwiGLU(dim, hidden_dim, dropout)
        else:
            self.ffn = StandardFFN(dim, hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MacaronTransformerBlock(nn.Module):
    """
    Macaron-style transformer block (AudioMAE++):
    FFN (½) -> Attention -> FFN with SwiGLU (½)

    The attention is sandwiched between two FFN layers,
    each with half the residual weight.
    """

    def __init__(self, dim, heads, dim_head, mlp_ratio=4.0, dropout=0.0,
                 use_swiglu=True, use_rope=False, max_seq_len=256):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        # First FFN (simple, half-weighted)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn1 = StandardFFN(dim, hidden_dim, dropout)

        # Attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout, use_rope, max_seq_len)

        # Second FFN with SwiGLU (half-weighted)
        self.norm3 = nn.LayerNorm(dim)
        if use_swiglu:
            self.ffn2 = SwiGLU(dim, hidden_dim, dropout)
        else:
            self.ffn2 = StandardFFN(dim, hidden_dim, dropout)

        # Half-weight factor for FFN residuals
        self.ffn_scale = 0.5

    def forward(self, x):
        # First FFN (half-weighted)
        x = x + self.ffn_scale * self.ffn1(self.norm1(x))

        # Attention (full weight)
        x = x + self.attn(self.norm2(x))

        # Second FFN with SwiGLU (half-weighted)
        x = x + self.ffn_scale * self.ffn2(self.norm3(x))

        return x


# ============================================
# Patch Embedding
# ============================================

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


# ============================================
# AudioMAE++ Full Model
# ============================================

class AudioMAEPlusPlus(nn.Module):
    """
    AudioMAE++ - Masked Autoencoder with:
    - Macaron-style transformer blocks
    - SwiGLU activation
    - Optional RoPE
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Position embeddings (if not using RoPE)
        if not config.use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.pos_embed = None

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))

        # Select transformer block type
        BlockClass = MacaronTransformerBlock if config.use_macaron else StandardTransformerBlock

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
                max_seq_len=num_patches + 1
            )
            for _ in range(config.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)

        # Encoder to decoder projection
        self.encoder_to_decoder = nn.Linear(config.embed_dim, config.decoder_embed_dim, bias=True)

        # Decoder position embeddings
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.decoder_embed_dim))

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
                max_seq_len=num_patches + 1
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

    def random_masking(self, x, mask_ratio):
        """
        Perform random masking by shuffling.
        x: [B, N, D] (patches)
        Returns: masked x, mask, ids_restore
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
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate binary mask: 0 = keep, 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """Encode only unmasked patches."""
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

    def forward_decoder(self, x, ids_restore):
        """Decode and reconstruct masked patches."""
        # Project to decoder dimension
        x = self.encoder_to_decoder(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Skip CLS, add masks

        # Unshuffle to restore original order
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

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

    def forward_loss(self, imgs, pred, mask):
        """
        Compute MSE loss on masked patches only.
        imgs: [B, 3, H, W]
        pred: [B, N, patch_size^2 * 3]
        mask: [B, N] (1 = masked, 0 = visible)
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

    def patchify(self, imgs):
        """Convert images to patches."""
        p = self.config.patch_size
        h = w = self.config.img_size // p

        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, h, w, p, p, 3]
        x = x.reshape(imgs.shape[0], h * w, p * p * 3)
        return x

    def unpatchify(self, x):
        """Convert patches back to images."""
        p = self.config.patch_size
        h = w = self.config.img_size // p

        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, 3, h, p, w, p]
        x = x.reshape(x.shape[0], 3, h * p, w * p)
        return x

    def get_embedding(self, imgs, pooling_mode=None):
        """
        Extract embeddings from encoder without masking.

        Args:
            imgs: [B, 3, H, W] input images
            pooling_mode: "cls", "mean", or "cls+mean" (defaults to config)

        Returns:
            [B, D] or [B, 2*D] embedding tensor
        """
        if pooling_mode is None:
            pooling_mode = self.config.pooling_mode

        # Forward encoder with no masking
        latent, _, _ = self.forward_encoder(imgs, mask_ratio=0.0)

        # Apply pooling
        return get_embedding(latent, mode=pooling_mode)

    def forward(self, imgs, mask_ratio=None, labels=None):
        """
        Full forward pass with optional contrastive and uniformity losses.

        Args:
            imgs: [B, 3, H, W] input images
            mask_ratio: masking ratio (defaults to config)
            labels: [B] class labels for contrastive loss (optional)

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
        # Use full encoder pass without masking for embedding quality
        with torch.no_grad():
            full_latent, _, _ = self.forward_encoder(imgs, mask_ratio=0.0)
        # Detach and re-enable gradients for the embedding computation
        embeddings = get_embedding(full_latent.detach(), mode=self.config.pooling_mode)
        embeddings.requires_grad_(True)

        # Actually, we want gradients to flow through, so let's redo this
        # We'll use the masked latent for efficiency during training
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


# ============================================
# Baseline MAE (Standard Transformer)
# ============================================

class BaselineMAE(nn.Module):
    """
    Baseline Masked Autoencoder with standard transformer blocks.
    No Macaron, no SwiGLU, no RoPE - just vanilla ViT architecture.
    """

    def __init__(self, config):
        super().__init__()
        # Create a config copy with baseline settings
        self.config = Config()
        for key, value in vars(config).items():
            setattr(self.config, key, value)

        # Override AudioMAE++ features
        self.config.use_macaron = False
        self.config.use_swiglu = False
        self.config.use_rope = False

        # Use AudioMAEPlusPlus implementation with baseline config
        self.model = AudioMAEPlusPlus(self.config)

    def forward(self, imgs, mask_ratio=None):
        return self.model(imgs, mask_ratio)

    def forward_encoder(self, x, mask_ratio):
        return self.model.forward_encoder(x, mask_ratio)

    def forward_decoder(self, x, ids_restore):
        return self.model.forward_decoder(x, ids_restore)

    def patchify(self, imgs):
        return self.model.patchify(imgs)

    def unpatchify(self, x):
        return self.model.unpatchify(x)


# ============================================
# Classifier Wrapper
# ============================================

class AudioMAEClassifier(nn.Module):
    """
    Classification head on top of AudioMAE encoder.
    Can be used with either AudioMAE++ or BaselineMAE.
    """

    def __init__(self, pretrained_model, num_classes, freeze_encoder=False):
        super().__init__()

        # Handle both AudioMAEPlusPlus and BaselineMAE
        if isinstance(pretrained_model, BaselineMAE):
            pretrained_model = pretrained_model.model

        self.config = pretrained_model.config

        # Copy encoder components
        self.patch_embed = pretrained_model.patch_embed
        self.cls_token = pretrained_model.cls_token
        self.pos_embed = pretrained_model.pos_embed
        self.encoder_blocks = pretrained_model.encoder_blocks
        self.encoder_norm = pretrained_model.encoder_norm

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for param in self.encoder_blocks.parameters():
                param.requires_grad = False
            for param in self.encoder_norm.parameters():
                param.requires_grad = False
            if self.pos_embed is not None:
                self.pos_embed.requires_grad = False
            self.cls_token.requires_grad = False

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, num_classes)
        )

    def forward(self, x, mask_ratio=0.0):
        """Forward pass for classification."""
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
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

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


# ============================================
# Helper Functions
# ============================================

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_audiomae_plusplus(config=None):
    """Create AudioMAE++ model with optional custom config."""
    if config is None:
        config = Config()
    return AudioMAEPlusPlus(config)


def create_baseline_mae(config=None):
    """Create baseline MAE model (standard transformer)."""
    if config is None:
        config = Config()
    return BaselineMAE(config)


if __name__ == "__main__":
    # Example usage
    config = Config()

    print("Creating AudioMAE++ model...")
    model_plus = create_audiomae_plusplus(config)
    total, trainable = count_parameters(model_plus)
    print(f"  Parameters: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print(f"  Macaron: {config.use_macaron}, SwiGLU: {config.use_swiglu}, RoPE: {config.use_rope}")

    print("\nCreating Baseline MAE model...")
    model_baseline = create_baseline_mae(config)
    total, trainable = count_parameters(model_baseline)
    print(f"  Parameters: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print(f"  Standard transformer (no advanced features)")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, config.img_size, config.img_size)

    loss_plus, pred_plus, mask_plus = model_plus(dummy_input)
    print(f"  AudioMAE++ - Loss: {loss_plus.item():.4f}")

    loss_base, pred_base, mask_base = model_baseline(dummy_input)
    print(f"  Baseline - Loss: {loss_base.item():.4f}")

    print("\nModels ready for import!")
