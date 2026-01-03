"""
Attention mechanisms with optional Rotary Position Embeddings (RoPE).

Components:
- RotaryPositionEmbedding: 2D rotary position encoding
- Attention: Multi-head attention with optional RoPE
- apply_rotary_pos_emb: Helper to apply RoPE to queries and keys
"""

import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 2D image patches.

    RoPE encodes position information directly into the attention mechanism
    by rotating query and key vectors based on their position. This allows
    the model to generalize to different sequence lengths.

    Args:
        dim: Dimension per attention head
        max_seq_len: Maximum sequence length to precompute embeddings for
    """

    def __init__(self, dim: int, max_seq_len: int = 256):
        super().__init__()
        self.dim = dim

        # Compute frequency bands (inverse frequencies)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute position encodings
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        Get cos and sin embeddings for the given sequence length.

        Args:
            x: Input tensor (used to determine device)
            seq_len: Sequence length (defaults to x.shape[1])

        Returns:
            cos: Cosine embeddings [seq_len, dim]
            sin: Sine embeddings [seq_len, dim]
        """
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions of the input.

    Used as part of the RoPE computation.

    Args:
        x: Input tensor [..., dim]

    Returns:
        Rotated tensor [..., dim]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple:
    """
    Apply rotary position embedding to queries and keys.

    Args:
        q: Query tensor [B, seq_len, heads, dim_head]
        k: Key tensor [B, seq_len, heads, dim_head]
        cos: Cosine embeddings [seq_len, dim_head]
        sin: Sine embeddings [seq_len, dim_head]

    Returns:
        q_embed: Position-embedded queries
        k_embed: Position-embedded keys
    """
    # Expand cos and sin to match q and k dimensions
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """
    Multi-head attention with optional Rotary Position Embeddings.

    Standard scaled dot-product attention with the option to use RoPE
    for position encoding instead of learned position embeddings.

    Args:
        dim: Input dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        dropout: Dropout rate
        use_rope: Whether to use rotary position embeddings
        max_seq_len: Maximum sequence length for RoPE precomputation
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 256
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D]
        """
        b, n, _ = x.shape

        # Compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads),
            qkv
        )

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rope(x, seq_len=n)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Rearrange for attention computation
        q = rearrange(q, 'b n h d -> b h n d')
        k = rearrange(k, 'b n h d -> b h n d')
        v = rearrange(v, 'b n h d -> b h n d')

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
