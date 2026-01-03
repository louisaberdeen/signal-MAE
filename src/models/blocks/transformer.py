"""
Transformer block variants for encoder and decoder.

Components:
- StandardTransformerBlock: Classic Attention -> FFN structure
- MacaronTransformerBlock: FFN -> Attention -> FFN sandwich (AudioMAE++)
"""

import torch
import torch.nn as nn

from src.models.blocks.attention import Attention
from src.models.blocks.ffn import SwiGLU, StandardFFN


class StandardTransformerBlock(nn.Module):
    """
    Standard transformer block: Attention -> FFN.

    The classic Pre-LN transformer architecture with residual connections.

    Args:
        dim: Input dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        mlp_ratio: FFN hidden dimension multiplier
        dropout: Dropout rate
        use_swiglu: Use SwiGLU instead of GELU FFN
        use_rope: Use rotary position embeddings
        max_seq_len: Maximum sequence length for RoPE
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_swiglu: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 256
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout, use_rope, max_seq_len)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        if use_swiglu:
            self.ffn = SwiGLU(dim, hidden_dim, dropout)
        else:
            self.ffn = StandardFFN(dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Norm -> Attention -> Add -> Norm -> FFN -> Add.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MacaronTransformerBlock(nn.Module):
    """
    Macaron-style transformer block (AudioMAE++).

    FFN (half) -> Attention -> FFN with SwiGLU (half)

    The attention is sandwiched between two FFN layers, each with
    half the residual weight. This architecture was shown to improve
    performance in speech and audio models.

    Reference: Macaron Net (Lu et al., 2019)

    Args:
        dim: Input dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        mlp_ratio: FFN hidden dimension multiplier
        dropout: Dropout rate
        use_swiglu: Use SwiGLU for second FFN (recommended)
        use_rope: Use rotary position embeddings
        max_seq_len: Maximum sequence length for RoPE
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 256
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: FFN(0.5) -> Attention(1.0) -> FFN(0.5).

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D]
        """
        # First FFN (half-weighted residual)
        x = x + self.ffn_scale * self.ffn1(self.norm1(x))

        # Attention (full weight residual)
        x = x + self.attn(self.norm2(x))

        # Second FFN with SwiGLU (half-weighted residual)
        x = x + self.ffn_scale * self.ffn2(self.norm3(x))

        return x


class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings using a convolutional projection.

    Args:
        img_size: Input image size (assumed square)
        patch_size: Patch size (assumed square)
        in_chans: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch embeddings.

        Args:
            x: Images [B, C, H, W]

        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        # x: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        # [B, embed_dim, H/P, W/P] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x
