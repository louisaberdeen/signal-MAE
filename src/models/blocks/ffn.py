"""
Feed-forward network variants for transformer blocks.

Components:
- SwiGLU: Gated linear unit with Swish activation (better than GELU)
- StandardFFN: Standard FFN with GELU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish(x @ W_gate) * (x @ W_value).

    A gated linear unit variant that uses Swish (SiLU) activation
    for the gate. Shown to improve performance in language models
    and vision transformers.

    Reference: GLU Variants Improve Transformer (Shazeer, 2020)

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4x input dim)
        dropout: Dropout rate
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_value = nn.Linear(dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated activation.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D]
        """
        gate = F.silu(self.w_gate(x))  # Swish activation
        value = self.w_value(x)
        x = gate * value  # Element-wise gating
        x = self.dropout(x)
        x = self.w_out(x)
        return x


class StandardFFN(nn.Module):
    """
    Standard feed-forward network with GELU activation.

    The classic transformer FFN: Linear -> GELU -> Dropout -> Linear -> Dropout

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4x input dim)
        dropout: Dropout rate
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FFN.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D]
        """
        return self.net(x)
