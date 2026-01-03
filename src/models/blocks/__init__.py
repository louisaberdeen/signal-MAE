"""
Reusable transformer building blocks.

Components:
- attention: Multi-head attention with optional RoPE
- ffn: Feed-forward networks (SwiGLU, StandardFFN)
- transformer: Transformer blocks (Standard, Macaron)
"""

from src.models.blocks.attention import Attention, RotaryPositionEmbedding, apply_rotary_pos_emb
from src.models.blocks.ffn import SwiGLU, StandardFFN
from src.models.blocks.transformer import StandardTransformerBlock, MacaronTransformerBlock

__all__ = [
    "Attention",
    "RotaryPositionEmbedding",
    "apply_rotary_pos_emb",
    "SwiGLU",
    "StandardFFN",
    "StandardTransformerBlock",
    "MacaronTransformerBlock",
]
