"""
Embedding generation and caching utilities.

Components:
- generator: EmbeddingGenerator for batch extraction
- cache: EmbeddingCache for persistent storage
- checkpoint: CheckpointLoader for model loading
"""

from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.cache import EmbeddingCache
from src.embeddings.checkpoint import CheckpointLoader

__all__ = [
    "EmbeddingGenerator",
    "EmbeddingCache",
    "CheckpointLoader",
]
