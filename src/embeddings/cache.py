"""
Embedding cache management with validation.

EmbeddingCache provides persistent storage for embeddings with
integrity checking and metadata tracking.
"""

import json
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np


class EmbeddingCache:
    """
    Manage embedding cache with validation.

    Provides:
    - Persistent storage of embeddings as .npy files
    - Metadata tracking (shape, timestamp, model info)
    - Integrity validation (shape matching, NaN detection)
    - Safe atomic writes

    Args:
        cache_dir: Directory to store cached embeddings
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.embeddings_file = self.cache_dir / "embeddings.npy"
        self.metadata_file = self.cache_dir / "metadata.json"
        self.config_file = self.cache_dir / "config.json"

    def exists(self) -> bool:
        """
        Check if cache exists.

        Returns:
            True if cache files exist
        """
        return (
            self.embeddings_file.exists() and
            self.metadata_file.exists()
        )

    def validate(self) -> bool:
        """
        Validate cache integrity.

        Returns:
            True if cache is valid
        """
        if not self.exists():
            return False

        try:
            # Load and check embeddings
            embeddings = np.load(self.embeddings_file)

            # Load and check metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            # Verify shape matches
            expected_shape = tuple(metadata.get('shape', []))
            if embeddings.shape != expected_shape:
                print(f"Warning: Embedding shape mismatch")
                print(f"  Expected: {expected_shape}")
                print(f"  Got: {embeddings.shape}")
                return False

            # Check for NaNs
            if np.isnan(embeddings).any():
                print("Warning: Cache contains NaN values")
                return False

            return True

        except Exception as e:
            print(f"Error validating cache: {e}")
            return False

    def load(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Load embeddings from cache.

        Returns:
            Tuple of (embeddings, metadata) or (None, None) if error
        """
        if not self.validate():
            print("Cache validation failed")
            return None, None

        try:
            embeddings = np.load(self.embeddings_file)

            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            return embeddings, metadata

        except Exception as e:
            print(f"Error loading cache: {e}")
            return None, None

    def save(
        self,
        embeddings: np.ndarray,
        metadata: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save embeddings to cache.

        Args:
            embeddings: NumPy array of embeddings
            metadata: Metadata dict (should include shape, num_samples)
            config: Optional model config dict

        Returns:
            True if save successful
        """
        try:
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Check disk space
            stat = shutil.disk_usage(self.cache_dir)
            free_gb = stat.free / (1024**3)
            if free_gb < 0.1:
                print(f"Error: Insufficient disk space ({free_gb:.2f} GB free)")
                return False

            # Ensure shape is in metadata
            metadata['shape'] = list(embeddings.shape)

            # Save embeddings atomically (write to temp, then rename)
            temp_file = self.embeddings_file.with_suffix('.tmp')
            np.save(temp_file, embeddings)
            temp_file.rename(self.embeddings_file)

            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save config if provided
            if config is not None:
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)

            # Verify save
            if not self.validate():
                print("Warning: Cache validation failed after save")
                return False

            return True

        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

    def invalidate(self) -> bool:
        """
        Delete cache.

        Returns:
            True if deletion successful
        """
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            return True
        except Exception as e:
            print(f"Error invalidating cache: {e}")
            return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        """
        Get cache information.

        Returns:
            Dict with cache info, or None if cache doesn't exist
        """
        if not self.exists():
            return None

        info = {
            'cache_dir': str(self.cache_dir),
            'embeddings_file': str(self.embeddings_file),
            'embeddings_size_mb': self.embeddings_file.stat().st_size / 1e6,
            'exists': True,
            'valid': self.validate()
        }

        # Add metadata if available
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            info['metadata'] = metadata

        return info
