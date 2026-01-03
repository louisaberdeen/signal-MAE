"""
Utility classes for AudioMAE embedding generation and caching.

This module provides:
- CheckpointLoader: Load AudioMAE checkpoints with validation
- EmbeddingGenerator: Batch process audio to generate embeddings
- EmbeddingCache: Manage embedding cache with validation
- ProgressTracker: Track and display pipeline progress
- SyntheticGeoGenerator: Generate synthetic lat/long for PoC
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CheckpointLoader:
    """Load AudioMAE checkpoints with validation and error handling."""

    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        """
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device to load model to ("cuda" or "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

    def validate(self) -> bool:
        """Validate checkpoint exists and is readable.

        Returns:
            True if valid, False otherwise
        """
        if not self.checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {self.checkpoint_path}")
            return False

        if not self.checkpoint_path.is_file():
            print(f"Error: Checkpoint path is not a file: {self.checkpoint_path}")
            return False

        # Check file size
        size_mb = self.checkpoint_path.stat().st_size / 1e6
        if size_mb < 1:
            print(f"Warning: Checkpoint file is very small ({size_mb:.2f} MB)")
            print("This may not be a valid model checkpoint.")
            return False

        return True

    def load(self, model: nn.Module, encoder_only: bool = False) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load checkpoint into model.

        Args:
            model: PyTorch model to load checkpoint into
            encoder_only: If True, allows loading encoder-only checkpoints (ignores missing decoder keys)

        Returns:
            Tuple of (model, checkpoint_info)

        Raises:
            RuntimeError: If checkpoint cannot be loaded
        """
        if not self.validate():
            raise RuntimeError(f"Checkpoint validation failed: {self.checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Handle both formats: {'model_state_dict': ...} or direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                info = {
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'loss': checkpoint.get('loss', 'unknown'),
                    'has_optimizer': 'optimizer_state_dict' in checkpoint
                }
            else:
                # Direct state dict (encoder_only.pt format)
                state_dict = checkpoint
                info = {'epoch': 'unknown', 'loss': 'unknown', 'has_optimizer': False}

            # Determine if this is an encoder-only checkpoint
            # Check for decoder-specific keys (excluding encoder keys that might have 'decoder' in path)
            decoder_specific_keys = [
                'mask_token', 'decoder_embed', 'decoder_pos_embed',
                'decoder_norm', 'pred_head', 'encoder_to_decoder'
            ]
            decoder_block_keys = [k for k in state_dict.keys() if k.startswith('decoder_blocks.')]
            has_decoder_specific = any(k in state_dict or k + '.weight' in state_dict
                                      for k in decoder_specific_keys)
            has_decoder_blocks = len(decoder_block_keys) > 0
            is_encoder_only = not (has_decoder_specific or has_decoder_blocks)

            # Override encoder_only parameter if auto-detected
            if is_encoder_only and not encoder_only:
                print("[INFO] Detected encoder-only checkpoint (auto-detected)")
                encoder_only = True

            # Try loading with appropriate strict setting
            try:
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict,
                    strict=not encoder_only  # strict=False for encoder-only
                )
            except RuntimeError as load_error:
                # If strict loading failed and we haven't tried encoder_only mode, retry
                if not encoder_only and "Missing key(s)" in str(load_error):
                    print("[INFO] Strict loading failed, retrying with encoder-only mode...")
                    encoder_only = True
                    missing_keys, unexpected_keys = model.load_state_dict(
                        state_dict,
                        strict=False
                    )
                else:
                    raise

            # Report what was loaded
            if encoder_only:
                print(f"[OK] Loaded {len(state_dict)} encoder parameters")
                decoder_missing = [k for k in missing_keys
                                 if any(p in k for p in ['decoder', 'mask_token', 'pred_head', 'decoder_embed', 'decoder_pos_embed'])]
                other_missing = [k for k in missing_keys
                               if not any(p in k for p in ['decoder', 'mask_token', 'pred_head', 'decoder_embed', 'decoder_pos_embed'])]

                if decoder_missing:
                    print(f"[INFO] Skipped {len(decoder_missing)} decoder parameters (expected for encoder-only)")
                if other_missing:
                    print(f"[WARNING] Missing non-decoder keys: {other_missing}")
            else:
                print(f"[OK] Loaded full model checkpoint")

            if unexpected_keys:
                print(f"[WARNING] Unexpected keys: {unexpected_keys[:5]}...")

            model = model.to(self.device)
            model.eval()

            return model, info

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get checkpoint info without loading model.

        Returns:
            Dict with checkpoint metadata, or None if error
        """
        if not self.validate():
            return None

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

            info = {
                'path': str(self.checkpoint_path),
                'size_mb': self.checkpoint_path.stat().st_size / 1e6,
                'has_model_state': 'model_state_dict' in checkpoint,
                'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
                'epoch': checkpoint.get('epoch', 'unknown'),
                'loss': checkpoint.get('loss', 'unknown'),
            }

            return info

        except Exception as e:
            print(f"Error reading checkpoint info: {e}")
            return None


class EmbeddingGenerator:
    """Batch process audio files to generate embeddings from AudioMAE model."""

    def __init__(self, model: nn.Module, device: str, batch_size: int = 32, pooling_mode: str = None):
        """
        Args:
            model: AudioMAE model
            device: Device to run inference on
            batch_size: Batch size for processing
            pooling_mode: Pooling mode for embedding extraction.
                          "cls" (CLS token), "mean" (mean of patches), "cls+mean" (concatenation)
                          If None, uses model's config.pooling_mode
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

        # Get pooling mode from config or parameter
        if pooling_mode is not None:
            self.pooling_mode = pooling_mode
        elif hasattr(model, 'config') and hasattr(model.config, 'pooling_mode'):
            self.pooling_mode = model.config.pooling_mode
        else:
            self.pooling_mode = "mean"  # Default to mean pooling

    def extract_embedding(self, spectrograms: torch.Tensor, pooling_mode: str = None) -> np.ndarray:
        """Extract embedding from AudioMAE encoder with configurable pooling.

        Critical: Uses mask_ratio=0.0 for inference (no masking).

        Args:
            spectrograms: Tensor of shape [B, 3, H, W]
            pooling_mode: Override pooling mode for this call (optional)

        Returns:
            embeddings: NumPy array of shape [B, embed_dim] or [B, 2*embed_dim] for cls+mean
        """
        mode = pooling_mode if pooling_mode is not None else self.pooling_mode

        self.model.eval()
        with torch.no_grad():
            # Forward encoder with NO masking (mask_ratio=0.0)
            latent, _, _ = self.model.forward_encoder(spectrograms, mask_ratio=0.0)

            # Apply pooling based on mode
            if mode == "cls":
                # CLS token is at position 0
                embedding = latent[:, 0, :]
            elif mode == "mean":
                # Mean of patch tokens (exclude CLS at position 0)
                embedding = latent[:, 1:, :].mean(dim=1)
            elif mode == "cls+mean":
                # Concatenate CLS and mean
                cls_embed = latent[:, 0, :]
                mean_embed = latent[:, 1:, :].mean(dim=1)
                embedding = torch.cat([cls_embed, mean_embed], dim=1)
            else:
                raise ValueError(f"Unknown pooling mode: {mode}. Use 'cls', 'mean', or 'cls+mean'")

        return embedding.cpu().numpy()

    def generate_embeddings(
        self,
        dataset: Dataset,
        desc: str = "Generating embeddings",
        num_workers: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate embeddings for entire dataset.

        Args:
            dataset: PyTorch Dataset returning (data, index) tuples
            desc: Description for progress bar
            num_workers: Number of dataloader workers (0 for Jupyter)

        Returns:
            Tuple of (embeddings, indices)
            - embeddings: NumPy array of shape [N, embed_dim]
            - indices: NumPy array of sample indices
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False
        )

        all_embeddings = []
        all_indices = []

        for batch_data, indices in tqdm(dataloader, desc=desc):
            batch_data = batch_data.to(self.device)

            # Extract embeddings
            embeddings = self.extract_embedding(batch_data)

            all_embeddings.append(embeddings)
            all_indices.append(indices.numpy())

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        indices = np.concatenate(all_indices)

        return embeddings, indices


class EmbeddingCache:
    """Manage embedding cache with validation."""

    def __init__(self, cache_dir: Path):
        """
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.embeddings_file = self.cache_dir / "embeddings.npy"
        self.metadata_file = self.cache_dir / "metadata.json"
        self.config_file = self.cache_dir / "config.json"

    def exists(self) -> bool:
        """Check if cache exists.

        Returns:
            True if cache files exist, False otherwise
        """
        return (
            self.embeddings_file.exists() and
            self.metadata_file.exists()
        )

    def validate(self) -> bool:
        """Validate cache integrity.

        Returns:
            True if cache is valid, False otherwise
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
        """Load embeddings from cache.

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
        """Save embeddings to cache.

        Args:
            embeddings: NumPy array of embeddings
            metadata: Metadata dict (should include shape, num_samples, etc.)
            config: Optional model config dict

        Returns:
            True if save successful, False otherwise
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

            # Save embeddings
            np.save(self.embeddings_file, embeddings)

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

    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get cache information.

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


class ProgressTracker:
    """Track and display pipeline progress."""

    def __init__(self):
        self.steps = {}

    def start(self, step_name: str):
        """Mark a step as started.

        Args:
            step_name: Name of the step
        """
        self.steps[step_name] = {
            "status": "running",
            "start_time": pd.Timestamp.now()
        }
        print(f"\n{'='*60}")
        print(f"Starting: {step_name}")
        print(f"{'='*60}")

    def complete(self, step_name: str):
        """Mark a step as completed.

        Args:
            step_name: Name of the step
        """
        if step_name in self.steps:
            elapsed = pd.Timestamp.now() - self.steps[step_name]["start_time"]
            self.steps[step_name]["status"] = "completed"
            self.steps[step_name]["elapsed"] = elapsed
            print(f"\nCompleted: {step_name} (Elapsed: {elapsed.total_seconds():.2f}s)")
        else:
            print(f"Warning: Step '{step_name}' not found in tracker")

    def error(self, step_name: str, error_msg: str):
        """Mark a step as failed.

        Args:
            step_name: Name of the step
            error_msg: Error message
        """
        if step_name in self.steps:
            self.steps[step_name]["status"] = "failed"
            self.steps[step_name]["error"] = error_msg
            print(f"\nFailed: {step_name}")
            print(f"Error: {error_msg}")

    def summary(self):
        """Display summary of all steps."""
        print(f"\n{'='*60}")
        print("Pipeline Summary")
        print(f"{'='*60}")

        for step, info in self.steps.items():
            status = info["status"]
            elapsed = info.get("elapsed", "N/A")

            if elapsed != "N/A":
                elapsed = f"{elapsed.total_seconds():.2f}s"

            status_symbol = {
                "completed": "[OK]",
                "running": "[RUNNING]",
                "failed": "[FAILED]"
            }.get(status, "[?]")

            print(f"  {status_symbol} {step}: {status} ({elapsed})")

            if status == "failed" and "error" in info:
                print(f"      Error: {info['error']}")


class SyntheticGeoGenerator:
    """Generate synthetic lat/long coordinates for PoC.

    Creates geographic patterns where similar categories are clustered
    in specific regions for meaningful visualization.
    """

    # Define category clusters (rough geographic regions)
    CATEGORY_CENTERS = {
        'Animals': (40.7128, -74.0060),  # New York area
        'Natural soundscapes and water sounds': (47.6062, -122.3321),  # Seattle
        'Human, non-speech sounds': (51.5074, -0.1278),  # London
        'Interior/domestic sounds': (35.6762, 139.6503),  # Tokyo
        'Exterior/urban noises': (34.0522, -118.2437)  # Los Angeles
    }

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    def generate(self, category: str) -> Tuple[float, float]:
        """Generate synthetic lat/long for a category.

        Args:
            category: Category name

        Returns:
            Tuple of (latitude, longitude)
        """
        # Set seed based on category for consistency
        np.random.seed(self.seed + hash(category) % 1000)

        # Get center for this category
        center_lat, center_lon = self.CATEGORY_CENTERS.get(
            category,
            (0.0, 0.0)  # Default to equator/prime meridian
        )

        # Add random offset (within ~50km radius)
        # 1 degree ≈ 111 km, so 0.5 degree ≈ 55 km
        lat_offset = np.random.normal(0, 0.5)
        lon_offset = np.random.normal(0, 0.5)

        latitude = center_lat + lat_offset
        longitude = center_lon + lon_offset

        return latitude, longitude

    def generate_for_dataframe(
        self,
        df: pd.DataFrame,
        category_column: str = 'category'
    ) -> pd.DataFrame:
        """Add synthetic lat/long columns to dataframe.

        Args:
            df: DataFrame with category column
            category_column: Name of category column

        Returns:
            DataFrame with added 'latitude' and 'longitude' columns
        """
        coords = df[category_column].apply(self.generate)

        df['latitude'] = coords.apply(lambda x: x[0])
        df['longitude'] = coords.apply(lambda x: x[1])

        return df
