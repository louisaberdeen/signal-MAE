"""
Checkpoint loading utilities with validation and error handling.

Provides safe loading of model checkpoints with automatic detection
of encoder-only vs full model checkpoints.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn


class CheckpointLoader:
    """
    Load AudioMAE checkpoints with validation and error handling.

    Supports:
    - Full model checkpoints (encoder + decoder)
    - Encoder-only checkpoints
    - Automatic format detection
    - Validation before loading

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model to ("cuda" or "cpu")
    """

    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

    def validate(self) -> bool:
        """
        Validate checkpoint exists and is readable.

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

    def load(
        self,
        model: nn.Module,
        encoder_only: bool = False
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load checkpoint into model.

        Args:
            model: PyTorch model to load checkpoint into
            encoder_only: If True, allows loading encoder-only checkpoints

        Returns:
            Tuple of (model, checkpoint_info)

        Raises:
            RuntimeError: If checkpoint cannot be loaded
        """
        if not self.validate():
            raise RuntimeError(f"Checkpoint validation failed: {self.checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            # Handle both formats: {'model_state_dict': ...} or direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                info = {
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'loss': checkpoint.get('loss', 'unknown'),
                    'has_optimizer': 'optimizer_state_dict' in checkpoint
                }
            else:
                # Direct state dict
                state_dict = checkpoint
                info = {'epoch': 'unknown', 'loss': 'unknown', 'has_optimizer': False}

            # Detect encoder-only checkpoint
            is_encoder_only = self._is_encoder_only(state_dict)
            if is_encoder_only and not encoder_only:
                print("[INFO] Detected encoder-only checkpoint (auto-detected)")
                encoder_only = True

            # Load state dict
            try:
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict,
                    strict=not encoder_only
                )
            except RuntimeError as load_error:
                if not encoder_only and "Missing key(s)" in str(load_error):
                    print("[INFO] Strict loading failed, retrying with encoder-only mode...")
                    encoder_only = True
                    missing_keys, unexpected_keys = model.load_state_dict(
                        state_dict,
                        strict=False
                    )
                else:
                    raise

            # Report loading results
            self._report_loading(encoder_only, missing_keys, unexpected_keys, state_dict)

            model = model.to(self.device)
            model.eval()

            return model, info

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def _is_encoder_only(self, state_dict: dict) -> bool:
        """Check if checkpoint is encoder-only."""
        decoder_keys = [
            'mask_token', 'decoder_embed', 'decoder_pos_embed',
            'decoder_norm', 'pred_head', 'encoder_to_decoder'
        ]
        decoder_block_keys = [k for k in state_dict.keys() if k.startswith('decoder_blocks.')]

        has_decoder = any(k in state_dict or f"{k}.weight" in state_dict for k in decoder_keys)
        has_decoder_blocks = len(decoder_block_keys) > 0

        return not (has_decoder or has_decoder_blocks)

    def _report_loading(
        self,
        encoder_only: bool,
        missing_keys: list,
        unexpected_keys: list,
        state_dict: dict
    ):
        """Report what was loaded."""
        if encoder_only:
            print(f"[OK] Loaded {len(state_dict)} encoder parameters")
            decoder_missing = [k for k in missing_keys if any(
                p in k for p in ['decoder', 'mask_token', 'pred_head']
            )]
            other_missing = [k for k in missing_keys if k not in decoder_missing]

            if decoder_missing:
                print(f"[INFO] Skipped {len(decoder_missing)} decoder parameters")
            if other_missing:
                print(f"[WARNING] Missing non-decoder keys: {other_missing[:5]}...")
        else:
            print(f"[OK] Loaded full model checkpoint")

        if unexpected_keys:
            print(f"[WARNING] Unexpected keys: {unexpected_keys[:5]}...")

    def get_info(self) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint info without loading model.

        Returns:
            Dict with checkpoint metadata, or None if error
        """
        if not self.validate():
            return None

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location='cpu',
                weights_only=False
            )

            info = {
                'path': str(self.checkpoint_path),
                'size_mb': self.checkpoint_path.stat().st_size / 1e6,
                'has_model_state': 'model_state_dict' in checkpoint if isinstance(checkpoint, dict) else True,
                'has_optimizer_state': 'optimizer_state_dict' in checkpoint if isinstance(checkpoint, dict) else False,
                'epoch': checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown',
                'loss': checkpoint.get('loss', 'unknown') if isinstance(checkpoint, dict) else 'unknown',
            }

            return info

        except Exception as e:
            print(f"Error reading checkpoint info: {e}")
            return None
