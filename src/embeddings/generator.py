"""
Embedding generation utilities for batch processing.

EmbeddingGenerator extracts embeddings from pretrained models
using configurable pooling strategies.
"""

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Batch process data to generate embeddings from AudioMAE model.

    Supports configurable pooling modes (CLS, mean, CLS+mean) and
    efficient batch processing with GPU acceleration.

    Args:
        model: AudioMAE model with forward_encoder method
        device: Device to run inference on
        batch_size: Batch size for processing
        pooling_mode: Pooling mode for embedding extraction
            - "cls": CLS token only
            - "mean": Mean of patch tokens
            - "cls+mean": Concatenation of both
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        batch_size: int = 32,
        pooling_mode: Optional[str] = None
    ):
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
            self.pooling_mode = "mean"

    def extract_embedding(
        self,
        spectrograms: torch.Tensor,
        pooling_mode: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract embedding from AudioMAE encoder.

        Uses mask_ratio=0.0 for inference (no masking).

        Args:
            spectrograms: Tensor of shape [B, 3, H, W]
            pooling_mode: Override pooling mode for this call

        Returns:
            Embeddings as NumPy array [B, embed_dim] or [B, 2*embed_dim]
        """
        mode = pooling_mode if pooling_mode is not None else self.pooling_mode

        self.model.eval()
        with torch.no_grad():
            # Forward encoder with NO masking (mask_ratio=0.0)
            latent, _, _ = self.model.forward_encoder(spectrograms, mask_ratio=0.0)

            # Apply pooling based on mode
            if mode == "cls":
                embedding = latent[:, 0, :]
            elif mode == "mean":
                embedding = latent[:, 1:, :].mean(dim=1)
            elif mode == "cls+mean":
                cls_embed = latent[:, 0, :]
                mean_embed = latent[:, 1:, :].mean(dim=1)
                embedding = torch.cat([cls_embed, mean_embed], dim=1)
            else:
                raise ValueError(f"Unknown pooling mode: {mode}")

        return embedding.cpu().numpy()

    def generate_embeddings(
        self,
        dataset: Dataset,
        desc: str = "Generating embeddings",
        num_workers: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for entire dataset.

        Args:
            dataset: PyTorch Dataset returning (data, index) tuples
            desc: Description for progress bar
            num_workers: Number of dataloader workers (0 for Jupyter)

        Returns:
            Tuple of (embeddings, indices)
            - embeddings: NumPy array [N, embed_dim]
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

    def generate_single(self, spectrogram: torch.Tensor) -> np.ndarray:
        """
        Generate embedding for a single spectrogram.

        Args:
            spectrogram: Tensor [3, H, W] or [1, 3, H, W]

        Returns:
            Embedding [embed_dim] or [2*embed_dim]
        """
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)

        spectrogram = spectrogram.to(self.device)
        embedding = self.extract_embedding(spectrogram)

        return embedding[0]
