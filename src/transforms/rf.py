"""
RF/IQ signal to spectrogram transforms.

Converts IQ (In-phase/Quadrature) samples from RF signals
to spectrograms for modulation classification.
"""

from typing import Union, Tuple

import numpy as np
import torch

from src.registry import transform_registry
from src.transforms.base import BaseTransform


@transform_registry.register("iq_spectrogram", version="1.0")
class IQToSpectrogram(BaseTransform):
    """
    Transform IQ samples to spectrogram.

    Converts complex IQ samples (In-phase/Quadrature) to a spectrogram
    representation suitable for vision transformers. Used for RF signal
    classification tasks like modulation recognition.

    Args:
        img_size: Output image size (default: 224)
        nperseg: Samples per segment for STFT (default: 64)
        noverlap: Overlap between segments (default: 32)
        normalize: Normalize output to [0, 1] (default: True)
        normalize_imagenet: Apply ImageNet normalization (default: True)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        img_size: int = 224,
        nperseg: int = 64,
        noverlap: int = 32,
        normalize: bool = True,
        normalize_imagenet: bool = True
    ):
        self.img_size = img_size
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.normalize = normalize
        self.normalize_imagenet = normalize_imagenet

    def __call__(
        self,
        iq_samples: Union[np.ndarray, torch.Tensor],
        sample_rate: int = None  # Not used for RadioML
    ) -> torch.Tensor:
        """
        Convert IQ samples to spectrogram.

        Args:
            iq_samples: Complex IQ data [2, N] (I and Q rows) or [N] complex
            sample_rate: Sample rate (not typically used for RadioML)

        Returns:
            Spectrogram tensor [3, img_size, img_size]
        """
        from scipy import signal
        from PIL import Image

        # Convert to numpy if tensor
        if isinstance(iq_samples, torch.Tensor):
            iq_samples = iq_samples.numpy()

        # Convert to complex if separate I/Q channels
        if iq_samples.ndim == 2 and iq_samples.shape[0] == 2:
            complex_signal = iq_samples[0] + 1j * iq_samples[1]
        elif np.iscomplexobj(iq_samples):
            complex_signal = iq_samples
        else:
            # Assume interleaved I/Q
            complex_signal = iq_samples[::2] + 1j * iq_samples[1::2]

        # Compute spectrogram using STFT
        f, t, Sxx = signal.spectrogram(
            complex_signal,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            return_onesided=False  # Full spectrum for complex input
        )

        # Convert to dB scale
        Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)

        # Normalize to [0, 1]
        if self.normalize:
            Sxx_norm = (Sxx_db - Sxx_db.min())
            Sxx_norm = Sxx_norm / (Sxx_norm.max() + 1e-8)
        else:
            Sxx_norm = Sxx_db

        # Resize to target size
        img = Image.fromarray((Sxx_norm * 255).astype(np.uint8))
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to 3-channel tensor
        tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

        # Apply ImageNet normalization
        if self.normalize_imagenet:
            mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    @property
    def output_channels(self) -> int:
        """Return 3 channels (RGB-like for ViT)."""
        return 3

    @property
    def output_size(self) -> Tuple[int, int]:
        """Return (img_size, img_size)."""
        return (self.img_size, self.img_size)


@transform_registry.register("iq_constellation", version="1.0")
class IQToConstellation(BaseTransform):
    """
    Transform IQ samples to constellation diagram image.

    Creates a 2D scatter plot of I vs Q values, useful for
    visualizing modulation schemes (QPSK, QAM, etc.).

    Args:
        img_size: Output image size (default: 224)
        normalize_imagenet: Apply ImageNet normalization (default: True)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        img_size: int = 224,
        normalize_imagenet: bool = True
    ):
        self.img_size = img_size
        self.normalize_imagenet = normalize_imagenet

    def __call__(
        self,
        iq_samples: Union[np.ndarray, torch.Tensor],
        sample_rate: int = None
    ) -> torch.Tensor:
        """
        Convert IQ samples to constellation diagram.

        Args:
            iq_samples: IQ data [2, N] or complex [N]
            sample_rate: Not used

        Returns:
            Constellation image tensor [3, img_size, img_size]
        """
        # Convert to numpy if tensor
        if isinstance(iq_samples, torch.Tensor):
            iq_samples = iq_samples.numpy()

        # Get I and Q components
        if iq_samples.ndim == 2 and iq_samples.shape[0] == 2:
            I, Q = iq_samples[0], iq_samples[1]
        elif np.iscomplexobj(iq_samples):
            I, Q = iq_samples.real, iq_samples.imag
        else:
            I, Q = iq_samples[::2], iq_samples[1::2]

        # Normalize to [-1, 1] range
        max_val = max(np.abs(I).max(), np.abs(Q).max()) + 1e-8
        I_norm = I / max_val
        Q_norm = Q / max_val

        # Create 2D histogram (constellation diagram)
        bins = self.img_size
        H, xedges, yedges = np.histogram2d(
            I_norm, Q_norm,
            bins=bins,
            range=[[-1, 1], [-1, 1]]
        )

        # Normalize histogram
        H = H / (H.max() + 1e-8)

        # Convert to tensor
        tensor = torch.tensor(H, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

        # Apply ImageNet normalization
        if self.normalize_imagenet:
            mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    @property
    def output_channels(self) -> int:
        return 3

    @property
    def output_size(self) -> Tuple[int, int]:
        return (self.img_size, self.img_size)
