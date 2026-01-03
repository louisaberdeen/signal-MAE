"""
Abstract base class for signal-to-spectrogram transforms.

All transform plugins must inherit from BaseTransform and implement
the required methods for converting raw signals to spectrograms.
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple

import torch
import numpy as np


class BaseTransform(ABC):
    """
    Abstract base class for signal-to-spectrogram transforms.

    Transforms convert raw signals (audio waveforms, IQ samples) to
    spectrogram tensors suitable for vision transformer processing.

    Subclasses must implement:
        - __call__: Transform signal to spectrogram
        - output_channels: Number of output channels
        - output_size: Output (height, width) tuple
    """

    @abstractmethod
    def __call__(
        self,
        signal: Union[np.ndarray, torch.Tensor],
        sample_rate: int
    ) -> torch.Tensor:
        """
        Transform signal to spectrogram tensor.

        Args:
            signal: Input signal (audio waveform or IQ samples)
            sample_rate: Sampling rate in Hz

        Returns:
            Spectrogram tensor [C, H, W]
        """
        pass

    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Return number of output channels."""
        pass

    @property
    @abstractmethod
    def output_size(self) -> Tuple[int, int]:
        """Return (height, width) of output."""
        pass


class ComposedTransform(BaseTransform):
    """
    Chain multiple transforms together.

    Applies transforms in sequence, passing output of each to the next.

    Args:
        transforms: List of transforms to apply in order
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        signal: Union[np.ndarray, torch.Tensor],
        sample_rate: int
    ) -> torch.Tensor:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            signal = transform(signal, sample_rate)
        return signal

    @property
    def output_channels(self) -> int:
        """Return channels of final transform."""
        return self.transforms[-1].output_channels

    @property
    def output_size(self) -> Tuple[int, int]:
        """Return size of final transform."""
        return self.transforms[-1].output_size
