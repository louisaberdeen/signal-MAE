"""
Audio-to-spectrogram transforms.

Converts audio waveforms to mel spectrograms suitable for
vision transformer processing.
"""

from typing import Union, Tuple, Optional

import numpy as np
import torch

from src.registry import transform_registry
from src.transforms.base import BaseTransform


@transform_registry.register("audio_spectrogram", version="2.0")
class AudioToSpectrogram(BaseTransform):
    """
    Convert audio waveform to mel spectrogram.

    Pipeline:
    1. Resample to target sample rate
    2. Convert to mono if stereo
    3. Pad/crop to target duration
    4. Compute mel spectrogram
    5. Convert to dB scale and normalize
    6. Resize to target image size
    7. Replicate to 3 channels (for ViT compatibility)
    8. Apply ImageNet normalization

    Args:
        sample_rate: Target sample rate (default: 22050)
        n_mels: Number of mel bins (default: 128)
        n_fft: FFT size (default: 2048)
        hop_length: Hop length (default: 512)
        duration: Target duration in seconds (default: 5)
        img_size: Output image size (default: 224)
        normalize_imagenet: Apply ImageNet normalization (default: True)
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: int = 5,
        img_size: int = 224,
        normalize_imagenet: bool = True
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.img_size = img_size
        self.normalize_imagenet = normalize_imagenet

        # Calculate target number of samples
        self.target_samples = sample_rate * duration

    def __call__(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int
    ) -> torch.Tensor:
        """
        Convert waveform to spectrogram tensor.

        Args:
            waveform: Audio waveform (numpy array or tensor)
            sample_rate: Original sample rate

        Returns:
            Spectrogram tensor [3, img_size, img_size]
        """
        import librosa
        from PIL import Image

        # Convert to numpy if tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()

        # Ensure 1D (mono)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )

        # Pad or crop to target length
        if len(waveform) < self.target_samples:
            # Pad with zeros
            padding = self.target_samples - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')
        elif len(waveform) > self.target_samples:
            # Random crop during training, center crop otherwise
            start = (len(waveform) - self.target_samples) // 2
            waveform = waveform[start:start + self.target_samples]

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min())
        mel_spec_norm = mel_spec_norm / (mel_spec_norm.max() + 1e-8)

        # Resize to target size
        img = Image.fromarray((mel_spec_norm * 255).astype(np.uint8))
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to tensor and replicate to 3 channels
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


@transform_registry.register("audio_spectrogram_raw", version="1.0")
class AudioToSpectrogramRaw(AudioToSpectrogram):
    """
    Audio to spectrogram without ImageNet normalization.

    Useful for visualization or when using a different normalization scheme.
    """

    def __init__(self, **kwargs):
        kwargs['normalize_imagenet'] = False
        super().__init__(**kwargs)
