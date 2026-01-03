"""
Signal-to-spectrogram transform plugins.

Available transforms:
- audio_spectrogram: Mel spectrogram for audio
- iq_spectrogram: Spectrogram for IQ/RF signals

Usage:
    from src import transform_registry

    transform = transform_registry.create("audio_spectrogram", img_size=224)
    spectrogram = transform(waveform, sample_rate)
"""

from src.registry import transform_registry

# Import classes to trigger registration
from src.transforms.audio import AudioToSpectrogram, AudioToSpectrogramRaw
from src.transforms.rf import IQToSpectrogram, IQToConstellation

__all__ = [
    "transform_registry",
    "AudioToSpectrogram",
    "AudioToSpectrogramRaw",
    "IQToSpectrogram",
    "IQToConstellation",
]
