"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Get available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_config():
    """Create a small config for fast testing."""
    from src.config import Config
    return Config(
        embed_dim=192,
        encoder_depth=2,
        encoder_heads=3,
        decoder_embed_dim=96,
        decoder_depth=2,
        decoder_heads=3,
        img_size=64,
        patch_size=8,
    )


@pytest.fixture
def sample_batch(small_config, device):
    """Create a sample batch of spectrograms."""
    batch_size = 2
    return torch.randn(
        batch_size, 3,
        small_config.img_size,
        small_config.img_size,
        device=device
    )


@pytest.fixture
def sample_labels(device):
    """Create sample labels for contrastive loss."""
    return torch.tensor([0, 0, 1, 1], device=device)


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_root(project_root):
    """Get data directory path."""
    return project_root / "data"


@pytest.fixture
def esc50_path(data_root):
    """Get ESC-50 dataset path."""
    path = data_root / "ESC-50-master"
    if not path.exists():
        pytest.skip("ESC-50 dataset not found")
    return path
