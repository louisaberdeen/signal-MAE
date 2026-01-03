"""
Automatic test generation for plugin components.

TestGenerator creates tests that verify:
1. Interface compliance (ABC method implementation)
2. Architecture compatibility (various input sizes)
3. Plugin registration correctness

Usage:
    python tests/generate_tests.py

This will generate test files in tests/generated/
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGenerator:
    """
    Generate tests for plugin components.

    Creates pytest test files that verify plugin interface compliance,
    architecture compatibility, and registration correctness.
    """

    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path(__file__).parent / "generated"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_interface_tests(self, registry_name: str) -> str:
        """
        Generate interface compliance tests for a registry.

        Args:
            registry_name: Name of the registry ('models', 'data_loaders', etc.)

        Returns:
            Generated test code as string
        """
        required_methods = self._get_required_methods(registry_name)

        test_code = f'''"""
Auto-generated interface tests for {registry_name} registry.

Generated: {datetime.now().isoformat()}
"""

import pytest
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.registry import PluginRegistry


class Test{registry_name.title().replace("_", "")}Interface:
    """Test that all {registry_name} plugins implement required interface."""

    @pytest.fixture
    def registry(self):
        """Get the {registry_name} registry."""
        registry = PluginRegistry.get_registry("{registry_name}")
        if registry is None:
            pytest.skip("{registry_name} registry not found")
        return registry

    def test_registry_exists(self, registry):
        """Test that registry exists and has plugins."""
        assert registry is not None
        assert len(registry) > 0, "Registry is empty"

    @pytest.mark.parametrize("method", {required_methods})
    def test_plugins_have_required_methods(self, registry, method):
        """Test that all plugins have required methods."""
        for key, cls in registry:
            assert hasattr(cls, method), f"{{key}} missing method: {{method}}"

    def test_plugins_not_abstract(self, registry):
        """Test that no plugin is abstract."""
        for key, cls in registry:
            assert not inspect.isabstract(cls), f"{{key}} is abstract"

    def test_plugin_metadata(self, registry):
        """Test that all plugins have metadata."""
        for key in registry.list():
            metadata = registry.get_metadata(key)
            assert 'class_name' in metadata
            assert 'module' in metadata
'''
        return test_code

    def generate_model_architecture_tests(self) -> str:
        """Generate architecture compatibility tests for models."""
        return '''"""
Auto-generated architecture tests for model plugins.

Tests models with various input configurations to ensure compatibility.

Generated: ''' + datetime.now().isoformat() + '''
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.registry import model_registry
from src.config import Config


# Test configurations
ARCHITECTURES = [
    {"img_size": 64, "patch_size": 8, "desc": "tiny"},
    {"img_size": 128, "patch_size": 16, "desc": "small"},
    {"img_size": 224, "patch_size": 16, "desc": "standard"},
]


@pytest.fixture(params=model_registry.list() if len(model_registry.list()) > 0 else ["skip"])
def model_key(request):
    if request.param == "skip":
        pytest.skip("No models registered")
    return request.param


@pytest.fixture(params=ARCHITECTURES, ids=lambda a: a["desc"])
def architecture(request):
    return request.param


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestModelArchitectures:
    """Test model plugins with various input configurations."""

    def test_model_creation(self, model_key, architecture):
        """Test model can be created with different architectures."""
        config = Config(
            img_size=architecture["img_size"],
            patch_size=architecture["patch_size"],
            embed_dim=192,
            encoder_depth=2,
            encoder_heads=3,
            decoder_embed_dim=96,
            decoder_depth=2,
            decoder_heads=3,
        )

        model = model_registry.create(model_key, config)
        assert model is not None

    def test_forward_encoder(self, model_key, architecture, device):
        """Test encoder forward pass."""
        config = Config(
            img_size=architecture["img_size"],
            patch_size=architecture["patch_size"],
            embed_dim=192,
            encoder_depth=2,
            encoder_heads=3,
            decoder_embed_dim=96,
            decoder_depth=2,
            decoder_heads=3,
        )

        model = model_registry.create(model_key, config)
        model = model.to(device)
        model.eval()

        batch = torch.randn(2, 3, config.img_size, config.img_size, device=device)

        with torch.no_grad():
            latent, mask, ids = model.forward_encoder(batch, mask_ratio=0.0)

        expected_patches = config.num_patches
        assert latent.shape == (2, expected_patches + 1, config.embed_dim)

    def test_embedding_extraction(self, model_key, architecture, device):
        """Test embedding extraction with different pooling modes."""
        config = Config(
            img_size=architecture["img_size"],
            patch_size=architecture["patch_size"],
            embed_dim=192,
            encoder_depth=2,
            encoder_heads=3,
            decoder_embed_dim=96,
            decoder_depth=2,
            decoder_heads=3,
        )

        model = model_registry.create(model_key, config)
        model = model.to(device)
        model.eval()

        batch = torch.randn(2, 3, config.img_size, config.img_size, device=device)

        for mode in ["cls", "mean", "cls+mean"]:
            embedding = model.get_embedding(batch, pooling_mode=mode)

            if mode == "cls+mean":
                assert embedding.shape == (2, config.embed_dim * 2)
            else:
                assert embedding.shape == (2, config.embed_dim)

    def test_full_forward(self, model_key, architecture, device):
        """Test full forward pass with reconstruction."""
        config = Config(
            img_size=architecture["img_size"],
            patch_size=architecture["patch_size"],
            embed_dim=192,
            encoder_depth=2,
            encoder_heads=3,
            decoder_embed_dim=96,
            decoder_depth=2,
            decoder_heads=3,
            use_contrastive_loss=False,
            use_uniformity_loss=False,
        )

        model = model_registry.create(model_key, config)
        model = model.to(device)
        model.eval()

        batch = torch.randn(2, 3, config.img_size, config.img_size, device=device)

        with torch.no_grad():
            loss, pred, mask = model(batch)

        assert loss.ndim == 0  # Scalar
        assert pred.shape[0] == 2
        assert mask.shape == (2, config.num_patches)
'''

    def generate_dataloader_tests(self) -> str:
        """Generate tests for data loader plugins."""
        return '''"""
Auto-generated tests for data loader plugins.

Generated: ''' + datetime.now().isoformat() + '''
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.registry import data_loader_registry


@pytest.fixture(params=data_loader_registry.list() if len(data_loader_registry.list()) > 0 else ["skip"])
def loader_key(request):
    if request.param == "skip":
        pytest.skip("No data loaders registered")
    return request.param


class TestDataLoaders:
    """Test data loader plugin interfaces."""

    def test_loader_registered(self, loader_key):
        """Test loader is properly registered."""
        assert loader_key in data_loader_registry

    def test_loader_has_load_metadata(self, loader_key):
        """Test loader has load_metadata method."""
        loader_cls = data_loader_registry.get(loader_key)
        assert hasattr(loader_cls, 'load_metadata')
        assert callable(getattr(loader_cls, 'load_metadata'))

    def test_loader_has_get_sample_paths(self, loader_key):
        """Test loader has get_sample_paths method."""
        loader_cls = data_loader_registry.get(loader_key)
        assert hasattr(loader_cls, 'get_sample_paths')
        assert callable(getattr(loader_cls, 'get_sample_paths'))

    def test_loader_has_validate(self, loader_key):
        """Test loader has validate method."""
        loader_cls = data_loader_registry.get(loader_key)
        assert hasattr(loader_cls, 'validate')
        assert callable(getattr(loader_cls, 'validate'))
'''

    def generate_transform_tests(self) -> str:
        """Generate tests for transform plugins."""
        return '''"""
Auto-generated tests for transform plugins.

Generated: ''' + datetime.now().isoformat() + '''
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.registry import transform_registry


@pytest.fixture(params=transform_registry.list() if len(transform_registry.list()) > 0 else ["skip"])
def transform_key(request):
    if request.param == "skip":
        pytest.skip("No transforms registered")
    return request.param


class TestTransforms:
    """Test transform plugin interfaces."""

    def test_transform_registered(self, transform_key):
        """Test transform is properly registered."""
        assert transform_key in transform_registry

    def test_transform_has_output_channels(self, transform_key):
        """Test transform has output_channels property."""
        transform_cls = transform_registry.get(transform_key)
        # Check it can be instantiated with defaults or minimal args
        try:
            transform = transform_cls()
            assert hasattr(transform, 'output_channels')
        except TypeError:
            # May require arguments, skip this test
            pytest.skip(f"{transform_key} requires constructor arguments")

    def test_transform_has_output_size(self, transform_key):
        """Test transform has output_size property."""
        transform_cls = transform_registry.get(transform_key)
        try:
            transform = transform_cls()
            assert hasattr(transform, 'output_size')
            size = transform.output_size
            assert isinstance(size, tuple)
            assert len(size) == 2
        except TypeError:
            pytest.skip(f"{transform_key} requires constructor arguments")
'''

    def _get_required_methods(self, registry_name: str) -> List[str]:
        """Return required methods for each registry type."""
        methods = {
            "models": ["forward_encoder", "get_embedding", "embed_dim", "num_patches"],
            "data_loaders": ["load_metadata", "get_sample_paths", "validate"],
            "transforms": ["__call__", "output_channels", "output_size"],
            "losses": ["forward"],
        }
        return methods.get(registry_name, [])

    def generate_all_tests(self) -> None:
        """Generate all test files."""
        print(f"Generating tests in {self.output_dir}")

        # Interface tests for each registry
        for registry_name in ["models", "data_loaders", "transforms"]:
            test_code = self.generate_interface_tests(registry_name)
            test_file = self.output_dir / f"test_{registry_name}_interface.py"
            test_file.write_text(test_code)
            print(f"  Created: {test_file.name}")

        # Architecture tests
        arch_tests = self.generate_model_architecture_tests()
        arch_file = self.output_dir / "test_model_architectures.py"
        arch_file.write_text(arch_tests)
        print(f"  Created: {arch_file.name}")

        # Data loader tests
        loader_tests = self.generate_dataloader_tests()
        loader_file = self.output_dir / "test_data_loaders.py"
        loader_file.write_text(loader_tests)
        print(f"  Created: {loader_file.name}")

        # Transform tests
        transform_tests = self.generate_transform_tests()
        transform_file = self.output_dir / "test_transforms.py"
        transform_file.write_text(transform_tests)
        print(f"  Created: {transform_file.name}")

        print(f"\nGenerated {len(list(self.output_dir.glob('*.py')))} test files")


def main():
    """Generate all tests."""
    generator = TestGenerator()
    generator.generate_all_tests()

    print("\nTo run tests:")
    print("  pytest tests/generated/ -v")


if __name__ == "__main__":
    main()
