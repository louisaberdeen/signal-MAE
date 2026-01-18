# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **modular audio/signal machine learning framework** with a plugin-based architecture for training masked autoencoders. The project implements **AudioMAE++** and supports extensibility to new datasets (RadioML, custom audio) through a plugin registry system.

**Core Capabilities**:
- Plugin registry for models, data loaders, and transforms
- Self-contained notebook generation for Google Colab
- Automatic test generation for plugin compatibility
- ESC-50 environmental sound classification (2000 clips, 50 classes)
- RadioML RF signal classification support

## Architecture (v2.0)

The project uses a modular `src/` structure with plugin registries:

```
.
├── src/                        # Core framework (v2.0)
│   ├── registry.py            # PluginRegistry class
│   ├── config.py              # Config dataclass
│   ├── models/                # Model plugins
│   │   ├── base.py           # BaseModel, BaseAutoencoder ABCs
│   │   ├── audiomae.py       # @model_registry.register("audiomae++")
│   │   ├── baseline.py       # @model_registry.register("baseline")
│   │   ├── classifier.py     # AudioMAEClassifier
│   │   └── blocks/           # Transformer components
│   ├── data/                  # Data loader plugins
│   │   ├── base.py           # BaseDataLoader ABC
│   │   ├── esc50.py          # @data_loader_registry.register("esc50")
│   │   └── custom.py         # Generic + RF loaders
│   ├── transforms/            # Transform plugins
│   │   ├── base.py           # BaseTransform ABC
│   │   ├── audio.py          # AudioToSpectrogram
│   │   └── rf.py             # IQToSpectrogram
│   ├── training/              # Training utilities
│   │   └── losses.py         # info_nce_loss, uniformity_loss
│   ├── embeddings/            # Embedding utilities
│   │   ├── generator.py      # EmbeddingGenerator
│   │   ├── cache.py          # EmbeddingCache
│   │   └── checkpoint.py     # CheckpointLoader
│   └── utils/                 # Shared utilities
├── tests/                     # Test suite
│   ├── conftest.py           # Pytest fixtures
│   ├── generate_tests.py     # Auto test generation
│   └── generated/            # Auto-generated tests (gitignored)
├── notebooks/                 # Notebook generation
│   ├── generate.py           # NotebookGenerator
│   └── generated/            # Self-contained notebooks (gitignored)
├── data/                      # Dataset storage
│   ├── ESC-50-master/        # ESC-50 dataset
│   ├── imgs/                 # Spectrograms
│   └── embeddings/           # Cached embeddings
├── checkpoints/               # Model checkpoints
└── audiomae.py               # Legacy wrapper (backward compat)
```

## Quick Start

### Environment Setup
```bash
source .venv/bin/activate
```

### Using the Plugin Registry
```python
from src import model_registry, data_loader_registry, transform_registry
from src.config import Config

# Create model
config = Config()
model = model_registry.create("audiomae++", config)

# Create data loader
from pathlib import Path
loader = data_loader_registry.create("esc50", Path("data/ESC-50-master"))
metadata = loader.load_metadata()

# Create transform
transform = transform_registry.create("audio_spectrogram", img_size=224)
```

### Generating Tests
```bash
# Generate plugin tests
python tests/generate_tests.py

# Run generated tests
pytest tests/generated/ -v
```

### Generating Notebooks
```bash
# AudioMAE++ for audio (ESC-50)
python notebooks/generate.py --model audiomae++ --dataset esc50

# SignalMAE baseline for RF signals
python notebooks/generate.py --model signalmae --dataset-config minimal

# SignalMAE++ advanced for RF signals
python notebooks/generate.py --model signalmae++ --dataset-config classification

# List all example commands
python notebooks/generate.py --list-examples
```

## Plugin Development

### Adding a New Model
```python
# src/models/my_model.py
from src.registry import model_registry
from src.models.base import BaseAutoencoder

@model_registry.register("my-model", version="1.0")
class MyModel(BaseAutoencoder):
    def forward_encoder(self, x, mask_ratio=0.0):
        ...
    def get_embedding(self, x, pooling_mode="mean"):
        ...
    # Implement all required ABC methods
```

### Adding a New Data Loader
```python
# src/data/my_dataset.py
from src.registry import data_loader_registry
from src.data.base import BaseDataLoader

@data_loader_registry.register("my-dataset")
class MyDataLoader(BaseDataLoader):
    def load_metadata(self):
        # Return DataFrame with: filepath, label, lat, lon
        ...
    def get_sample_paths(self):
        # Return list of Path objects
        ...
```

### Adding a New Transform
```python
# src/transforms/my_transform.py
from src.registry import transform_registry
from src.transforms.base import BaseTransform

@transform_registry.register("my-transform")
class MyTransform(BaseTransform):
    def __call__(self, signal, sample_rate):
        # Return tensor [3, H, W]
        ...
    @property
    def output_channels(self): return 3
    @property
    def output_size(self): return (224, 224)
```

## Model Architecture

### AudioMAE++ (Default for Audio)
- **Macaron Blocks**: FFN(1/2) -> Attention -> FFN(1/2) sandwich
- **SwiGLU Activation**: Better than GELU
- **RoPE**: Rotary position embeddings
- **Asymmetric**: Large encoder (12L/768D), small decoder (8L/512D)

### SignalMAE (RF Signals)
Two variants available for RF signal processing:

| Model | Features | Use Case |
|-------|----------|----------|
| `signalmae` | Standard ViT-MAE (baseline) | Simple experiments, lower compute |
| `signalmae++` | Macaron, SwiGLU, RoPE | Maximum capacity, best performance |
| `signalmae-small` | Reduced dimensions | Quick prototyping, limited GPU |

### Configuration
```python
from src.config import Config, create_rf_config

# AudioMAE++ configuration
config = Config(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    encoder_depth=12,
    decoder_depth=8,
    mask_ratio=0.75,
    use_macaron=True,
    use_swiglu=True,
    use_rope=True,
)

# SignalMAE configurations
config_baseline = create_rf_config(size="base", advanced=False)  # SignalMAE
config_advanced = create_rf_config(size="base", advanced=True)   # SignalMAE++
config_small = create_rf_config(size="small")                    # SignalMAE-Small
```

### Using SignalMAE Models
```python
from src.models.signalmae import SignalMAE, SignalMAEPlusPlus, SignalMAESmall
from src.config import create_rf_config

# Baseline model (standard ViT-MAE)
config = create_rf_config(size="base", advanced=False)
model = SignalMAE(config)

# Advanced model (Macaron + SwiGLU + RoPE)
config = create_rf_config(size="base", advanced=True)
model = SignalMAEPlusPlus(config)

# Small model for fast experiments
config = create_rf_config(size="small")
model = SignalMAESmall(config)
```

## Training Workflow

1. **Pre-training** (masked reconstruction):
```python
model = model_registry.create("audiomae++", config)
loss, pred, mask = model(spectrograms)
```

2. **Fine-tuning** (classification):
```python
from src.models.classifier import AudioMAEClassifier
classifier = AudioMAEClassifier(model, num_classes=50)
logits = classifier(spectrograms)
```

3. **Embedding extraction**:
```python
from src.embeddings import EmbeddingGenerator
generator = EmbeddingGenerator(model, device="cuda")
embeddings = generator.extract_embedding(spectrograms)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/registry.py` | Plugin registration system |
| `src/config.py` | Configuration dataclass + `create_rf_config()` |
| `src/models/audiomae.py` | AudioMAE++ implementation |
| `src/models/signalmae.py` | SignalMAE, SignalMAE++, SignalMAESmall |
| `src/data/esc50.py` | ESC-50 data loader |
| `src/data/torchsig_config.py` | TorchSig dataset configuration |
| `src/data/torchsig_generator.py` | RF dataset generator |
| `src/training/rf_trainer.py` | RF signal training pipeline |
| `tests/generate_tests.py` | Test generation |
| `notebooks/generate.py` | Notebook generation (Audio + RF) |
| `scripts/generate_rf_dataset.py` | CLI for RF dataset generation |

## Code Style

- **No Emojis**: Do not use emojis anywhere in the codebase (code, comments, docstrings, commit messages, or generated outputs). Keep all communication professional and text-based.
- **Config-Based Parameters**: All training parameters should be read from the Config object, not hardcoded.
- **Docstrings**: Use clear, concise docstrings following Google style.
- **Type Hints**: Include type hints for function parameters and return values.

## Important Notes

- **Plugin Registration**: All plugins auto-register when imported. Ensure modules are imported in `__init__.py`.
- **Test Generation**: Run after adding new plugins to verify interface compliance.
- **Notebook Generation**: Creates self-contained notebooks with code embedded inline.
- **Backward Compatibility**: `audiomae.py` at root provides legacy imports.
- **GPU Memory**: Batch size 16 needs ~8-12GB VRAM. Reduce for smaller GPUs.

## FiftyOne Visualization

```python
import fiftyone as fo
dataset = fo.load_dataset("esc50_audiomae")
session = fo.launch_app(dataset)

# Similarity search
similar = dataset.sort_by_similarity(sample_id, k=10)
```

## Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Generate tests after adding plugins
python tests/generate_tests.py

# Generate AudioMAE++ notebook (audio)
python notebooks/generate.py --model audiomae++ --dataset esc50

# Generate SignalMAE notebooks (RF signals)
python notebooks/generate.py --model signalmae --dataset-config minimal
python notebooks/generate.py --model signalmae++ --dataset-config classification

# List available modules
python notebooks/generate.py --list-modules

# Show example commands for notebook generation
python notebooks/generate.py --list-examples

# Generate RF dataset (requires TorchSig)
python scripts/generate_rf_dataset.py --config classification --output data/rf/
```
