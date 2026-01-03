# AudioMAE++ Framework

A modular audio/signal machine learning framework with plugin-based architecture for training masked autoencoders. Supports ESC-50 environmental sound classification and extensibility to RF signals (RadioML).

## Features

- **Plugin Registry System**: Decorator-based registration for models, data loaders, and transforms
- **Self-Contained Notebooks**: Generate portable notebooks for Google Colab
- **Automatic Test Generation**: Verify plugin interface compliance
- **FiftyOne Integration**: Visualize embeddings with similarity search and UMAP/t-SNE

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Verify plugins are registered
python -c "from src import model_registry; print(model_registry.list())"
# Output: ['audiomae++', 'baseline']
```

## Project Structure

```
.
├── src/                        # Core framework
│   ├── registry.py            # PluginRegistry class
│   ├── config.py              # Config dataclass
│   ├── models/                # Model plugins
│   │   ├── audiomae.py       # AudioMAE++ implementation
│   │   ├── baseline.py       # Baseline MAE
│   │   └── classifier.py     # Classification wrapper
│   ├── data/                  # Data loader plugins
│   │   ├── esc50.py          # ESC-50 loader
│   │   └── custom.py         # Generic + RF loaders
│   ├── transforms/            # Transform plugins
│   │   ├── audio.py          # Audio spectrograms
│   │   └── rf.py             # RF spectrograms
│   ├── training/              # Loss functions
│   └── embeddings/            # Embedding utilities
├── tests/                     # Test suite
│   ├── generate_tests.py     # Auto test generation
│   └── generated/            # Generated tests (gitignored)
├── notebooks/                 # Notebook generation
│   ├── generate.py           # NotebookGenerator
│   └── generated/            # Generated notebooks (gitignored)
├── data/                      # Datasets
│   └── ESC-50-master/        # ESC-50 dataset
└── checkpoints/               # Model checkpoints
```

## Usage

### Using the Plugin Registry

```python
from src import model_registry, data_loader_registry, transform_registry
from src.config import Config

# Create a model
config = Config(img_size=224, patch_size=16, embed_dim=768)
model = model_registry.create("audiomae++", config)

# Create a data loader
from pathlib import Path
loader = data_loader_registry.create("esc50", Path("data/ESC-50-master"))
metadata = loader.load_metadata()

# Create a transform
transform = transform_registry.create("audio_spectrogram", img_size=224)
```

### Extract Embeddings

```python
import torch
from src import model_registry
from src.config import Config

config = Config()
model = model_registry.create("audiomae++", config)
model.eval()

# Input: batch of spectrograms [B, 3, 224, 224]
x = torch.randn(4, 3, 224, 224)

# Get embeddings
embedding = model.get_embedding(x, pooling_mode="mean")  # [4, 768]
```

### Fine-tune for Classification

```python
from src.models.classifier import AudioMAEClassifier

# Wrap pretrained model for classification
classifier = AudioMAEClassifier(model, num_classes=50, freeze_encoder=True)
logits = classifier(spectrograms)  # [B, 50]
```

## Commands

### Generate Tests

After adding new plugins, generate tests to verify interface compliance:

```bash
python tests/generate_tests.py
```

This creates test files in `tests/generated/`:
- `test_models_interface.py` - Model ABC compliance
- `test_data_loaders_interface.py` - Data loader ABC compliance
- `test_transforms_interface.py` - Transform ABC compliance
- `test_model_architectures.py` - Various input size compatibility

### Run Tests

```bash
# Run all generated tests
python -m pytest tests/generated/ -v

# Run specific test file
python -m pytest tests/generated/test_model_architectures.py -v

# Run with coverage
python -m pytest tests/generated/ --cov=src
```

### Generate Training Notebooks

Create self-contained notebooks for Google Colab:

```bash
# Generate notebook for AudioMAE++ on ESC-50
python notebooks/generate.py --model audiomae++ --dataset esc50

# List available modules
python notebooks/generate.py --list-modules
```

Generated notebooks are saved to `notebooks/generated/` and contain all code inline (no external imports required).

## Adding New Plugins

### New Model

```python
# src/models/my_model.py
from src.registry import model_registry
from src.models.base import BaseAutoencoder

@model_registry.register("my-model", version="1.0")
class MyModel(BaseAutoencoder):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ... build model

    def forward_encoder(self, x, mask_ratio=0.75):
        # Return: latent, mask, ids_restore
        ...

    def get_embedding(self, x, pooling_mode="mean"):
        # Return: embedding [B, embed_dim]
        ...

    @property
    def embed_dim(self): return self.config.embed_dim

    @property
    def num_patches(self): return self.config.num_patches
```

### New Data Loader

```python
# src/data/my_dataset.py
from src.registry import data_loader_registry
from src.data.base import BaseDataLoader

@data_loader_registry.register("my-dataset")
class MyDataLoader(BaseDataLoader):
    def __init__(self, data_root):
        self.data_root = data_root

    def load_metadata(self):
        # Return DataFrame with: filepath, label, lat, lon
        ...

    def get_sample_paths(self):
        # Return list of Path objects
        ...

    def validate(self):
        # Return True if dataset is valid
        ...
```

### New Transform

```python
# src/transforms/my_transform.py
from src.registry import transform_registry
from src.transforms.base import BaseTransform

@transform_registry.register("my-transform")
class MyTransform(BaseTransform):
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, signal, sample_rate):
        # Return tensor [3, H, W]
        ...

    @property
    def output_channels(self): return 3

    @property
    def output_size(self): return (self.img_size, self.img_size)
```

After adding a plugin, import it in the corresponding `__init__.py` to trigger registration.

## Available Plugins

### Models
| Key | Description |
|-----|-------------|
| `audiomae++` | AudioMAE++ with Macaron blocks, SwiGLU, RoPE |
| `baseline` | Standard ViT-MAE for comparison |

### Data Loaders
| Key | Description |
|-----|-------------|
| `esc50` | ESC-50 environmental sounds (2000 clips, 50 classes) |
| `custom` | Generic audio dataset loader |
| `rf` | RF/IQ signal dataset loader |

### Transforms
| Key | Description |
|-----|-------------|
| `audio_spectrogram` | Audio to mel spectrogram (3-channel RGB) |
| `audio_spectrogram_raw` | Audio to mel spectrogram (1-channel) |
| `iq_spectrogram` | IQ signal to spectrogram |
| `iq_constellation` | IQ signal to constellation diagram |

## Configuration

Key configuration options in `src/config.py`:

```python
from src.config import Config

config = Config(
    # Audio processing
    sample_rate=22050,
    n_mels=128,
    audio_duration=5,

    # Model architecture
    img_size=224,
    patch_size=16,
    embed_dim=768,
    encoder_depth=12,
    decoder_depth=8,

    # Training
    mask_ratio=0.75,
    use_contrastive_loss=True,

    # Architecture variants
    use_macaron=True,    # Macaron-style blocks
    use_swiglu=True,     # SwiGLU activation
    use_rope=True,       # Rotary position embeddings
)
```

## FiftyOne Visualization

After generating embeddings, visualize with FiftyOne:

```python
import fiftyone as fo

# Load dataset
dataset = fo.load_dataset("esc50_audiomae")

# Launch app
session = fo.launch_app(dataset)

# Similarity search
similar = dataset.sort_by_similarity(sample_id, k=10)
```

## License

MIT
