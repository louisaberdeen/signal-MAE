"""
Notebook generation utilities.

NotebookGenerator creates self-contained Jupyter notebooks by embedding
code from src/ modules directly into notebook cells. This enables
notebooks to be uploaded to Google Colab without external dependencies.

Usage:
    python notebooks/generate.py --model audiomae++ --dataset esc50 --output training.ipynb
"""

import ast
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class NotebookGenerator:
    """
    Generate self-contained notebooks from src/ modules.

    Extracts Python code from source files and embeds it inline
    in Jupyter notebook cells, creating portable notebooks that
    can run on Google Colab without local dependencies.
    """

    def __init__(
        self,
        src_dir: Path = None,
        template_dir: Path = None,
        output_dir: Path = None
    ):
        base = Path(__file__).parent.parent
        self.src_dir = src_dir or base / "src"
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.output_dir = output_dir or Path(__file__).parent / "generated"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_module_code(
        self,
        module_path: Path,
        exclude_imports: bool = True,
        exclude_main: bool = True,
        exclude_docstring: bool = False
    ) -> str:
        """
        Extract code from a Python module for inline embedding.

        Args:
            module_path: Path to .py file
            exclude_imports: Remove import statements
            exclude_main: Remove if __name__ == "__main__" block
            exclude_docstring: Remove module-level docstring

        Returns:
            Cleaned code string
        """
        with open(module_path) as f:
            source = f.read()

        if not exclude_imports and not exclude_main and not exclude_docstring:
            return source

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source  # Return as-is if parsing fails

        lines = source.split('\n')
        exclude_ranges = []

        for node in ast.walk(tree):
            # Exclude imports
            if exclude_imports and isinstance(node, (ast.Import, ast.ImportFrom)):
                exclude_ranges.append((node.lineno - 1, node.end_lineno))

            # Exclude if __name__ == "__main__"
            if exclude_main and isinstance(node, ast.If):
                if (isinstance(node.test, ast.Compare) and
                    hasattr(node.test.left, 'id') and
                    node.test.left.id == '__name__'):
                    exclude_ranges.append((node.lineno - 1, node.end_lineno))

        # Exclude module docstring
        if exclude_docstring and tree.body:
            first = tree.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                exclude_ranges.append((first.lineno - 1, first.end_lineno))

        # Build filtered code
        kept_lines = []
        for i, line in enumerate(lines):
            if not any(start <= i < end for start, end in exclude_ranges):
                kept_lines.append(line)

        # Remove excessive blank lines
        result = '\n'.join(kept_lines)
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')

        return result.strip()

    def _create_markdown_cell(self, source: str) -> Dict:
        """Create a markdown cell."""
        lines = source.split('\n')
        # Add newlines to all but the last line
        source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source_lines
        }

    def _create_code_cell(self, source: str) -> Dict:
        """Create a code cell."""
        lines = source.split('\n')
        # Add newlines to all but the last line
        source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
        return {
            "cell_type": "code",
            "metadata": {},
            "source": source_lines,
            "outputs": [],
            "execution_count": None
        }

    def generate_training_notebook(
        self,
        model_key: str = "audiomae++",
        dataset_key: str = "esc50",
        title: str = None,
        description: str = None
    ) -> Path:
        """
        Generate a complete training notebook.

        Args:
            model_key: Model plugin key
            dataset_key: Data loader plugin key
            title: Notebook title
            description: Notebook description

        Returns:
            Path to generated notebook
        """
        if title is None:
            title = f"{model_key.upper()} Training on {dataset_key.upper()}"

        if description is None:
            description = f"Self-contained training notebook for {model_key} model on {dataset_key} dataset."

        cells = []

        # Header
        cells.append(self._create_markdown_cell(f"""# {title}

{description}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model**: `{model_key}`
**Dataset**: `{dataset_key}`

This notebook is self-contained and can be uploaded to Google Colab.
"""))

        # Installation cell
        cells.append(self._create_markdown_cell("## 1. Installation"))
        cells.append(self._create_code_cell("""# Uncomment to install dependencies on Google Colab
# !pip install torch torchvision librosa einops tqdm mlflow fiftyone scipy pillow
"""))

        # Imports cell
        cells.append(self._create_markdown_cell("## 2. Imports"))
        cells.append(self._create_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field, asdict
from tqdm.auto import tqdm
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
"""))

        # Configuration
        cells.append(self._create_markdown_cell("## 3. Configuration"))
        config_code = self.extract_module_code(
            self.src_dir / "config.py",
            exclude_imports=True,
            exclude_main=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(config_code + "\n\n# Create configuration\nconfig = Config()"))

        # Model components
        cells.append(self._create_markdown_cell("## 4. Model Components"))

        # Losses
        cells.append(self._create_markdown_cell("### 4.1 Loss Functions"))
        losses_code = self.extract_module_code(
            self.src_dir / "training" / "losses.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(losses_code))

        # Attention
        cells.append(self._create_markdown_cell("### 4.2 Attention & RoPE"))
        attention_code = self.extract_module_code(
            self.src_dir / "models" / "blocks" / "attention.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(attention_code))

        # FFN
        cells.append(self._create_markdown_cell("### 4.3 Feed-Forward Networks"))
        ffn_code = self.extract_module_code(
            self.src_dir / "models" / "blocks" / "ffn.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(ffn_code))

        # Transformer blocks
        cells.append(self._create_markdown_cell("### 4.4 Transformer Blocks"))
        transformer_code = self.extract_module_code(
            self.src_dir / "models" / "blocks" / "transformer.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        # Remove local imports
        transformer_code = transformer_code.replace(
            "from src.models.blocks.attention import Attention, RotaryPositionEmbedding, apply_rotary_pos_emb",
            "# Attention already defined above"
        ).replace(
            "from src.models.blocks.ffn import SwiGLU, StandardFFN",
            "# FFN already defined above"
        )
        cells.append(self._create_code_cell(transformer_code))

        # Training functions
        cells.append(self._create_markdown_cell("## 5. Training Functions"))
        cells.append(self._create_code_cell("""def train_epoch(model, dataloader, optimizer, device, epoch):
    \"\"\"Train for one epoch.\"\"\"
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        if hasattr(model.config, 'use_contrastive_loss') and model.config.use_contrastive_loss:
            loss, pred, mask, loss_dict = model(data, labels=labels)
        else:
            loss, pred, mask = model(data)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    \"\"\"Validate the model.\"\"\"
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            loss, pred, mask = model(data)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    \"\"\"Save model checkpoint.\"\"\"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config.to_dict() if hasattr(model.config, 'to_dict') else None,
    }, path)
    print(f"Saved checkpoint to {path}")
"""))

        # Dataset class
        cells.append(self._create_markdown_cell("## 6. Dataset"))
        cells.append(self._create_code_cell("""class SpectrogramDataset(torch.utils.data.Dataset):
    \"\"\"Dataset for loading precomputed spectrograms.\"\"\"

    def __init__(self, spectrogram_dir, metadata_df=None, transform=None):
        self.spectrogram_dir = Path(spectrogram_dir)
        self.transform = transform

        # Find all .npy spectrogram files
        self.files = sorted(list(self.spectrogram_dir.glob("*.npy")))

        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {spectrogram_dir}")

        # Build label mapping from metadata if provided
        self.labels = {}
        if metadata_df is not None:
            for _, row in metadata_df.iterrows():
                filename = Path(row['filename']).stem
                self.labels[filename] = row.get('target', 0)

        print(f"Loaded {len(self.files)} spectrograms")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec_path = self.files[idx]

        # Load spectrogram
        spec = np.load(spec_path)

        # Convert to tensor and ensure 3 channels
        if spec.ndim == 2:
            spec = np.stack([spec, spec, spec], axis=0)
        elif spec.ndim == 3 and spec.shape[0] == 1:
            spec = np.repeat(spec, 3, axis=0)

        spec = torch.from_numpy(spec).float()

        # Resize if needed
        if spec.shape[1] != config.img_size or spec.shape[2] != config.img_size:
            spec = F.interpolate(
                spec.unsqueeze(0),
                size=(config.img_size, config.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Get label
        filename = spec_path.stem
        label = self.labels.get(filename, 0)

        return spec, label
"""))

        # Model architecture
        cells.append(self._create_markdown_cell("## 7. Model Architecture"))

        # Add the AudioMAE model code
        audiomae_code = self.extract_module_code(
            self.src_dir / "models" / "audiomae.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        # Remove src imports
        audiomae_code = audiomae_code.replace(
            "from src.registry import model_registry",
            "# Registry not needed in notebook"
        ).replace(
            "from src.config import Config",
            "# Config already defined above"
        ).replace(
            "from src.models.base import BaseAutoencoder",
            "# Using nn.Module directly"
        ).replace(
            "from src.models.blocks.attention import Attention, RotaryPositionEmbedding",
            "# Attention already defined above"
        ).replace(
            "from src.models.blocks.ffn import SwiGLU, StandardFFN",
            "# FFN already defined above"
        ).replace(
            "from src.models.blocks.transformer import TransformerBlock, MacaronBlock",
            "# Transformer blocks already defined above"
        ).replace(
            "from src.training.losses import info_nce_loss, uniformity_loss",
            "# Losses already defined above"
        ).replace(
            "@model_registry.register(\"audiomae++\", version=\"2.0\")",
            "# AudioMAE++ Model"
        ).replace(
            "(BaseAutoencoder)",
            "(nn.Module)"
        )
        cells.append(self._create_code_cell(audiomae_code))

        # Data download and preparation
        cells.append(self._create_markdown_cell("## 8. Data Download & Preparation"))
        cells.append(self._create_code_cell("""# ============================================
# PATHS AND DIRECTORIES
# ============================================

# Paths
DATA_ROOT = Path("data")
ESC50_DIR = DATA_ROOT / "ESC-50-master"
AUDIO_DIR = ESC50_DIR / "audio"
SPECTROGRAM_DIR = DATA_ROOT / "spectrograms"
METADATA_CSV = ESC50_DIR / "meta" / "esc50.csv"
CHECKPOINT_DIR = Path("checkpoints")

# Create directories
DATA_ROOT.mkdir(exist_ok=True)
SPECTROGRAM_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Display training configuration
print("Training Configuration:")
print(f"  Epochs: {config.epochs}")
print(f"  Batch Size: {config.batch_size}")
print(f"  Learning Rate: {config.learning_rate}")
print(f"  Warmup Epochs: {config.warmup_epochs}")
print(f"  Checkpoint Interval: {config.checkpoint_interval}")
print(f"  Mask Ratio: {config.mask_ratio}")
print(f"  Weight Decay: {config.weight_decay}")
"""))

        cells.append(self._create_markdown_cell("### 8.1 Download ESC-50 Dataset"))
        cells.append(self._create_code_cell("""import subprocess
import zipfile
import os

def download_esc50():
    \"\"\"Download ESC-50 dataset if not present.\"\"\"
    if AUDIO_DIR.exists() and len(list(AUDIO_DIR.glob("*.wav"))) > 0:
        print(f"ESC-50 already downloaded ({len(list(AUDIO_DIR.glob('*.wav')))} files)")
        return True

    print("Downloading ESC-50 dataset...")
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = DATA_ROOT / "ESC-50-master.zip"

    try:
        # Download using wget or curl
        if subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
            subprocess.run(["wget", "-q", "--show-progress", url, "-O", str(zip_path)], check=True)
        else:
            subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)

        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_ROOT)

        # Clean up
        zip_path.unlink()
        print(f"Downloaded ESC-50 to {ESC50_DIR}")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from: https://github.com/karoldvl/ESC-50")
        return False

# Download dataset
download_esc50()
"""))

        cells.append(self._create_markdown_cell("### 8.2 Generate Spectrograms"))
        cells.append(self._create_code_cell("""import librosa
import librosa.display
from scipy import signal as scipy_signal

def audio_to_spectrogram(audio_path, config):
    \"\"\"Convert audio file to mel spectrogram.\"\"\"
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.sample_rate, duration=config.audio_duration)

    # Pad if too short
    target_length = config.sample_rate * config.audio_duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmax=config.sample_rate // 2
    )

    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    # Resize to target size
    mel_spec_resized = np.array(
        PIL.Image.fromarray((mel_spec_norm * 255).astype(np.uint8)).resize(
            (config.img_size, config.img_size),
            PIL.Image.Resampling.BILINEAR
        )
    ) / 255.0

    # Stack to 3 channels
    mel_spec_3ch = np.stack([mel_spec_resized, mel_spec_resized, mel_spec_resized], axis=0)

    return mel_spec_3ch.astype(np.float32)


def generate_spectrograms(audio_dir, output_dir, config):
    \"\"\"Generate spectrograms for all audio files.\"\"\"
    audio_files = list(Path(audio_dir).glob("*.wav"))

    if len(audio_files) == 0:
        print(f"No audio files found in {audio_dir}")
        return False

    # Check if already generated
    existing = list(Path(output_dir).glob("*.npy"))
    if len(existing) >= len(audio_files):
        print(f"Spectrograms already generated ({len(existing)} files)")
        return True

    print(f"Generating spectrograms for {len(audio_files)} audio files...")

    for audio_path in tqdm(audio_files, desc="Generating spectrograms"):
        output_path = Path(output_dir) / f"{audio_path.stem}.npy"

        if output_path.exists():
            continue

        try:
            spec = audio_to_spectrogram(audio_path, config)
            np.save(output_path, spec)
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")

    print(f"Generated {len(list(Path(output_dir).glob('*.npy')))} spectrograms")
    return True


# Import PIL for resizing
try:
    import PIL.Image
except ImportError:
    print("Installing Pillow...")
    subprocess.run(["pip", "install", "pillow", "-q"])
    import PIL.Image

# Install librosa if needed
try:
    import librosa
except ImportError:
    print("Installing librosa...")
    subprocess.run(["pip", "install", "librosa", "-q"])
    import librosa

# Generate spectrograms
if AUDIO_DIR.exists():
    generate_spectrograms(AUDIO_DIR, SPECTROGRAM_DIR, config)
else:
    print(f"Audio directory not found: {AUDIO_DIR}")
    print("Please run the download cell first")
"""))

        cells.append(self._create_markdown_cell("### 8.3 Load Dataset"))
        cells.append(self._create_code_cell("""import pandas as pd

# Load metadata
metadata_df = None
if METADATA_CSV.exists():
    metadata_df = pd.read_csv(METADATA_CSV)
    print(f"Loaded metadata with {len(metadata_df)} entries")
    print(f"Classes: {metadata_df['category'].nunique()} categories")

# Create dataset
dataset = SpectrogramDataset(SPECTROGRAM_DIR, metadata_df)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True if device == "cuda" else False
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
"""))

        cells.append(self._create_markdown_cell("### 8.4 Create Model"))
        cells.append(self._create_code_cell("""# Create model with configuration
model = AudioMAEPlusPlus(config)
model = model.to(device)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
"""))

        cells.append(self._create_markdown_cell("### 8.5 Setup Optimizer"))
        cells.append(self._create_code_cell("""# AdamW optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95)
)

# Learning rate scheduler with warmup
def get_lr(epoch, config):
    \"\"\"Get learning rate for given epoch using warmup + cosine decay.\"\"\"
    if epoch < config.warmup_epochs:
        # Linear warmup
        return config.learning_rate * (epoch + 1) / config.warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: get_lr(epoch, config) / config.learning_rate
)
"""))

        cells.append(self._create_markdown_cell("### 8.6 Training Loop"))
        cells.append(self._create_code_cell("""# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'lr': []
}

print(f"Starting training for {config.epochs} epochs...")
print(f"Device: {device}")
print("-" * 50)

best_val_loss = float('inf')

for epoch in range(config.epochs):
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)

    # Train
    train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
    history['train_loss'].append(train_loss)

    # Validate
    val_loss = validate(model, val_loader, device)
    history['val_loss'].append(val_loss)

    # Update learning rate
    scheduler.step()

    # Print progress
    print(f"Epoch {epoch + 1}/{config.epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"LR: {current_lr:.2e}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            Path(CHECKPOINT_DIR) / "best_model.pt"
        )

    # Save periodic checkpoint
    if (epoch + 1) % config.checkpoint_interval == 0:
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            Path(CHECKPOINT_DIR) / f"checkpoint_epoch_{epoch + 1}.pt"
        )

print("-" * 50)
print(f"Training complete! Best val loss: {best_val_loss:.4f}")
"""))

        cells.append(self._create_markdown_cell("## 9. Visualize Results"))
        cells.append(self._create_code_cell("""# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Learning rate
axes[1].plot(history['lr'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Learning Rate Schedule')
axes[1].grid(True)

plt.tight_layout()
plt.show()
"""))

        cells.append(self._create_markdown_cell("## 10. Test Reconstruction"))
        cells.append(self._create_code_cell("""# Visualize reconstruction on a sample
model.eval()

# Get a sample batch
sample_batch, _ = next(iter(val_loader))
sample_batch = sample_batch.to(device)

with torch.no_grad():
    # Forward pass with masking
    loss, pred, mask = model(sample_batch[:4])

    # Get reconstruction
    # Reshape prediction to image
    patch_size = config.patch_size
    h = w = config.img_size // patch_size

    pred_img = pred.reshape(-1, h, w, patch_size, patch_size, 3)
    pred_img = pred_img.permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, config.img_size, config.img_size)

# Plot original vs reconstruction
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    # Original
    orig = sample_batch[i].cpu().permute(1, 2, 0).numpy()
    orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
    axes[0, i].imshow(orig)
    axes[0, i].set_title(f'Original {i+1}')
    axes[0, i].axis('off')

    # Reconstruction
    recon = pred_img[i].cpu().permute(1, 2, 0).numpy()
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    axes[1, i].imshow(recon)
    axes[1, i].set_title(f'Reconstruction {i+1}')
    axes[1, i].axis('off')

plt.suptitle(f'Reconstruction (Loss: {loss.item():.4f})')
plt.tight_layout()
plt.show()
"""))

        cells.append(self._create_markdown_cell("""## Next Steps

After pre-training, you can:

1. **Fine-tune for classification**: Use the pretrained encoder with a classification head
2. **Extract embeddings**: Use `model.get_embedding(x)` to get 768-dim embeddings
3. **Visualize with FiftyOne**: Load embeddings into FiftyOne for similarity search

```python
# Example: Extract embeddings
model.eval()
with torch.no_grad():
    embeddings = model.get_embedding(spectrograms, pooling_mode="mean")
```
"""))

        # Save as notebook
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                }
            },
            "cells": cells
        }

        # Write notebook
        output_name = f"training_{model_key.replace('+', 'plus')}_{dataset_key}.ipynb"
        output_path = self.output_dir / output_name

        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)

        print(f"Generated: {output_path}")
        return output_path

    def list_available_modules(self) -> Dict[str, List[str]]:
        """List available modules for embedding."""
        modules = {}

        for subdir in ['models', 'data', 'transforms', 'training', 'embeddings', 'utils']:
            subpath = self.src_dir / subdir
            if subpath.exists():
                modules[subdir] = [f.name for f in subpath.glob('*.py') if f.name != '__init__.py']

        return modules

    def generate_signalmae_notebook(
        self,
        model_key: str = "signalmae",
        dataset_config: str = "minimal",
        title: str = None,
        description: str = None
    ) -> Path:
        """
        Generate a SignalMAE training notebook for RF signals.

        Args:
            model_key: Model type - "signalmae" (baseline) or "signalmae++" (advanced)
            dataset_config: TorchSig config preset - "minimal", "classification", or "detection"
            title: Notebook title
            description: Notebook description

        Returns:
            Path to generated notebook
        """
        is_advanced = "++" in model_key or "plus" in model_key.lower()
        model_display = "SignalMAE++" if is_advanced else "SignalMAE"

        if title is None:
            title = f"{model_display} Training on RF Signals ({dataset_config})"

        if description is None:
            description = f"""Self-contained training notebook for {model_display} model on synthetic RF signals.

**Model Features**:
- {'Macaron blocks, SwiGLU activation, RoPE embeddings' if is_advanced else 'Standard ViT-MAE architecture (baseline)'}
- Masked autoencoder pre-training on RF spectrograms
- Supports modulation classification fine-tuning

**Dataset**: Synthetic RF signals via TorchSig ({dataset_config} configuration)
"""

        cells = []

        # Header
        cells.append(self._create_markdown_cell(f"""# {title}

{description}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model**: `{model_key}`
**Dataset Config**: `{dataset_config}`

This notebook is self-contained and can be uploaded to Google Colab.
"""))

        # Installation
        cells.append(self._create_markdown_cell("## 1. Installation"))
        cells.append(self._create_code_cell("""# Install dependencies
# Uncomment for Google Colab
# !pip install torch torchvision torchaudio einops tqdm scipy pillow matplotlib numpy pandas

# Optional: Install TorchSig for real RF signal generation
# !pip install torchsig

import subprocess
import sys

def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or package])

# Core dependencies
install_if_missing("torch")
install_if_missing("einops")
install_if_missing("tqdm")
install_if_missing("PIL", "pillow")
"""))

        # Imports
        cells.append(self._create_markdown_cell("## 2. Imports"))
        cells.append(self._create_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from tqdm.auto import tqdm
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Check for TorchSig
TORCHSIG_AVAILABLE = False
try:
    import torchsig
    TORCHSIG_AVAILABLE = True
    print("TorchSig available - will generate real RF signals")
except ImportError:
    print("TorchSig not available - will use synthetic fallback data")
"""))

        # Configuration
        cells.append(self._create_markdown_cell("## 3. Configuration"))
        config_code = self.extract_module_code(
            self.src_dir / "config.py",
            exclude_imports=True,
            exclude_main=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(config_code))

        # Model configuration cell
        cells.append(self._create_markdown_cell("### 3.1 Create RF Model Configuration"))
        if is_advanced:
            cells.append(self._create_code_cell("""# SignalMAE++ Configuration (advanced features enabled)
config = create_rf_config(size="base", advanced=True)

print("SignalMAE++ Configuration:")
print(f"  - Image size: {config.img_size}x{config.img_size}")
print(f"  - Patch size: {config.patch_size}")
print(f"  - Embed dim: {config.embed_dim}")
print(f"  - Encoder: {config.encoder_depth} layers, {config.encoder_heads} heads")
print(f"  - Decoder: {config.decoder_depth} layers, {config.decoder_heads} heads")
print(f"  - Mask ratio: {config.mask_ratio}")
print(f"  - Macaron blocks: {config.use_macaron}")
print(f"  - SwiGLU activation: {config.use_swiglu}")
print(f"  - RoPE embeddings: {config.use_rope}")
"""))
        else:
            cells.append(self._create_code_cell("""# SignalMAE Configuration (baseline - no advanced features)
config = create_rf_config(size="base", advanced=False)

print("SignalMAE Baseline Configuration:")
print(f"  - Image size: {config.img_size}x{config.img_size}")
print(f"  - Patch size: {config.patch_size}")
print(f"  - Embed dim: {config.embed_dim}")
print(f"  - Encoder: {config.encoder_depth} layers, {config.encoder_heads} heads")
print(f"  - Decoder: {config.decoder_depth} layers, {config.decoder_heads} heads")
print(f"  - Mask ratio: {config.mask_ratio}")
print(f"  - Advanced features: DISABLED (standard ViT-MAE)")
"""))

        # TorchSig Configuration
        cells.append(self._create_markdown_cell("### 3.2 TorchSig Dataset Configuration"))
        torchsig_config_code = self.extract_module_code(
            self.src_dir / "data" / "torchsig_config.py",
            exclude_imports=True,
            exclude_main=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(torchsig_config_code))

        # Dataset config selection
        dataset_configs = {
            "minimal": """# Minimal test configuration (fast training, few samples)
dataset_config = TorchSigConfig.minimal_test_preset()
print(f"Dataset: {dataset_config.num_samples} samples, {len(dataset_config.modulations)} modulations")
print(f"Modulations: {dataset_config.modulations}")""",
            "classification": """# Classification preset (good for modulation recognition)
dataset_config = TorchSigConfig.classification_preset()
print(f"Dataset: {dataset_config.num_samples} samples, {len(dataset_config.modulations)} modulations")
print(f"SNR range: {dataset_config.snr_db_min} to {dataset_config.snr_db_max} dB")""",
            "detection": """# Detection preset (signal detection task)
dataset_config = TorchSigConfig.detection_preset()
print(f"Dataset: {dataset_config.num_samples} samples")
print(f"Includes noise-only samples for detection task")"""
        }
        cells.append(self._create_code_cell(dataset_configs.get(dataset_config, dataset_configs["minimal"])))

        # Model components
        cells.append(self._create_markdown_cell("## 4. Model Components"))

        # Losses
        cells.append(self._create_markdown_cell("### 4.1 Loss Functions"))
        losses_code = self.extract_module_code(
            self.src_dir / "training" / "losses.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(losses_code))

        # Attention
        cells.append(self._create_markdown_cell("### 4.2 Attention & RoPE"))
        attention_code = self.extract_module_code(
            self.src_dir / "models" / "blocks" / "attention.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(attention_code))

        # FFN
        cells.append(self._create_markdown_cell("### 4.3 Feed-Forward Networks"))
        ffn_code = self.extract_module_code(
            self.src_dir / "models" / "blocks" / "ffn.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        cells.append(self._create_code_cell(ffn_code))

        # Transformer blocks
        cells.append(self._create_markdown_cell("### 4.4 Transformer Blocks"))
        transformer_code = self.extract_module_code(
            self.src_dir / "models" / "blocks" / "transformer.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        transformer_code = transformer_code.replace(
            "from src.models.blocks.attention import Attention, RotaryPositionEmbedding, apply_rotary_pos_emb",
            "# Attention already defined above"
        ).replace(
            "from src.models.blocks.ffn import SwiGLU, StandardFFN",
            "# FFN already defined above"
        )
        cells.append(self._create_code_cell(transformer_code))

        # AudioMAE base (needed for inheritance)
        cells.append(self._create_markdown_cell("### 4.5 Base MAE Architecture"))
        audiomae_code = self.extract_module_code(
            self.src_dir / "models" / "audiomae.py",
            exclude_imports=True,
            exclude_docstring=True
        )
        audiomae_code = audiomae_code.replace(
            "from src.registry import model_registry",
            "# Registry not needed in notebook"
        ).replace(
            "from src.config import Config",
            "# Config already defined above"
        ).replace(
            "from src.models.base import BaseAutoencoder",
            "# Using nn.Module directly"
        ).replace(
            "from src.models.blocks.attention import Attention, RotaryPositionEmbedding",
            "# Attention already defined above"
        ).replace(
            "from src.models.blocks.ffn import SwiGLU, StandardFFN",
            "# FFN already defined above"
        ).replace(
            "from src.models.blocks.transformer import TransformerBlock, MacaronBlock",
            "# Transformer blocks already defined above"
        ).replace(
            "from src.training.losses import info_nce_loss, uniformity_loss",
            "# Losses already defined above"
        ).replace(
            "@model_registry.register(\"audiomae++\", version=\"2.0\")",
            "# AudioMAE++ Base Model"
        ).replace(
            "(BaseAutoencoder)",
            "(nn.Module)"
        )
        cells.append(self._create_code_cell(audiomae_code))

        # SignalMAE model
        cells.append(self._create_markdown_cell(f"### 4.6 {model_display} Model"))
        cells.append(self._create_code_cell(f"""# {model_display} - {'Advanced' if is_advanced else 'Baseline'} model for RF signals
class {model_display.replace('+', 'Plus')}(AudioMAEPlusPlus):
    \"\"\"
    {'SignalMAE++ with Macaron blocks, SwiGLU, and RoPE.' if is_advanced else 'SignalMAE baseline using standard ViT-MAE architecture.'}

    Processes RF spectrograms for masked autoencoder pre-training.
    \"\"\"

    def __init__(self, config: Config):
        {'# Ensure advanced features are enabled' if is_advanced else '# Disable advanced features for baseline'}
        modified_config = Config(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            audio_duration=config.audio_duration,
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_heads=config.encoder_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_heads=config.decoder_heads,
            mlp_ratio=config.mlp_ratio,
            mask_ratio=config.mask_ratio,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            pooling_mode=config.pooling_mode,
            use_macaron={str(is_advanced)},
            use_swiglu={str(is_advanced)},
            use_rope={str(is_advanced)},
            use_contrastive_loss=config.use_contrastive_loss,
            use_uniformity_loss=config.use_uniformity_loss,
        )
        super().__init__(modified_config)
        self._signal_type = "rf_spectrogram"

    @property
    def signal_type(self) -> str:
        return self._signal_type

    def get_model_info(self) -> Dict[str, Any]:
        return {{
            "model_name": "{model_display}",
            "signal_type": self.signal_type,
            "embed_dim": self.embed_dim,
            "num_patches": self.num_patches,
            "uses_macaron": {str(is_advanced)},
            "uses_swiglu": {str(is_advanced)},
            "uses_rope": {str(is_advanced)},
        }}

print("{model_display} class defined")
"""))

        # Data generation
        cells.append(self._create_markdown_cell("## 5. RF Data Generation"))
        cells.append(self._create_code_cell("""# IQ to Spectrogram conversion
def iq_to_spectrogram(iq_data: np.ndarray, img_size: int = 224) -> np.ndarray:
    \"\"\"
    Convert IQ samples to spectrogram image.

    Args:
        iq_data: Complex IQ samples [num_samples]
        img_size: Output image size

    Returns:
        Spectrogram as numpy array [3, img_size, img_size]
    \"\"\"
    from scipy import signal as scipy_signal

    # Compute spectrogram
    f, t, Sxx = scipy_signal.spectrogram(
        iq_data,
        fs=1.0,  # Normalized frequency
        nperseg=min(256, len(iq_data) // 4),
        noverlap=min(128, len(iq_data) // 8),
        return_onesided=False
    )

    # Convert to dB
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)

    # Normalize to [0, 1]
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-8)

    # Resize to target size
    import PIL.Image
    img = PIL.Image.fromarray((Sxx_norm * 255).astype(np.uint8))
    img = img.resize((img_size, img_size), PIL.Image.Resampling.BILINEAR)
    spec = np.array(img) / 255.0

    # Stack to 3 channels
    spec_3ch = np.stack([spec, spec, spec], axis=0)

    return spec_3ch.astype(np.float32)


def generate_synthetic_iq(
    modulation: str,
    num_samples: int = 1024,
    snr_db: float = 10.0
) -> np.ndarray:
    \"\"\"
    Generate synthetic IQ samples for a given modulation type.

    This is a fallback when TorchSig is not available.
    \"\"\"
    t = np.arange(num_samples) / num_samples

    # Generate base signal based on modulation type
    if modulation in ['bpsk', 'BPSK']:
        # Binary symbols
        symbols = np.random.choice([-1, 1], size=num_samples // 16)
        signal = np.repeat(symbols, 16)[:num_samples]
        carrier = np.exp(2j * np.pi * 0.1 * np.arange(num_samples))
        iq = signal * carrier

    elif modulation in ['qpsk', 'QPSK']:
        # Quadrature symbols
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=num_samples // 16)
        signal = np.repeat(symbols, 16)[:num_samples]
        carrier = np.exp(2j * np.pi * 0.1 * np.arange(num_samples))
        iq = signal * carrier

    elif modulation in ['qam16', '16QAM']:
        # 16-QAM constellation
        levels = [-3, -1, 1, 3]
        symbols = np.array([complex(np.random.choice(levels), np.random.choice(levels))
                          for _ in range(num_samples // 16)])
        signal = np.repeat(symbols, 16)[:num_samples]
        carrier = np.exp(2j * np.pi * 0.1 * np.arange(num_samples))
        iq = signal * carrier / 3  # Normalize

    elif modulation in ['ofdm', 'OFDM']:
        # Simple OFDM-like signal
        num_carriers = 64
        symbols = (np.random.randn(num_carriers) + 1j * np.random.randn(num_carriers)) / np.sqrt(2)
        iq = np.fft.ifft(symbols, n=num_samples)

    elif modulation in ['fm', 'FM', 'wbfm', 'WBFM']:
        # FM signal
        message = np.cumsum(np.random.randn(num_samples)) * 0.1
        iq = np.exp(2j * np.pi * message)

    else:
        # Default: random QPSK-like
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=num_samples // 16)
        signal = np.repeat(symbols, 16)[:num_samples]
        carrier = np.exp(2j * np.pi * 0.1 * np.arange(num_samples))
        iq = signal * carrier

    # Normalize signal power
    iq = iq / np.sqrt(np.mean(np.abs(iq)**2))

    # Add noise based on SNR
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    return iq + noise


print("IQ processing functions defined")
"""))

        # Dataset class
        cells.append(self._create_markdown_cell("### 5.1 RF Spectrogram Dataset"))
        cells.append(self._create_code_cell("""class RFSpectrogramDataset(torch.utils.data.Dataset):
    \"\"\"Dataset for RF signal spectrograms.\"\"\"

    def __init__(
        self,
        config: 'TorchSigConfig',
        model_config: Config,
        use_torchsig: bool = True
    ):
        self.config = config
        self.model_config = model_config
        self.use_torchsig = use_torchsig and TORCHSIG_AVAILABLE

        # Build modulation to label mapping
        self.modulations = config.modulations
        self.mod_to_label = {mod: i for i, mod in enumerate(self.modulations)}

        # Pre-generate dataset
        print(f"Generating {config.num_samples} RF samples...")
        self.samples = []
        self.labels = []

        samples_per_mod = config.num_samples // len(self.modulations)

        for mod in tqdm(self.modulations, desc="Generating signals"):
            for _ in range(samples_per_mod):
                # Random SNR within range
                snr = np.random.uniform(config.snr_db_min, config.snr_db_max)

                # Generate IQ data
                if self.use_torchsig:
                    # Use TorchSig for realistic signal generation
                    # (implementation depends on TorchSig API)
                    iq = generate_synthetic_iq(mod, config.num_iq_samples, snr)
                else:
                    # Fallback to synthetic generation
                    iq = generate_synthetic_iq(mod, config.num_iq_samples, snr)

                # Convert to spectrogram
                spec = iq_to_spectrogram(iq, model_config.img_size)

                self.samples.append(spec)
                self.labels.append(self.mod_to_label[mod])

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

        print(f"Generated {len(self.samples)} spectrograms")
        print(f"Classes: {len(self.modulations)} modulation types")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec = torch.from_numpy(self.samples[idx]).float()
        label = self.labels[idx]
        return spec, label

    @property
    def num_classes(self):
        return len(self.modulations)


print("RFSpectrogramDataset class defined")
"""))

        # Training functions
        cells.append(self._create_markdown_cell("## 6. Training Functions"))
        cells.append(self._create_code_cell("""def train_epoch(model, dataloader, optimizer, device, epoch):
    \"\"\"Train for one epoch.\"\"\"
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass (masked autoencoder)
        loss, pred, mask = model(data)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    \"\"\"Validate the model.\"\"\"
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            loss, pred, mask = model(data)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    \"\"\"Save model checkpoint.\"\"\"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else None,
    }, path)
    print(f"Saved checkpoint to {path}")


print("Training functions defined")
"""))

        # Create dataset and model
        cells.append(self._create_markdown_cell("## 7. Create Dataset and Model"))

        # Add config display cell
        cells.append(self._create_code_cell("""# Display training configuration
print("\\nTraining Configuration:")
print(f"  Epochs: {config.epochs}")
print(f"  Batch Size: {config.batch_size}")
print(f"  Learning Rate: {config.learning_rate}")
print(f"  Warmup Epochs: {config.warmup_epochs}")
print(f"  Checkpoint Interval: {config.checkpoint_interval}")
print(f"  Mask Ratio: {config.mask_ratio}")
print(f"  Weight Decay: {config.weight_decay}")
print()
"""))

        cells.append(self._create_code_cell(f"""# Setup directories
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Create dataset
print("Creating RF dataset...")
rf_dataset = RFSpectrogramDataset(
    config=dataset_config,
    model_config=config,
    use_torchsig=TORCHSIG_AVAILABLE
)

# Train/val split
train_size = int(0.8 * len(rf_dataset))
val_size = len(rf_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    rf_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {{len(train_dataset)}}, Val: {{len(val_dataset)}}")

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=device == "cuda"
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0
)

print(f"Train batches: {{len(train_loader)}}, Val batches: {{len(val_loader)}}")
"""))

        cells.append(self._create_code_cell(f"""# Create {model_display} model
print("Creating {model_display} model...")
model = {model_display.replace('+', 'Plus')}(config)
model = model.to(device)

# Print model info
info = model.get_model_info()
print(f"\\nModel: {{info['model_name']}}")
print(f"Signal type: {{info['signal_type']}}")
print(f"Embed dim: {{info['embed_dim']}}")
print(f"Num patches: {{info['num_patches']}}")
print(f"Macaron blocks: {{info['uses_macaron']}}")
print(f"SwiGLU: {{info['uses_swiglu']}}")
print(f"RoPE: {{info['uses_rope']}}")

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\nTotal parameters: {{total_params:,}}")
print(f"Trainable parameters: {{trainable_params:,}}")
"""))

        # Optimizer
        cells.append(self._create_markdown_cell("## 8. Training"))
        cells.append(self._create_code_cell("""# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95)
)

# Learning rate scheduler with warmup
def get_lr(epoch, config):
    \"\"\"Get learning rate for given epoch using warmup + cosine decay.\"\"\"
    if epoch < config.warmup_epochs:
        # Linear warmup
        return config.learning_rate * (epoch + 1) / config.warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: get_lr(epoch, config) / config.learning_rate
)

print("Optimizer and scheduler configured")
"""))

        # Training loop
        cells.append(self._create_code_cell(f"""# Training loop
history = {{'train_loss': [], 'val_loss': [], 'lr': []}}

print(f"\\nStarting {model_display} training for {{config.epochs}} epochs...")
print(f"Device: {{device}}")
print("-" * 50)

best_val_loss = float('inf')

for epoch in range(config.epochs):
    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)

    # Train
    train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
    history['train_loss'].append(train_loss)

    # Validate
    val_loss = validate(model, val_loader, device)
    history['val_loss'].append(val_loss)

    # Update LR
    scheduler.step()

    print(f"Epoch {{epoch + 1}}/{{config.epochs}} | "
          f"Train: {{train_loss:.4f}} | Val: {{val_loss:.4f}} | LR: {{current_lr:.2e}}")

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss,
                       CHECKPOINT_DIR / "{model_display.lower().replace('+', 'plus')}_best.pt")

    # Save periodic checkpoint
    if (epoch + 1) % config.checkpoint_interval == 0:
        save_checkpoint(model, optimizer, epoch, val_loss,
                       CHECKPOINT_DIR / f"{model_display.lower().replace('+', 'plus')}_epoch_{{epoch + 1}}.pt")

print("-" * 50)
print(f"Training complete! Best val loss: {{best_val_loss:.4f}}")
"""))

        # Visualization
        cells.append(self._create_markdown_cell("## 9. Results Visualization"))
        cells.append(self._create_code_cell("""# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history['lr'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Learning Rate Schedule')
axes[1].grid(True)

plt.tight_layout()
plt.show()
"""))

        # Reconstruction visualization
        cells.append(self._create_code_cell("""# Visualize reconstruction
model.eval()

sample_batch, sample_labels = next(iter(val_loader))
sample_batch = sample_batch.to(device)

with torch.no_grad():
    loss, pred, mask = model(sample_batch[:4])

    # Reshape prediction
    patch_size = config.patch_size
    h = w = config.img_size // patch_size
    pred_img = pred.reshape(-1, h, w, patch_size, patch_size, 3)
    pred_img = pred_img.permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, config.img_size, config.img_size)

# Plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    # Original
    orig = sample_batch[i].cpu().permute(1, 2, 0).numpy()
    orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
    axes[0, i].imshow(orig)
    axes[0, i].set_title(f'Original (mod {sample_labels[i].item()})')
    axes[0, i].axis('off')

    # Reconstruction
    recon = pred_img[i].cpu().permute(1, 2, 0).numpy()
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    axes[1, i].imshow(recon)
    axes[1, i].set_title(f'Reconstruction')
    axes[1, i].axis('off')

plt.suptitle(f'RF Spectrogram Reconstruction (Loss: {loss.item():.4f})')
plt.tight_layout()
plt.show()
"""))

        # Embedding extraction
        cells.append(self._create_markdown_cell("## 10. Embedding Extraction"))
        cells.append(self._create_code_cell("""# Extract embeddings for downstream tasks
model.eval()

all_embeddings = []
all_labels = []

with torch.no_grad():
    for data, labels in tqdm(val_loader, desc="Extracting embeddings"):
        data = data.to(device)
        embeddings = model.get_embedding(data, pooling_mode="mean")
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

all_embeddings = torch.cat(all_embeddings, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print(f"Extracted {len(all_embeddings)} embeddings")
print(f"Embedding shape: {all_embeddings.shape}")

# Visualize with t-SNE (optional)
try:
    from sklearn.manifold import TSNE

    print("\\nComputing t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings.numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=all_labels.numpy(), cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Modulation Type')
    plt.title('t-SNE Visualization of RF Signal Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, alpha=0.3)
    plt.show()
except ImportError:
    print("sklearn not available for t-SNE visualization")
"""))

        # Next steps
        cells.append(self._create_markdown_cell(f"""## Next Steps

After pre-training {model_display}, you can:

1. **Fine-tune for classification**: Add a classification head for modulation recognition
2. **Transfer learning**: Use embeddings for other RF tasks
3. **Increase data**: Use TorchSig's full dataset generation capabilities
4. **Compare models**: Try both SignalMAE (baseline) and SignalMAE++ (advanced)

### Example: Loading Different Model Configurations

```python
# Baseline SignalMAE (no advanced features)
config_baseline = create_rf_config(size="base", advanced=False)
model_baseline = SignalMAE(config_baseline)

# Advanced SignalMAE++ (Macaron, SwiGLU, RoPE)
config_advanced = create_rf_config(size="base", advanced=True)
model_advanced = SignalMAEPlusPlus(config_advanced)

# Small model for fast experiments
config_small = create_rf_config(size="small", advanced=False)
model_small = SignalMAESmall(config_small)
```

### Example: Different Dataset Configurations

```python
# Minimal (100 samples, 4 modulations) - for testing
config_minimal = TorchSigConfig.minimal_test_preset()

# Classification (10k samples, 12 modulations)
config_class = TorchSigConfig.classification_preset()

# Detection (5k samples, includes noise)
config_detect = TorchSigConfig.detection_preset()

# Custom configuration
config_custom = TorchSigConfig(
    num_samples=5000,
    modulations=['BPSK', 'QPSK', '16QAM', '64QAM', 'OFDM'],
    snr_db_min=-5,
    snr_db_max=20,
    num_iq_samples=2048,
)
```
"""))

        # Save notebook
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "cells": cells
        }

        output_name = f"training_{model_key.replace('+', 'plus')}_{dataset_config}.ipynb"
        output_path = self.output_dir / output_name

        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)

        print(f"Generated: {output_path}")
        return output_path


def main():
    """Generate notebooks from command line."""
    parser = argparse.ArgumentParser(
        description="Generate training notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate AudioMAE++ notebook for ESC-50 audio
  python notebooks/generate.py --model audiomae++ --dataset esc50

  # Generate SignalMAE baseline notebook for RF signals
  python notebooks/generate.py --model signalmae --dataset-config minimal

  # Generate SignalMAE++ (advanced) notebook with classification dataset
  python notebooks/generate.py --model signalmae++ --dataset-config classification

  # Generate SignalMAE with detection dataset configuration
  python notebooks/generate.py --model signalmae --dataset-config detection

  # List available modules
  python notebooks/generate.py --list-modules

Available Models:
  audiomae++    : Audio MAE with advanced features (for audio/ESC-50)
  signalmae     : RF Signal MAE baseline (standard ViT-MAE)
  signalmae++   : RF Signal MAE with Macaron, SwiGLU, RoPE

Dataset Configurations (for SignalMAE):
  minimal       : 100 samples, 4 modulations (quick testing)
  classification: 10k samples, 12 modulations (modulation recognition)
  detection     : 5k samples with noise (signal detection)
"""
    )
    parser.add_argument("--model", default="audiomae++",
                       help="Model: audiomae++, signalmae, signalmae++")
    parser.add_argument("--dataset", default="esc50",
                       help="Dataset for AudioMAE (e.g., esc50)")
    parser.add_argument("--dataset-config", default="minimal",
                       help="TorchSig config for SignalMAE: minimal, classification, detection")
    parser.add_argument("--output", help="Output path (optional)")
    parser.add_argument("--list-modules", action="store_true",
                       help="List available modules")
    parser.add_argument("--list-examples", action="store_true",
                       help="Show example commands")

    args = parser.parse_args()

    generator = NotebookGenerator()

    if args.list_modules:
        modules = generator.list_available_modules()
        print("Available modules for embedding:")
        for subdir, files in modules.items():
            print(f"\n  {subdir}/")
            for f in files:
                print(f"    - {f}")
        return

    if args.list_examples:
        print("""
Example Commands for Notebook Generation:
==========================================

# AudioMAE++ for audio classification (ESC-50 dataset)
python notebooks/generate.py --model audiomae++ --dataset esc50

# SignalMAE baseline for RF signals (minimal test data)
python notebooks/generate.py --model signalmae --dataset-config minimal

# SignalMAE++ advanced model (classification dataset)
python notebooks/generate.py --model signalmae++ --dataset-config classification

# SignalMAE for signal detection task
python notebooks/generate.py --model signalmae --dataset-config detection


Loading Models in Python:
=========================

from src.config import create_rf_config
from src.models.signalmae import SignalMAE, SignalMAEPlusPlus, SignalMAESmall

# Baseline SignalMAE (standard ViT-MAE, no advanced features)
config = create_rf_config(size="base", advanced=False)
model = SignalMAE(config)

# SignalMAE++ with all advanced features
config = create_rf_config(size="base", advanced=True)
model = SignalMAEPlusPlus(config)

# Small model for quick experiments
config = create_rf_config(size="small")
model = SignalMAESmall(config)


Dataset Configurations:
=======================

from src.data.torchsig_config import TorchSigConfig

# Minimal (100 samples, 4 mods) - testing
config = TorchSigConfig.minimal_test_preset()

# Classification (10k samples, 12 mods)
config = TorchSigConfig.classification_preset()

# Detection (5k samples, includes noise)
config = TorchSigConfig.detection_preset()

# Custom
config = TorchSigConfig(
    num_samples=5000,
    modulations=['BPSK', 'QPSK', '16QAM', 'OFDM'],
    snr_db_min=-5,
    snr_db_max=20,
)
""")
        return

    # Determine which generator to use based on model type
    model_lower = args.model.lower()

    if 'signal' in model_lower:
        # Use SignalMAE notebook generator
        generator.generate_signalmae_notebook(
            model_key=args.model,
            dataset_config=args.dataset_config
        )
    else:
        # Use AudioMAE notebook generator
        generator.generate_training_notebook(
            model_key=args.model,
            dataset_key=args.dataset
        )


if __name__ == "__main__":
    main()
