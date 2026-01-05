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
# CONFIGURATION
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

# Training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1.5e-4
SAVE_EVERY = 5  # Save checkpoint every N epochs
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True if device == "cuda" else False
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
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
    lr=LEARNING_RATE,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95)
)

# Learning rate scheduler with warmup
def get_lr(epoch, warmup_epochs=5, total_epochs=NUM_EPOCHS):
    if epoch < warmup_epochs:
        return LEARNING_RATE * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: get_lr(epoch) / LEARNING_RATE
)
"""))

        cells.append(self._create_markdown_cell("### 8.6 Training Loop"))
        cells.append(self._create_code_cell("""# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'lr': []
}

print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"Device: {device}")
print("-" * 50)

best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
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
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
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
    if (epoch + 1) % SAVE_EVERY == 0:
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


def main():
    """Generate notebooks from command line."""
    parser = argparse.ArgumentParser(description="Generate training notebooks")
    parser.add_argument("--model", default="audiomae++", help="Model plugin key")
    parser.add_argument("--dataset", default="esc50", help="Dataset plugin key")
    parser.add_argument("--output", help="Output path (optional)")
    parser.add_argument("--list-modules", action="store_true", help="List available modules")

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

    generator.generate_training_notebook(
        model_key=args.model,
        dataset_key=args.dataset
    )


if __name__ == "__main__":
    main()
