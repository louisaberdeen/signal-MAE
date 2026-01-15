# Specification: FiftyOne Audio Embedding Visualization System

**Version**: 2.0
**Date**: 2025-01-03
**Status**: Implemented

## Executive Summary

This specification defines a system for generating embeddings from trained AudioMAE models and visualizing audio datasets in FiftyOne with similarity search, geographic mapping, and dimensionality reduction. The system is designed as a proof of concept with ESC-50 but architected for extensibility to any IQ data (audio, RF signals).

## Goals

### Primary Goals
1. Generate 768-dimensional embeddings from AudioMAE encoder for audio datasets
2. Load embeddings and metadata into FiftyOne for interactive exploration
3. Enable similarity search to find acoustically similar audio clips
4. Visualize embedding space using UMAP and t-SNE
5. Support geographic visualization with lat/long coordinates
6. Design for extensibility to custom datasets and RF signals

### Non-Goals
- Real-time audio processing or streaming
- Training new models (use existing checkpoints)
- Audio augmentation or preprocessing beyond existing pipeline
- Production deployment or API endpoints

## User Requirements

### Functional Requirements

**FR-1: Embedding Generation**
- Load trained AudioMAE checkpoint from user-specified path
- Process audio files from a directory
- Extract 768-dimensional CLS token embeddings from encoder
- Support batch processing for efficiency (target: 2000 files in <3 minutes on GPU)

**FR-2: Data Loading**
- Support ESC-50 dataset format (2000 5-second audio clips)
- Load metadata from CSV (filename, class, fold, category)
- Support extensible data loader architecture for custom datasets
- Handle missing or corrupted audio files gracefully

**FR-3: Metadata Integration**
- Join audio files with tabular metadata (pandas/CSV format)
- Support custom metadata columns
- Add geographic coordinates (lat/long) from external source or synthetic
- Validate and sanitize metadata before FiftyOne integration

**FR-4: FiftyOne Dataset Creation**
- Create persistent FiftyOne dataset with unique name
- Add PNG spectrogram images as primary media for visual exploration
- Store original audio file paths as secondary reference field
- Attach embeddings as vector fields
- Include all metadata as sample fields
- Add classification labels with proper schema
- Add GeoLocation fields for map visualization

**FR-5: Similarity Search**
- Compute similarity index from embeddings
- Support query-by-sample (find similar to given audio)
- Support query-by-vector (find similar to arbitrary embedding)
- Return top-k most similar results (default k=20)
- Use sklearn backend for proof of concept

**FR-6: Embedding Space Visualization**
- Compute UMAP dimensionality reduction (2D)
- Compute t-SNE dimensionality reduction (2D) as alternative
- Create interactive plots in FiftyOne App
- Support lasso selection for subset exploration
- Configure visualization parameters (n_neighbors, perplexity, etc.)

**FR-7: Geographic Visualization**
- Display audio samples on interactive map
- Support lat/long coordinate fields
- For PoC: Generate synthetic geographic data clustered by category
- Future: Integrate with Freesound API for real location data

**FR-8: Metadata Filtering**
- Filter dataset by classification labels
- Filter by cross-validation fold
- Filter by category (animals, nature, urban, etc.)
- Support complex queries combining multiple fields
- Filter by geographic region (bounding box)

**FR-9: Spectrogram Visualization**
- Display PNG spectrogram previews as primary media in FiftyOne grid view
- Enable visual pattern recognition across audio samples
- Support thumbnail browsing of spectrograms
- Maintain reference to original audio files for playback
- Auto-validate PNG files exist before dataset creation

### Non-Functional Requirements

**NFR-1: Performance**
- Embedding generation: ≤3 minutes for 2000 samples (GPU)
- Similarity index computation: ≤30 seconds
- UMAP computation: ≤90 seconds
- t-SNE computation: ≤180 seconds
- Total pipeline: ≤5 minutes end-to-end

**NFR-2: Usability**
- Single notebook interface (`fiftyone_visualization.ipynb`)
- Clear configuration section (checkpoint path, data dirs)
- Progress bars for all long-running operations
- Informative error messages with suggested fixes
- Automatic cache detection and reuse

**NFR-3: Extensibility**
- Abstract data loader interface for different datasets
- Configurable column mapping for custom metadata
- Support for different spectrogram formats (mel, STFT, CQT)
- Model adapter pattern for non-AudioMAE models
- Plugin system for custom visualizations

**NFR-4: Reliability**
- Graceful handling of corrupted audio files
- Checkpoint validation before inference
- Embedding cache validation (detect stale cache)
- Atomic cache writes (no partial corruption)
- Comprehensive error logging

## System Architecture

### High-Level Components (v2.0)

The system uses a modular plugin-based architecture with registries:

```
┌─────────────────────────────────────────────────────────────┐
│                    Plugin Registry System                    │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ │
│  │ model_registry │  │ data_loader_   │  │ transform_    │ │
│  │                │  │ registry       │  │ registry      │ │
│  │ • audiomae++   │  │ • esc50        │  │ • audio_spec  │ │
│  │ • baseline     │  │ • custom       │  │ • iq_spec     │ │
│  │ • classifier   │  │ • rf           │  │ • iq_const    │ │
│  └────────────────┘  └────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Generated Self-Contained Notebooks                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ notebooks/generated/training_audiomaepp_esc50.ipynb    ││
│  │ • All code embedded inline (no external imports)       ││
│  │ • Ready for Google Colab upload                        ││
│  │ • Complete training + evaluation pipeline              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  FiftyOne Visualization Pipeline                             │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Configuration │─▶│ Data Loading │─▶│ Embedding        │ │
│  │               │  │              │  │ Generation       │ │
│  └───────────────┘  └──────────────┘  └──────────────────┘ │
│                            │                     │          │
│                            ▼                     ▼          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              FiftyOne Integration                     │ │
│  │  • Dataset Creation  • Similarity Index              │ │
│  │  • Metadata Fields   • UMAP/t-SNE                    │ │
│  │  • Embeddings        • Geographic Map                │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

Modular src/ Structure:
┌──────────────────────────────────────────────────────────────┐
│ src/                                                          │
│ ├── registry.py       # PluginRegistry with decorators       │
│ ├── config.py         # Config dataclass                     │
│ ├── models/           # Model plugins (BaseAutoencoder ABC)  │
│ │   ├── audiomae.py   # @model_registry.register("audiomae++")│
│ │   ├── baseline.py   # @model_registry.register("baseline") │
│ │   └── blocks/       # Transformer components               │
│ ├── data/             # Data loader plugins                  │
│ │   ├── esc50.py      # @data_loader_registry.register       │
│ │   └── custom.py     # Generic + RF loaders                 │
│ ├── transforms/       # Transform plugins                    │
│ │   ├── audio.py      # Audio → Spectrogram                  │
│ │   └── rf.py         # IQ → Spectrogram/Constellation       │
│ ├── training/         # Loss functions                       │
│ └── embeddings/       # EmbeddingGenerator, Cache, Checkpoint│
└──────────────────────────────────────────────────────────────┘

Test Generation:
┌──────────────────────────────────────────────────────────────┐
│  tests/generate_tests.py → tests/generated/                  │
│  • Interface compliance tests (ABC method implementation)    │
│  • Architecture compatibility tests (various input sizes)    │
│  • Plugin registration correctness tests                     │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Audio directory + Metadata CSV + Model checkpoint
2. **Preprocessing**: Load audio → Generate/load spectrograms (both .npy and .png)
3. **Inference**: Batch process through AudioMAE encoder → Extract CLS embeddings
4. **Caching**: Save embeddings as .npy + metadata JSON
5. **FiftyOne Integration**:
   - Create dataset with PNG spectrograms as primary media
   - Add audio filepaths as secondary reference
   - Attach embeddings and metadata
6. **Indexing**: Compute similarity index + UMAP/t-SNE
7. **Output**: Interactive FiftyOne App session with visual spectrogram grid

## Technical Specification

### Embedding Extraction

**Model**: AudioMAE++ (from `audiomae.py`)
- Architecture: Vision Transformer with Macaron blocks, SwiGLU, RoPE
- Encoder output: `[batch_size, 197, 768]` (196 patches + 1 CLS token)
- Embedding vector: CLS token at position 0 → `[batch_size, 768]`

**Inference Pattern**:
```python
model.eval()
with torch.no_grad():
    latent, _, _ = model.forward_encoder(spectrogram, mask_ratio=0.0)
    embedding = latent[:, 0, :].cpu().numpy()  # [batch_size, 768]
```

**Critical Parameters**:
- `mask_ratio=0.0`: No masking during inference (use all patches)
- Device: CUDA preferred, CPU fallback
- Batch size: 32 (configurable based on GPU memory)

### Data Schema

**Metadata CSV Format** (ESC-50):
```
filename,fold,target,category,esc10,src_file,take
1-100032-A-0.wav,1,0,dog,False,100032,A
```

**Standard Schema** (after transformation):
```python
{
    "filepath": str,           # Absolute path to PNG spectrogram (primary media)
    "audio_filepath": str,     # Absolute path to original audio file
    "label": str,              # Human-readable category
    "target": int,             # Numeric class ID (0-49 for ESC-50)
    "fold": int,               # Cross-validation fold (1-5)
    "category": str,           # Same as label
    "lat": float,              # Latitude (-90 to 90)
    "lon": float,              # Longitude (-180 to 180)
    "esc10": bool,             # Subset flag (ESC-50 specific)
    "src_file": str,           # Source file ID
    "take": str,               # Take identifier
}
```

**FiftyOne Sample Schema**:
```python
{
    "filepath": str,                           # Required: path to PNG spectrogram
    "audio_filepath": str,                     # Original audio file path
    "embedding": List[float],                  # 768-dim vector (from spectrogram)
    "label": fo.Classification(label=str),     # Classification object
    "location": fo.GeoLocation(point=[lon, lat]), # [lon, lat] order!
    "fold": int,
    "target": int,
    "category": str,
    "esc10": bool,
    "filename": str,                           # Audio filename
    "src_file": str,
    "take": str,
}
```

### Caching Strategy

**Cache Structure**:
```
data/embeddings/{dataset_name}/
├── embeddings.npy      # [N, 768] numpy array
├── metadata.json       # {filenames: [...], timestamp: ..., model_info: ...}
└── config.json         # {checkpoint_hash: ..., model_config: ...}
```

**Cache Validation**:
- Check file count matches current dataset
- Verify checkpoint hash matches current model
- Invalidate if metadata changed
- Allow partial updates (only process new files)

**Cache Operations**:
- Save: Atomic write via temp file + move
- Load: Validate before returning
- Invalidate: Delete cache directory
- Update: Append new embeddings + update metadata

### FiftyOne Integration

**Similarity Search**:
```python
fob.compute_similarity(
    dataset,
    embeddings="embedding",        # Field name
    brain_key="audiomae_sim",      # Unique identifier
    backend="sklearn",             # sklearn | mongodb
)
```

**UMAP Visualization**:
```python
fob.compute_visualization(
    dataset,
    embeddings="embedding",
    method="umap",
    num_dims=2,
    brain_key="umap",
    n_neighbors=15,    # Local vs global structure
    min_dist=0.1,      # Point separation
)
```

**t-SNE Visualization**:
```python
fob.compute_visualization(
    dataset,
    embeddings="embedding",
    method="tsne",
    num_dims=2,
    brain_key="tsne",
    perplexity=30,     # Effective number of neighbors
    learning_rate=200, # Step size
)
```

**Geographic Field**:
```python
sample["location"] = fo.GeoLocation(point=[longitude, latitude])
# WARNING: Order is [lon, lat] not [lat, lon]!
```

## Interface Specification

### Configuration Object

```python
@dataclass
class Config:
    # Model
    checkpoint_path: Path                    # User-specified checkpoint
    device: str = "cuda"                     # cuda | cpu

    # Data
    data_root: Path                          # Root directory for dataset
    spectrogram_dir: Path                    # Precomputed spectrograms
    metadata_csv: Optional[Path] = None      # Auto-detect if None

    # Processing
    batch_size: int = 32                     # Inference batch size
    num_workers: int = 0                     # DataLoader workers (0 for notebook)
    use_cache: bool = True                   # Use embedding cache
    cache_dir: Path                          # Cache storage location

    # FiftyOne
    dataset_name: str                        # Unique FO dataset name
    persistent: bool = True                  # Save to MongoDB
    embedding_field: str = "embedding"       # Field name for embeddings

    # Visualization
    compute_umap: bool = True                # Compute UMAP
    compute_tsne: bool = True                # Compute t-SNE
    add_synthetic_geo: bool = True           # Generate synthetic coords (PoC)
```

### API: EmbeddingGenerator Class

```python
class EmbeddingGenerator:
    """Generate embeddings from AudioMAE model."""

    def __init__(self, model, config, transform):
        """Initialize with model, config, and spectrogram transform."""

    def generate_embeddings(
        self,
        audio_paths: List[Path],
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate embeddings for audio files.

        Args:
            audio_paths: List of audio file paths
            use_cache: Load from cache if available

        Returns:
            embeddings: [N, 768] numpy array
            metadata: {"filenames": [...], "failed": [...], "timestamp": ...}
        """
```

### API: DataLoader Interface

```python
class AudioDataLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata with standard schema.

        Returns:
            DataFrame with columns: filepath, label, lat, lon, [custom...]
        """

    @abstractmethod
    def get_audio_paths(self) -> List[Path]:
        """Return list of audio file paths."""

    @abstractmethod
    def get_spectrogram_paths(self) -> Dict[str, Path]:
        """Return mapping: {audio_filename: spectrogram_path}"""
```

## Testing & Validation

### Test Cases

**TC-1: Checkpoint Loading**
- Valid checkpoint → loads successfully
- Invalid path → clear error message
- Corrupted checkpoint → validation error

**TC-2: Embedding Generation**
- Single audio file → [1, 768] embedding
- Batch of 16 files → [16, 768] embeddings
- Empty directory → empty output with warning
- Corrupted audio → skip with error log

**TC-3: Cache Operations**
- First run → generate and save cache
- Second run → load from cache (fast)
- Model change → invalidate and regenerate
- File added → partial update

**TC-4: FiftyOne Dataset**
- Dataset creation → 2000 samples
- Embeddings attached → all non-null
- Labels present → classification objects
- Location fields → valid coordinates

**TC-5: Similarity Search**
- Query by sample → returns 20 results
- Query by vector → returns 20 results
- Similar classes cluster together
- Different classes separate

**TC-6: Visualizations**
- UMAP completes in <90s
- t-SNE completes in <180s
- Points clustered by category
- Lasso selection works

### Validation Metrics

**Embedding Quality**:
- Intra-class distance < Inter-class distance
- Similar sounds rank high in similarity search
- UMAP shows category clusters

**System Performance**:
- Pipeline completes in <5 minutes
- Cache reduces runtime by >80%
- No memory leaks during processing

**User Experience**:
- Clear progress indicators
- Informative error messages
- Successful App launch on first try

## Deployment

### Prerequisites

**Python Environment**:
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- Virtual environment at `.venv/`

**Dependencies**:
- PyTorch 2.0+
- FiftyOne 1.11.0+
- librosa 0.10+
- numpy 1.26+
- pandas 2.0+
- umap-learn 0.5+
- scikit-learn 1.3+

**Installation**:
```bash
pip install fiftyone umap-learn scikit-learn
```

### File Locations

```
/home/louis/Documents/fifty-one/
├── src/                              # Modular framework (v2.0)
│   ├── __init__.py                   # Exports registries
│   ├── registry.py                   # PluginRegistry class
│   ├── config.py                     # Config dataclass
│   ├── models/                       # Model plugins
│   │   ├── base.py                   # BaseModel, BaseAutoencoder ABCs
│   │   ├── audiomae.py               # AudioMAE++ implementation
│   │   ├── baseline.py               # Baseline MAE
│   │   ├── classifier.py             # Classification wrapper
│   │   └── blocks/                   # Transformer components
│   ├── data/                         # Data loader plugins
│   │   ├── base.py                   # BaseDataLoader ABC
│   │   ├── esc50.py                  # ESC-50 loader
│   │   └── custom.py                 # Generic + RF loaders
│   ├── transforms/                   # Transform plugins
│   │   ├── base.py                   # BaseTransform ABC
│   │   ├── audio.py                  # Audio spectrograms
│   │   └── rf.py                     # RF spectrograms
│   ├── training/                     # Training utilities
│   │   └── losses.py                 # Loss functions
│   └── embeddings/                   # Embedding utilities
│       ├── generator.py              # EmbeddingGenerator
│       ├── cache.py                  # EmbeddingCache
│       └── checkpoint.py             # CheckpointLoader
├── tests/                            # Test suite
│   ├── conftest.py                   # Pytest fixtures
│   ├── generate_tests.py             # Test generator
│   └── generated/                    # Auto-generated tests (gitignored)
├── notebooks/                        # Notebook generation
│   ├── generate.py                   # NotebookGenerator
│   └── generated/                    # Self-contained notebooks (gitignored)
├── fiftyone_visualization.ipynb      # Main FiftyOne visualization notebook
├── audiomaepp.ipynb                  # Complete pipeline (data prep + pretrain + fine-tune)
├── audiomae.py                       # Legacy wrapper (backward compat)
├── embeddings_utils.py               # Legacy wrapper (backward compat)
├── data_loaders.py                   # Legacy wrapper (backward compat)
├── spec.md                           # This specification
├── CLAUDE.md                         # Claude Code instructions
├── .env                              # Environment variables (gitignored)
├── .gitignore                        # Git ignore rules
├── checkpoints/
│   ├── encoder_only_fixed.pt         # Trained encoder checkpoint
│   ├── encoder_only.pt               # Alternative checkpoint
│   └── audiomae_pretrain_finetune/   # Full model checkpoint
├── data/
│   ├── ESC-50-master/                # ESC-50 dataset
│   │   ├── audio/                    # 2000 .wav files
│   │   └── meta/esc50.csv            # Metadata
│   ├── imgs/
│   │   ├── full/                     # Precomputed spectrograms (.npy)
│   │   └── pre/                      # Preview PNGs for FiftyOne
│   └── embeddings/                   # Cached embeddings
│       └── esc50_audiomae/           # ESC-50 embeddings
└── .venv/                            # Python virtual environment
```

### Usage

**Quick Start**:
1. Open `fiftyone_visualization.ipynb`
2. Set `checkpoint_path` in Cell 2
3. Run all cells (Kernel → Restart & Run All)
4. Wait ~3-5 minutes for processing
5. FiftyOne App launches automatically

**Customization**:
- Adjust `batch_size` for memory constraints
- Modify visualization parameters (n_neighbors, perplexity)
- Add custom metadata columns
- Change dataset name for versioning

## Future Enhancements

### Phase 2 (Post-PoC)

1. **Real Geographic Data**
   - Integrate Freesound API for location metadata
   - Support user-provided lat/long CSV
   - Geocoding from text descriptions

2. **Advanced Visualizations**
   - 3D UMAP/t-SNE plots
   - Audio waveform overlays in detail view
   - ✅ Spectrogram thumbnails in App (IMPLEMENTED: PNG previews as primary media)
   - Interactive confusion matrices
   - Dual-view mode (spectrogram + audio playback)

3. **Performance Optimizations**
   - Multi-GPU embedding generation
   - Distributed processing for large datasets
   - MongoDB backend for similarity (persistent index)
   - Incremental cache updates

4. **RF Signal Support**
   - IQ data loaders (I/Q samples)
   - RF-specific spectrograms (waterfall, constellation)
   - Modulation classification
   - Spectrum occupancy visualization

5. **Export & Reporting**
   - CSV export with embeddings
   - PDF reports with visualizations
   - Embedding clustering analysis
   - Anomaly detection reports

### Phase 3 (Production)

1. **API Endpoints**
   - REST API for embedding generation
   - Similarity search service
   - Batch processing API

2. **Web Interface**
   - Browser-based visualization
   - Drag-and-drop audio upload
   - Collaborative annotations

3. **Integration**
   - Weights & Biases logging
   - MLflow experiment tracking
   - CI/CD pipeline for testing

## Appendix

### Glossary

- **AudioMAE**: Masked Autoencoder for Audio - self-supervised learning model
- **CLS Token**: Classification token - aggregated representation from transformer
- **Embedding**: Fixed-size vector representation of audio (768-dim)
- **FiftyOne**: Open-source tool for dataset visualization and exploration
- **IQ Data**: In-phase and Quadrature signal components (RF)
- **UMAP**: Uniform Manifold Approximation and Projection - dimensionality reduction
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding - dimensionality reduction

### References

- [FiftyOne Documentation](https://docs.voxel51.com/)
- [FiftyOne Brain (Embeddings)](https://docs.voxel51.com/brain.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [AudioMAE Paper](https://arxiv.org/abs/2203.16691) (reference architecture)

### Contact & Support

For questions or issues:
1. Check plan file: `/home/louis/.claude/plans/jaunty-forging-forest.md`
2. Review CLAUDE.md in project root
3. Consult FiftyOne documentation for App usage
