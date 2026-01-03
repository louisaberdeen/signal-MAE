# Implementation Plan: AudioMAE++ Training Improvements

## Goal
Add configurable training improvements for better semantic embeddings:
1. Lower mask ratio (85% → 75%)
2. Mean pooling option for embedding extraction
3. Contrastive loss (InfoNCE)
4. Uniformity loss
5. Dimension change validation

## Files to Modify

### 1. `audiomae.py` - Core Model Changes

**Config Class Updates:**
```python
# Add to Config class
mask_ratio = 0.75              # Changed from 0.8/0.85
use_contrastive_loss = True    # NEW
contrastive_weight = 0.01      # NEW
use_uniformity_loss = True     # NEW
uniformity_weight = 0.1        # NEW
pooling_mode = "mean"          # NEW: "cls", "mean", or "cls+mean"
```

**New Functions to Add:**
- `info_nce_loss(embeddings, labels, temperature)` - Contrastive loss
- `uniformity_loss(embeddings, t)` - Uniformity regularization
- `extract_embeddings(latent, mode)` - Configurable pooling

**Model Changes:**
- Modify `AudioMAEPlusPlus.forward()` to optionally return embeddings
- Add new forward method variant for training with losses

### 2. `embeddings_utils.py` - Embedding Extraction

**Update `EmbeddingGenerator.extract_embedding()`:**
- Add `pooling_mode` parameter
- Support "cls", "mean", "cls+mean" modes

### 3. `audiomaepp.ipynb` - Training Notebook

**Update training loop:**
- Pass labels to model when using contrastive loss
- Log additional losses (contrastive, uniformity)
- Add config options in setup cell

## Implementation Steps

### Step 1: Update Config Class in audiomae.py
- Add new config parameters with sensible defaults
- Ensure backward compatibility (existing code still works)

### Step 2: Add Loss Functions in audiomae.py
- `info_nce_loss()` - contrastive loss using class labels
- `uniformity_loss()` - embedding space uniformity

### Step 3: Add Pooling Helper in audiomae.py
- `get_embedding()` method that supports different pooling modes

### Step 4: Modify AudioMAEPlusPlus Forward
- Add optional `labels` parameter
- Return additional losses when training with contrastive/uniformity

### Step 5: Update embeddings_utils.py
- Modify `extract_embedding()` to use configurable pooling

### Step 6: Test Dimension Changes
- Create test script to verify model works with different img_size/patch_size
- Test: 384×384 with 16×16 patches, 448×448 with 32×32 patches

### Step 7: Verify Training Runs
- Quick smoke test that training loop works with new losses

## Testing Plan

1. **Unit test losses**: Verify loss functions compute correctly
2. **Dimension test**: Verify model initializes with different sizes
3. **Forward pass test**: Verify model runs with new parameters
4. **Training smoke test**: Run 1-2 epochs to verify no errors

## Rollback Safety

All new features are **opt-in via config**:
- `use_contrastive_loss = False` disables contrastive loss
- `use_uniformity_loss = False` disables uniformity loss
- `pooling_mode = "cls"` keeps original behavior

Existing code paths unchanged when new features disabled.
