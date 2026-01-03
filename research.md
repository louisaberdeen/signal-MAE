# Research: Improving AudioMAE++ Embeddings for Similarity Search

**Date**: 2025-12-29
**Goal**: Improve semantic quality of AudioMAE++ embeddings for audio similarity search

---

## Executive Summary

Your current AudioMAE++ setup uses standard masked autoencoder training with 80-85% mask ratio. While MAE produces good representations for classification tasks, **pure reconstruction loss often leads to embeddings that cluster by acoustic texture rather than semantic meaning**. This is problematic for similarity search where you want "dog barking" to be near "wolf howling" (same category) rather than near "static noise" (similar texture).

### Quick Wins (Ranked by Impact/Effort)

| Priority | Change | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| 1 | Switch CLS → Mean pooling | 5 min | +5-15% retrieval |
| 2 | Lower mask ratio to 75% | Config | Potentially better |
| 3 | Add contrastive loss | Medium | +10-20% semantic |
| 4 | Higher resolution (384×384) | Retrain | Better detail |
| 5 | CLAP-style fine-tuning | High | Best semantic |

---

## 1. Pooling Strategy: CLS Token vs Mean Pooling

### The Problem

Your current embedding extraction uses the CLS token:
```python
# Current approach in embeddings_utils.py
latent, _, _ = model.forward_encoder(spectrograms, mask_ratio=0.0)
embedding = latent[:, 0, :]  # CLS token at position 0
```

**Research Finding**: For similarity search tasks, CLS token embeddings often underperform mean pooling, especially when the model wasn't specifically fine-tuned for the similarity task.

> "Mean pooling often produces higher-quality embeddings for tasks requiring comprehensive understanding of the sequence, such as semantic similarity or clustering." - [Milvus AI Reference](https://milvus.io/ai-quick-reference/how-does-the-choice-of-pooling-strategy-mean-pooling-vs-using-the-cls-token-potentially-affect-the-quality-of-the-embeddings-and-the-speed-of-computation)

> "The SBERT paper tested three different pooling methods: mean, max, and [CLS]-pooling. The mean-pooling approach was best performing for both NLI and STSb datasets." - [Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/models.html)

### Recommended Fix

```python
def extract_embedding(self, spectrograms: torch.Tensor) -> np.ndarray:
    """Extract embedding using mean pooling over all patch tokens."""
    self.model.eval()
    with torch.no_grad():
        latent, _, _ = self.model.forward_encoder(spectrograms, mask_ratio=0.0)

        # Option 1: Mean pooling (exclude CLS token)
        patch_embeddings = latent[:, 1:, :]  # [B, 196, 768]
        embedding = patch_embeddings.mean(dim=1)  # [B, 768]

        # Option 2: Include CLS in mean (sometimes better)
        # embedding = latent.mean(dim=1)  # [B, 768]

        # Option 3: Weighted mean with attention scores
        # (requires storing attention weights during forward pass)

        return embedding.cpu().numpy()
```

### Alternative: Attention Pooling

For potentially better results, implement learnable attention pooling:

```python
class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling."""
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B, N, D] where N = num_patches + 1 (CLS)
        weights = F.softmax(self.attention(x), dim=1)  # [B, N, 1]
        pooled = (x * weights).sum(dim=1)  # [B, D]
        return pooled
```

---

## 2. Contrastive Loss: Preventing Embedding Collapse

### The Core Problem

Pure reconstruction loss in MAE can lead to **dimensional collapse** - embeddings that don't spread well in the embedding space. This makes similarity search less discriminative.

> "The reconstruction task induces positive-pair alignment... adding loss uniformity can mitigate feature collapse." - [Understanding MAE from Local Contrastive Perspective](https://arxiv.org/html/2310.01994v2)

### CAV-MAE Approach

[CAV-MAE](https://github.com/YuanGongND/cav-mae) (Contrastive Audio-Visual Masked Autoencoder) combines reconstruction with contrastive learning:

```
L_total = L_reconstruction + λ × L_contrastive
```

**Recommended hyperparameters** from CAV-MAE:
- `mae_loss_weight = 1.0`
- `contrast_loss_weight = 0.01` (small but important!)
- `mask_ratio = 0.75` (not 0.80-0.85)

### Implementation: Add InfoNCE Contrastive Loss

```python
def info_nce_loss(embeddings, labels, temperature=0.07):
    """
    InfoNCE contrastive loss for same-class positive pairs.

    Args:
        embeddings: [B, D] normalized embeddings
        labels: [B] class labels
        temperature: softmax temperature (0.07 typical)
    """
    embeddings = F.normalize(embeddings, dim=1)

    # Similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Positive mask: same class = positive pair
    labels = labels.unsqueeze(0)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)  # Exclude self-similarity

    # InfoNCE loss
    exp_sim = torch.exp(sim_matrix)
    exp_sim.fill_diagonal_(0)

    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    neg_sim = exp_sim.sum(dim=1)

    loss = -torch.log(pos_sim / (neg_sim + 1e-8) + 1e-8)
    return loss[pos_mask.sum(dim=1) > 0].mean()


class AudioMAEWithContrastive(nn.Module):
    """AudioMAE++ with added contrastive loss."""

    def __init__(self, base_model, contrast_weight=0.01):
        super().__init__()
        self.mae = base_model
        self.contrast_weight = contrast_weight

    def forward(self, imgs, labels=None, mask_ratio=0.8):
        # Standard MAE forward
        mae_loss, pred, mask = self.mae(imgs, mask_ratio)

        if labels is not None and self.contrast_weight > 0:
            # Extract embeddings for contrastive loss
            with torch.no_grad():
                latent, _, _ = self.mae.forward_encoder(imgs, mask_ratio=0.0)
            embeddings = latent[:, 1:, :].mean(dim=1)  # Mean pool patches

            contrast_loss = info_nce_loss(embeddings, labels)
            total_loss = mae_loss + self.contrast_weight * contrast_loss
            return total_loss, pred, mask, contrast_loss

        return mae_loss, pred, mask, torch.tensor(0.0)
```

### Uniformity Loss (Alternative)

Add uniformity loss to spread embeddings on hypersphere:

```python
def uniformity_loss(embeddings, t=2):
    """
    Uniformity loss from 'Understanding Contrastive Representation Learning'.
    Encourages embeddings to be uniformly distributed on unit sphere.
    """
    embeddings = F.normalize(embeddings, dim=1)
    sq_pdist = torch.pdist(embeddings, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()
```

---

## 3. Resolution and Patch Size

### Current Configuration Analysis

```python
# Your current setup
img_size = 224          # Spectrogram image size
patch_size = 16         # 16×16 patches
num_patches = 196       # (224/16)² = 196 patches
n_mels = 128           # Mel frequency bins

# Original spectrogram before resize
# 5 seconds @ 22050Hz, hop_length=512 → ~216 time frames
# Original shape: [128, 216] → resized to [224, 224]
```

**Issue**: Resizing 128×216 to 224×224 loses frequency resolution detail and introduces aspect ratio distortion.

### Standard AST Approach

[Audio Spectrogram Transformer](https://github.com/YuanGongND/ast) uses:
- `num_mel_bins = 128`
- `patch_size = 16` with **overlapping patches** (stride 10)
- Variable-length spectrograms (not resized to square)
- Position embedding interpolation for different lengths

### Recommended Experiments

#### Option A: Higher Resolution with Same Patch Count
```python
# Preserve more detail, same compute
img_size = 384          # Larger image
patch_size = 16         # Same patch size
num_patches = 576       # (384/16)² = 576 patches
# Pros: 4× more spatial detail
# Cons: ~3× more compute (attention scales with N²)
```

#### Option B: Larger Patches at Higher Resolution
```python
# User's suggested approach
img_size = 448          # Or 512
patch_size = 32         # Larger patches
num_patches = 196       # (448/32)² ≈ 196 patches
# Pros: Same compute, more input detail per patch
# Cons: Coarser spatial reasoning within patches
```

#### Option C: Native Aspect Ratio (AST-style)
```python
# Don't resize to square - use native spectrogram shape
n_mels = 128
time_frames = 1024      # 10 seconds of audio
patch_size = 16
freq_patches = 128 // 16  # 8 patches in frequency
time_patches = 1024 // 16  # 64 patches in time
num_patches = 8 * 64 = 512
# Pros: No distortion, natural audio structure
# Cons: Requires architecture changes for non-square input
```

#### Option D: Overlapping Patches
```python
# AST uses stride 10 instead of 16
patch_size = 16
stride = 10  # 6-pixel overlap
# More patches but smoother representations
# Requires modifying PatchEmbed class
```

### Modifying PatchEmbed for Overlapping Patches

```python
class PatchEmbedOverlap(nn.Module):
    """Patch embedding with configurable overlap."""

    def __init__(self, img_size=224, patch_size=16, stride=10, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride

        # Calculate number of patches
        self.num_patches_h = (img_size - patch_size) // stride + 1
        self.num_patches_w = (img_size - patch_size) // stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride  # Key change: stride != patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x
```

---

## 4. CLAP-Style Semantic Training

### Why CLAP Works Better for Semantic Search

[CLAP](https://github.com/LAION-AI/CLAP) (Contrastive Language-Audio Pretraining) trains audio encoders to align with text descriptions. This creates embeddings where semantic similarity matters more than acoustic similarity.

> "CLAP embeddings cluster by semantic meaning... 'dog barking' and 'wolf howling' will be close (same semantic category)."

### Simplified CLAP-Lite for ESC-50

Since you have class labels, you can approximate CLAP by treating labels as pseudo-text:

```python
class CLAPLiteTrainer:
    """
    CLAP-style training using class labels as pseudo-text.

    Instead of:  audio <-> "a sound of a dog barking"
    We use:      audio <-> class_embedding["dog"]
    """

    def __init__(self, encoder, num_classes=50, embed_dim=768):
        self.encoder = encoder
        # Learnable class embeddings (like text embeddings)
        self.class_embeddings = nn.Parameter(
            torch.randn(num_classes, embed_dim) * 0.02
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, spectrograms, labels):
        # Encode audio
        audio_embeds = self.encode_audio(spectrograms)
        audio_embeds = F.normalize(audio_embeds, dim=1)

        # Get class embeddings for batch
        class_embeds = F.normalize(self.class_embeddings, dim=1)

        # CLIP-style contrastive loss
        logits_per_audio = audio_embeds @ class_embeds.T / self.temperature
        logits_per_class = logits_per_audio.T

        # Cross-entropy both directions
        loss_audio = F.cross_entropy(logits_per_audio, labels)
        loss_class = F.cross_entropy(logits_per_class, labels)

        return (loss_audio + loss_class) / 2

    def encode_audio(self, spectrograms):
        with torch.no_grad():
            latent, _, _ = self.encoder.forward_encoder(spectrograms, mask_ratio=0.0)
        return latent[:, 1:, :].mean(dim=1)  # Mean pooling
```

### Full CLAP with Text Encoder

For maximum semantic quality, use actual text descriptions:

```python
from transformers import AutoTokenizer, AutoModel

class AudioCLAPTrainer:
    def __init__(self, audio_encoder, text_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.audio_encoder = audio_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_encoder = AutoModel.from_pretrained(text_model)

        # Projection heads
        self.audio_proj = nn.Linear(768, 512)
        self.text_proj = nn.Linear(384, 512)  # MiniLM dim

        self.temperature = 0.07

    def encode_text(self, descriptions):
        """Encode text descriptions like 'sound of dog barking'."""
        inputs = self.tokenizer(descriptions, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pool

    def forward(self, spectrograms, descriptions):
        # Encode both modalities
        audio_embeds = self.encode_audio(spectrograms)
        text_embeds = self.encode_text(descriptions)

        # Project to shared space
        audio_embeds = F.normalize(self.audio_proj(audio_embeds), dim=1)
        text_embeds = F.normalize(self.text_proj(text_embeds), dim=1)

        # Contrastive loss
        logits = audio_embeds @ text_embeds.T / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss
```

---

## 5. Training Recommendations

### Mask Ratio

Your experiments with 80% → 85% mask ratio may not help because:

> "Masking ratio sweeps (50%, 65%, 75%, 85%) show optimal performance around 75%." - CAV-MAE ablations

**Recommendation**: Try 75% mask ratio instead of 80-85%.

### Training Duration

MAE benefits from long training:

> "Standard MAE requires an excessively long schedule of 1,600 pre-training epochs." - [MAE Paper](https://arxiv.org/abs/2111.06377)

However, with optimized recipes:
- Higher mask ratio (90%) can reduce epochs needed
- Contrastive loss accelerates convergence
- 100-300 epochs may suffice with contrastive regularization

### Recommended Training Recipe

```python
config = Config()

# Architecture
config.img_size = 384           # Higher resolution
config.patch_size = 16          # Standard patches (576 total)
config.embed_dim = 768          # Keep base size
config.encoder_depth = 12       # Keep depth

# Training
config.mask_ratio = 0.75        # Lower than current
config.contrast_weight = 0.01   # Add contrastive loss
config.learning_rate = 1.5e-4
config.weight_decay = 0.05
config.epochs = 200             # Can be shorter with contrastive
config.warmup_epochs = 10

# Embedding extraction
config.pooling = "mean"         # Not CLS
```

---

## 6. Experiment Checklist

### Phase 1: Quick Wins (No Retraining)
- [ ] Switch embedding extraction from CLS to mean pooling
- [ ] Evaluate on similarity search benchmark
- [ ] Compare CLS vs mean vs CLS+mean concatenation

### Phase 2: Training Modifications
- [ ] Reduce mask ratio from 85% to 75%
- [ ] Add contrastive loss with weight 0.01
- [ ] Add uniformity loss
- [ ] Train for 100-200 epochs

### Phase 3: Architecture Changes
- [ ] Increase resolution to 384×384
- [ ] Try overlapping patches (stride 10)
- [ ] Implement attention pooling

### Phase 4: Semantic Enhancement
- [ ] CLAP-lite training with class embeddings
- [ ] Full CLAP with text encoder (if compute allows)
- [ ] Fine-tune with triplet loss on ESC-50 classes

---

## References

1. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) - Original MAE paper
2. [Audio-MAE: Masked Autoencoders that Listen](https://arxiv.org/abs/2207.06405) - Audio MAE
3. [CAV-MAE: Contrastive Audio-Visual Masked Autoencoder](https://github.com/YuanGongND/cav-mae) - Contrastive + MAE
4. [CLAP: Contrastive Language-Audio Pretraining](https://github.com/LAION-AI/CLAP) - Semantic audio embeddings
5. [Audio Spectrogram Transformer](https://github.com/YuanGongND/ast) - AST architecture
6. [Understanding MAE from Local Contrastive Perspective](https://arxiv.org/html/2310.01994v2) - MAE analysis
7. [Sentence-BERT Pooling Analysis](https://www.sbert.net/) - Pooling strategies for embeddings

---

## Summary

**Start here**: Change CLS → mean pooling (5 minutes, likely +5-15% improvement)

**Next**: Add contrastive loss to training (requires retraining, +10-20% semantic quality)

**Advanced**: CLAP-style fine-tuning or higher resolution

The core insight is that **reconstruction loss alone doesn't optimize for similarity search** - you need either contrastive learning or a pooling strategy that aggregates all patch information.
