"""
Loss functions for embedding quality and training.

Components:
- info_nce_loss: Contrastive loss for semantic clustering
- uniformity_loss: Prevents embedding collapse
- get_embedding: Pooling helper for embedding extraction
"""

import torch
import torch.nn.functional as F


def get_embedding(latent: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    Extract embedding from encoder output using specified pooling.

    Args:
        latent: Encoder output [B, N+1, D] (CLS token at position 0)
        mode: Pooling mode
            - "cls": CLS token only
            - "mean": Mean of patch tokens (excludes CLS)
            - "cls+mean": Concatenation of CLS and mean

    Returns:
        Embedding [B, D] or [B, 2*D] for cls+mean
    """
    if mode == "cls":
        return latent[:, 0, :]
    elif mode == "mean":
        # Mean of patch tokens (exclude CLS at position 0)
        return latent[:, 1:, :].mean(dim=1)
    elif mode == "cls+mean":
        cls_embed = latent[:, 0, :]
        mean_embed = latent[:, 1:, :].mean(dim=1)
        return torch.cat([cls_embed, mean_embed], dim=1)
    else:
        raise ValueError(
            f"Unknown pooling mode: {mode}. Use 'cls', 'mean', or 'cls+mean'"
        )


def info_nce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for same-class positive pairs.

    Encourages embeddings of the same class to be closer together
    while pushing different classes apart. This improves semantic
    clustering of the embedding space.

    Args:
        embeddings: Embeddings [B, D] (will be L2 normalized)
        labels: Class labels [B]
        temperature: Softmax temperature (lower = harder negatives)

    Returns:
        Scalar loss value
    """
    # Normalize embeddings to unit sphere
    embeddings = F.normalize(embeddings, dim=1)
    batch_size = embeddings.shape[0]

    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Create positive mask: same class = positive pair
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)  # Exclude self-similarity

    # Check if we have any positive pairs
    num_positives = pos_mask.sum(dim=1)
    valid_samples = num_positives > 0

    if not valid_samples.any():
        # No positive pairs in batch, return zero loss
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Compute log softmax
    # Mask out self-similarity by setting diagonal to large negative
    logits_mask = torch.ones_like(sim_matrix)
    logits_mask.fill_diagonal_(0)

    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

    # Loss is negative log-likelihood (only for samples with positives)
    loss = -mean_log_prob_pos[valid_samples].mean()

    return loss


def uniformity_loss(embeddings: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity loss from "Understanding Contrastive Representation Learning".

    Encourages embeddings to be uniformly distributed on the unit hypersphere,
    preventing dimensional collapse where embeddings cluster too tightly.

    Args:
        embeddings: Embeddings [B, D] (will be L2 normalized)
        t: Temperature parameter (higher = stronger push apart)

    Returns:
        Scalar loss value (lower is more uniform)
    """
    # Normalize embeddings to unit sphere
    embeddings = F.normalize(embeddings, dim=1)

    # Compute pairwise squared distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 2 - 2*a.b (for unit vectors)
    sq_pdist = torch.pdist(embeddings, p=2).pow(2)

    # Uniformity loss: log of average Gaussian kernel
    loss = sq_pdist.mul(-t).exp().mean().log()

    return loss


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    normalize_target: bool = True
) -> torch.Tensor:
    """
    Compute MSE reconstruction loss on masked patches only.

    Args:
        pred: Predicted patches [B, N, patch_pixels]
        target: Target patches [B, N, patch_pixels]
        mask: Binary mask [B, N] (1=masked, 0=visible)
        normalize_target: Whether to normalize target per-patch

    Returns:
        Scalar loss value
    """
    if normalize_target:
        # Normalize target (per-patch)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

    # MSE loss on masked patches only
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # Mean over patch pixels
    loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches

    return loss
