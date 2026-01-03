"""
Test script for AudioMAE++ improvements.

Tests:
1. New config parameters
2. Contrastive and uniformity losses
3. Mean pooling
4. Different image/patch dimensions
5. Training loop with new losses
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from audiomae import (
    Config,
    AudioMAEPlusPlus,
    info_nce_loss,
    uniformity_loss,
    get_embedding,
    count_parameters
)


def test_config():
    """Test new config parameters exist and have correct defaults."""
    print("\n" + "="*60)
    print("TEST 1: Config Parameters")
    print("="*60)

    config = Config()

    # Check new parameters exist
    assert hasattr(config, 'pooling_mode'), "Missing pooling_mode"
    assert hasattr(config, 'use_contrastive_loss'), "Missing use_contrastive_loss"
    assert hasattr(config, 'contrastive_weight'), "Missing contrastive_weight"
    assert hasattr(config, 'use_uniformity_loss'), "Missing use_uniformity_loss"
    assert hasattr(config, 'uniformity_weight'), "Missing uniformity_weight"

    # Check defaults
    assert config.mask_ratio == 0.75, f"Expected mask_ratio=0.75, got {config.mask_ratio}"
    assert config.pooling_mode == "mean", f"Expected pooling_mode='mean', got {config.pooling_mode}"
    assert config.use_contrastive_loss == True, "Expected use_contrastive_loss=True"
    assert config.use_uniformity_loss == True, "Expected use_uniformity_loss=True"

    print("[OK] All new config parameters present with correct defaults")
    print(f"  mask_ratio: {config.mask_ratio}")
    print(f"  pooling_mode: {config.pooling_mode}")
    print(f"  use_contrastive_loss: {config.use_contrastive_loss}")
    print(f"  contrastive_weight: {config.contrastive_weight}")
    print(f"  use_uniformity_loss: {config.use_uniformity_loss}")
    print(f"  uniformity_weight: {config.uniformity_weight}")

    return True


def test_loss_functions():
    """Test contrastive and uniformity loss functions."""
    print("\n" + "="*60)
    print("TEST 2: Loss Functions")
    print("="*60)

    batch_size = 8
    embed_dim = 768

    # Create fake embeddings and labels (requires_grad=True to test gradient flow)
    embeddings = torch.randn(batch_size, embed_dim, requires_grad=True)
    # Create labels with some duplicates (for positive pairs)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    # Test contrastive loss
    c_loss = info_nce_loss(embeddings, labels, temperature=0.07)
    assert not torch.isnan(c_loss), "Contrastive loss is NaN"
    assert c_loss.requires_grad, "Contrastive loss should require gradients"
    print(f"[OK] Contrastive loss: {c_loss.item():.4f}")

    # Test uniformity loss
    u_loss = uniformity_loss(embeddings, t=2.0)
    assert not torch.isnan(u_loss), "Uniformity loss is NaN"
    assert u_loss.requires_grad, "Uniformity loss should require gradients"
    print(f"[OK] Uniformity loss: {u_loss.item():.4f}")

    # Test edge case: no positive pairs
    unique_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    c_loss_no_pairs = info_nce_loss(embeddings, unique_labels)
    print(f"[OK] Contrastive loss (no pairs): {c_loss_no_pairs.item():.4f}")

    return True


def test_pooling():
    """Test get_embedding with different pooling modes."""
    print("\n" + "="*60)
    print("TEST 3: Pooling Modes")
    print("="*60)

    batch_size = 4
    num_patches = 196  # 14x14 for 224x224 with 16x16 patches
    embed_dim = 768

    # Simulate encoder output: [B, N+1, D] where N+1 includes CLS token
    latent = torch.randn(batch_size, num_patches + 1, embed_dim)

    # Test CLS pooling
    cls_embed = get_embedding(latent, mode="cls")
    assert cls_embed.shape == (batch_size, embed_dim), f"CLS shape wrong: {cls_embed.shape}"
    print(f"[OK] CLS pooling: {cls_embed.shape}")

    # Test mean pooling
    mean_embed = get_embedding(latent, mode="mean")
    assert mean_embed.shape == (batch_size, embed_dim), f"Mean shape wrong: {mean_embed.shape}"
    print(f"[OK] Mean pooling: {mean_embed.shape}")

    # Test cls+mean pooling
    concat_embed = get_embedding(latent, mode="cls+mean")
    assert concat_embed.shape == (batch_size, 2 * embed_dim), f"Concat shape wrong: {concat_embed.shape}"
    print(f"[OK] CLS+Mean pooling: {concat_embed.shape}")

    return True


def test_dimension_changes():
    """Test model works with different image sizes and patch sizes."""
    print("\n" + "="*60)
    print("TEST 4: Dimension Changes")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    test_configs = [
        # (img_size, patch_size, description)
        (224, 16, "Standard (224x224, 16x16 patches, 196 patches)"),
        (384, 16, "Higher res (384x384, 16x16 patches, 576 patches)"),
        (448, 32, "Large patches (448x448, 32x32 patches, 196 patches)"),
        (512, 32, "High res + large patches (512x512, 32x32 patches, 256 patches)"),
    ]

    for img_size, patch_size, desc in test_configs:
        print(f"\nTesting: {desc}")

        # Create config
        config = Config()
        config.img_size = img_size
        config.patch_size = patch_size
        config.num_patches = (img_size // patch_size) ** 2

        # Reduce model size for faster testing
        config.encoder_depth = 2
        config.decoder_depth = 2

        try:
            # Create model
            model = AudioMAEPlusPlus(config).to(device)

            # Create dummy input
            dummy_input = torch.randn(2, 3, img_size, img_size).to(device)
            dummy_labels = torch.tensor([0, 1]).to(device)

            # Test forward pass without labels (backward compatible)
            loss, pred, mask = model(dummy_input)
            print(f"  [OK] Forward (no labels): loss={loss.item():.4f}")

            # Test forward pass with labels (new losses)
            total_loss, pred, mask, loss_dict = model(dummy_input, labels=dummy_labels)
            print(f"  [OK] Forward (with labels): total={loss_dict['total']:.4f}, "
                  f"recon={loss_dict['reconstruction']:.4f}, "
                  f"contrastive={loss_dict['contrastive']:.4f}, "
                  f"uniformity={loss_dict['uniformity']:.4f}")

            # Test embedding extraction
            embedding = model.get_embedding(dummy_input)
            expected_dim = config.embed_dim if config.pooling_mode != "cls+mean" else 2 * config.embed_dim
            assert embedding.shape == (2, expected_dim), f"Wrong embedding shape: {embedding.shape}"
            print(f"  [OK] Embedding extraction: {embedding.shape}")

            # Cleanup
            del model
            torch.cuda.empty_cache() if device == "cuda" else None

        except Exception as e:
            print(f"  [FAILED] {e}")
            return False

    return True


def test_training_loop():
    """Test that training loop works with new losses."""
    print("\n" + "="*60)
    print("TEST 5: Training Loop")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create small model for testing
    config = Config()
    config.encoder_depth = 2
    config.decoder_depth = 2
    config.use_contrastive_loss = True
    config.use_uniformity_loss = True

    model = AudioMAEPlusPlus(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate a few training steps
    print("Running 3 training steps...")
    model.train()

    for step in range(3):
        # Create dummy batch with labels
        batch_size = 8
        imgs = torch.randn(batch_size, 3, config.img_size, config.img_size).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        # Forward pass
        total_loss, pred, mask, loss_dict = model(imgs, labels=labels)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"  Step {step+1}: total={loss_dict['total']:.4f}, "
              f"recon={loss_dict['reconstruction']:.4f}, "
              f"contrast={loss_dict['contrastive']:.4f}, "
              f"uniform={loss_dict['uniformity']:.4f}")

    print("[OK] Training loop completed successfully")

    # Test with losses disabled
    print("\nTesting with losses disabled...")
    config.use_contrastive_loss = False
    config.use_uniformity_loss = False
    model = AudioMAEPlusPlus(config).to(device)

    imgs = torch.randn(4, 3, config.img_size, config.img_size).to(device)

    # Without labels - should return 3 values
    output = model(imgs)
    assert len(output) == 3, f"Expected 3 outputs without labels, got {len(output)}"
    print("[OK] Forward without labels returns 3 values")

    # With labels but losses disabled - should still return 3 values
    labels = torch.randint(0, 10, (4,)).to(device)
    output = model(imgs, labels=labels)
    assert len(output) == 3, f"Expected 3 outputs with losses disabled, got {len(output)}"
    print("[OK] Forward with losses disabled returns 3 values")

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("AudioMAE++ Improvements Test Suite")
    print("="*60)

    tests = [
        ("Config Parameters", test_config),
        ("Loss Functions", test_loss_functions),
        ("Pooling Modes", test_pooling),
        ("Dimension Changes", test_dimension_changes),
        ("Training Loop", test_training_loop),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAILED] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[FAILURE] Some tests failed.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
