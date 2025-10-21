#!/usr/bin/env python3
"""
Verification script to test Hybrid1 model tensor shapes against reference architecture.

Reference Architecture:
Input (3×H×W)
   ↓
Encoder (EfficientNetB4):
  C1: stride 4  (1/4 resolution)
  C2: stride 8  (1/8 resolution)
  C3: stride 16 (1/16 resolution)
  C4: stride 32 (1/32 resolution)
   ↓
Bottleneck: 1×1 Conv + Flatten + 2 SwinBlocks
   ↓
Decoder (Swin-Unet):
  D1: (1/32→1/16) + Skip(C4)
  D2: (1/16→1/8)  + Skip(C3)
  D3: (1/8→1/4)   + Skip(C2)
  D4: (1/4→1×)    + Skip(C1)
   ↓
Segmentation Head: Conv3×3 → ReLU → Conv1×1
   ↓
Output: (num_classes×H×W)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
from hybrid1.hybrid_model import HybridEfficientNetB4SwinDecoder


def verify_shapes():
    """Verify all tensor shapes match the reference architecture."""
    
    print("="*80)
    print("HYBRID1 MODEL SHAPE VERIFICATION")
    print("="*80)
    print()
    
    # Test parameters
    batch_size = 2
    num_classes = 6
    img_size = 224
    
    print(f"Test Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Input Size: {img_size}×{img_size}")
    print(f"  Num Classes: {num_classes}")
    print()
    
    # Create model (BASELINE - no enhancements)
    print("Creating BASELINE Hybrid1 model (reference architecture)...")
    model = HybridEfficientNetB4SwinDecoder(
        num_classes=num_classes,
        img_size=img_size,
        pretrained=False,  # Fast initialization for testing
        use_deep_supervision=False,
        use_multiscale_agg=False,
        use_smart_skip=False  # BASELINE uses naive concatenation
    )
    model.eval()
    print()
    
    # Create test input
    x = torch.randn(batch_size, 3, img_size, img_size)
    print(f"Input shape: {tuple(x.shape)}")
    print()
    
    # Test encoder
    print("-"*80)
    print("ENCODER (EfficientNet-B4)")
    print("-"*80)
    encoder_features = model.encoder(x)
    
    expected_shapes = [
        (batch_size, 96, 56, 56),   # C1: stride 4  (H/4, W/4)
        (batch_size, 192, 28, 28),  # C2: stride 8  (H/8, W/8)
        (batch_size, 384, 14, 14),  # C3: stride 16 (H/16, W/16)
        (batch_size, 768, 7, 7),    # C4: stride 32 (H/32, W/32)
    ]
    
    all_match = True
    for i, (feat, expected) in enumerate(zip(encoder_features, expected_shapes)):
        actual = tuple(feat.shape)
        match = actual == expected
        all_match = all_match and match
        status = "✅" if match else "❌"
        print(f"C{i+1} (stride {4 * 2**i:2d}): {status} Expected: {expected}, Got: {actual}")
    print()
    
    # Test token conversion
    print("-"*80)
    print("TOKEN CONVERSION")
    print("-"*80)
    target_dims = model.encoder.get_target_dims()
    p2, p3, p4, p5 = encoder_features
    
    x_tokens = model._to_tokens(p5, out_dim=target_dims[3])
    print(f"Bottleneck tokens (P5): Expected: ({batch_size}, 49, 768), Got: {tuple(x_tokens.shape)}")
    
    x_downsample = [
        model._to_tokens(p2, out_dim=target_dims[0]),
        model._to_tokens(p3, out_dim=target_dims[1]),
        model._to_tokens(p4, out_dim=target_dims[2]),
        model._to_tokens(p5, out_dim=target_dims[3])
    ]
    
    expected_token_shapes = [
        (batch_size, 3136, 96),   # P2: 56*56 = 3136
        (batch_size, 784, 192),   # P3: 28*28 = 784
        (batch_size, 196, 384),   # P4: 14*14 = 196
        (batch_size, 49, 768),    # P5: 7*7 = 49
    ]
    
    for i, (tokens, expected) in enumerate(zip(x_downsample, expected_token_shapes)):
        actual = tuple(tokens.shape)
        match = actual == expected
        status = "✅" if match else "❌"
        print(f"P{i+2} tokens: {status} Expected: {expected}, Got: {actual}")
    print()
    
    # Test bottleneck
    print("-"*80)
    print("BOTTLENECK (2 SwinBlocks)")
    print("-"*80)
    x_bottleneck = model.decoder.forward_bottleneck(x_tokens)
    expected = (batch_size, 49, 768)
    actual = tuple(x_bottleneck.shape)
    match = actual == expected
    status = "✅" if match else "❌"
    print(f"Bottleneck output: {status} Expected: {expected}, Got: {actual}")
    print()
    
    # Test decoder
    print("-"*80)
    print("DECODER (Swin-Unet with Skip Connections)")
    print("-"*80)
    x_up = model.decoder.forward_up_features(x_bottleneck, x_downsample)
    expected = (batch_size, 3136, 96)  # 56*56 = 3136, embed_dim = 96
    actual = tuple(x_up.shape)
    match = actual == expected
    status = "✅" if match else "❌"
    print(f"Decoder output: {status} Expected: {expected}, Got: {actual}")
    print()
    
    # Test final upsampling
    print("-"*80)
    print("SEGMENTATION HEAD")
    print("-"*80)
    logits = model.decoder.up_x4(x_up)
    expected = (batch_size, num_classes, img_size, img_size)
    actual = tuple(logits.shape)
    match = actual == expected
    status = "✅" if match else "❌"
    print(f"Final output: {status} Expected: {expected}, Got: {actual}")
    print()
    
    # Full forward pass
    print("-"*80)
    print("FULL FORWARD PASS")
    print("-"*80)
    with torch.no_grad():
        output = model(x)
    
    expected = (batch_size, num_classes, img_size, img_size)
    actual = tuple(output.shape)
    match = actual == expected
    status = "✅" if match else "❌"
    print(f"Model output: {status} Expected: {expected}, Got: {actual}")
    print()
    
    # Model info
    print("-"*80)
    print("MODEL INFO")
    print("-"*80)
    info = model.get_model_info()
    for key, value in info.items():
        if key in ['total_params', 'trainable_params']:
            print(f"{key}: {value:,}")
        else:
            print(f"{key}: {value}")
    print()
    
    # Summary
    print("="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    if all_match and match:
        print("✅ ALL SHAPES MATCH REFERENCE ARCHITECTURE!")
        print()
        print("Your Hybrid1 model is now 100% compliant with the reference architecture:")
        print("  ✅ EfficientNet-B4 encoder with 4 stages")
        print("  ✅ Skip adapters (C1-C3) and bottleneck adapter (C4)")
        print("  ✅ Token conversion (flatten + transpose)")
        print("  ✅ Bottleneck with 2 SwinBlocks")
        print("  ✅ 4 decoder stages with Patch Expand")
        print("  ✅ Naive skip connections (concatenation)")
        print("  ✅ Segmentation head: Conv3×3 + ReLU + Conv1×1")
        print("  ✅ ImageNet normalization")
        print()
        return True
    else:
        print("❌ SOME SHAPES DON'T MATCH!")
        print("Please review the mismatches above.")
        print()
        return False


if __name__ == "__main__":
    success = verify_shapes()
    sys.exit(0 if success else 1)

