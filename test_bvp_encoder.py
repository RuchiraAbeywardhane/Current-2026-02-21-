"""
BVP Encoder Test Script
========================

This script tests the BVP encoder module to verify:
- Encoder architecture (BiLSTM with/without attention)
- Forward pass with real BVP data
- Output shape verification
- Integration with BVP data loader
- Gradient flow check
- Feature visualization

Usage:
    python test_bvp_encoder.py

Author: Final Year Project
Date: 2026
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Import BVP modules
from bvp_data_loader import load_bvp_data
from bvp_encoder import BVPEncoder, BVPEncoderWithAttention


# ==================================================
# CONFIGURATION
# ==================================================

class TestConfig:
    """Configuration for testing BVP encoder."""
    # Data path
    DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
    
    # BVP parameters
    BVP_FS = 64
    BVP_WINDOW_SEC = 10
    BVP_OVERLAP = 0.0
    
    # BVP Preprocessing
    USE_BVP_BASELINE_CORRECTION = True
    
    # Encoder parameters
    INPUT_SIZE = 1  # Raw BVP signal
    HIDDEN_SIZE = 32
    DROPOUT = 0.3
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_CLASSES = 4
    SUBJECT_INDEPENDENT = True
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Label mappings
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",
        "FEAR": "Q2",
        "SADNESS": "Q3",
        "NEUTRAL": "Q4",
    }


# Set random seeds
config = TestConfig()
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


# ==================================================
# TEST FUNCTIONS
# ==================================================

def test_encoder_architecture():
    """Test 1: Verify encoder architecture and output shapes."""
    print("\n" + "="*80)
    print("TEST 1: ENCODER ARCHITECTURE")
    print("="*80)
    
    batch_size = 8
    time_steps = config.BVP_WINDOW_SEC * config.BVP_FS  # 640 samples
    
    # Create dummy data
    dummy_input = torch.randn(batch_size, time_steps, config.INPUT_SIZE)
    print(f"\n📊 Test Input Shape: {dummy_input.shape}")
    print(f"   [Batch, Time, Features] = [{batch_size}, {time_steps}, {config.INPUT_SIZE}]")
    
    # Test BVPEncoder (Average Pooling)
    print("\n🔹 Testing BVPEncoder (Average Pooling):")
    encoder = BVPEncoder(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    dummy_input = dummy_input.to(config.DEVICE)
    
    # Test context output
    context = encoder(dummy_input, return_sequence=False)
    print(f"   ✅ Context output shape: {context.shape}")
    print(f"      Expected: [{batch_size}, {config.HIDDEN_SIZE * 2}]")
    
    # Test sequence output
    context, features = encoder(dummy_input, return_sequence=True)
    print(f"   ✅ Context shape: {context.shape}")
    print(f"   ✅ Features shape: {features.shape}")
    print(f"      Expected: [{batch_size}, {time_steps}, {config.HIDDEN_SIZE * 2}]")
    
    # Test BVPEncoderWithAttention
    print("\n🔹 Testing BVPEncoderWithAttention:")
    encoder_attn = BVPEncoderWithAttention(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    context_attn = encoder_attn(dummy_input, return_sequence=False)
    print(f"   ✅ Attention context shape: {context_attn.shape}")
    
    # Compare outputs
    print(f"\n📊 Output Dimensions:")
    print(f"   Encoder output dim: {encoder.get_output_dim()}")
    print(f"   Attention encoder output dim: {encoder_attn.get_output_dim()}")
    
    return encoder, encoder_attn


def test_parameter_count():
    """Test 2: Count and display model parameters."""
    print("\n" + "="*80)
    print("TEST 2: MODEL PARAMETERS")
    print("="*80)
    
    encoder = BVPEncoder(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    )
    
    encoder_attn = BVPEncoderWithAttention(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    )
    
    # Count parameters
    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    total_basic, train_basic = count_params(encoder)
    total_attn, train_attn = count_params(encoder_attn)
    
    print(f"\n🔹 BVPEncoder (Average Pooling):")
    print(f"   Total parameters: {total_basic:,}")
    print(f"   Trainable parameters: {train_basic:,}")
    
    print(f"\n🔹 BVPEncoderWithAttention:")
    print(f"   Total parameters: {total_attn:,}")
    print(f"   Trainable parameters: {train_attn:,}")
    print(f"   Additional params (attention): {total_attn - total_basic:,}")
    
    # Display layer-by-layer breakdown
    print(f"\n📋 Layer Breakdown (BVPEncoder):")
    for name, param in encoder.named_parameters():
        print(f"   {name:30s} | Shape: {str(list(param.shape)):20s} | Params: {param.numel():,}")


def test_with_real_data():
    """Test 3: Test encoder with real BVP data from data loader."""
    print("\n" + "="*80)
    print("TEST 3: ENCODER WITH REAL BVP DATA")
    print("="*80)
    
    try:
        # Load real BVP data
        print("\n📂 Loading real BVP data...")
        X_raw, y_labels, subject_ids, label_to_id = load_bvp_data(config.DATA_ROOT, config)
        
        print(f"\n✅ Data loaded: {X_raw.shape}")
        
        # Take a small batch
        batch_size = min(config.BATCH_SIZE, len(X_raw))
        X_batch = X_raw[:batch_size]  # (B, T)
        y_batch = y_labels[:batch_size]
        
        # Convert to PyTorch tensors and add feature dimension
        X_batch = torch.from_numpy(X_batch).float().unsqueeze(-1)  # (B, T, 1)
        y_batch = torch.from_numpy(y_batch).long()
        
        X_batch = X_batch.to(config.DEVICE)
        y_batch = y_batch.to(config.DEVICE)
        
        print(f"\n📊 Batch Statistics:")
        print(f"   Input shape: {X_batch.shape}")
        print(f"   Labels shape: {y_batch.shape}")
        print(f"   Input range: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
        print(f"   Label distribution: {torch.bincount(y_batch).cpu().numpy()}")
        
        # Test encoder
        encoder = BVPEncoder(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        encoder.eval()
        with torch.no_grad():
            context = encoder(X_batch, return_sequence=False)
            context_seq, features = encoder(X_batch, return_sequence=True)
        
        print(f"\n✅ Encoder Forward Pass:")
        print(f"   Context shape: {context.shape}")
        print(f"   Features shape: {features.shape}")
        print(f"   Context range: [{context.min():.3f}, {context.max():.3f}]")
        print(f"   Features range: [{features.min():.3f}, {features.max():.3f}]")
        
        # Test attention encoder
        encoder_attn = BVPEncoderWithAttention(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        encoder_attn.eval()
        with torch.no_grad():
            context_attn = encoder_attn(X_batch, return_sequence=False)
        
        print(f"\n✅ Attention Encoder Forward Pass:")
        print(f"   Context shape: {context_attn.shape}")
        print(f"   Context range: [{context_attn.min():.3f}, {context_attn.max():.3f}]")
        
        # Compare outputs
        print(f"\n📊 Output Comparison:")
        print(f"   Average pooling mean: {context.mean():.3f}")
        print(f"   Attention pooling mean: {context_attn.mean():.3f}")
        print(f"   Difference: {torch.abs(context - context_attn).mean():.3f}")
        
        return X_batch, encoder, encoder_attn
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_gradient_flow(encoder, X_batch):
    """Test 4: Verify gradients flow correctly through the encoder."""
    print("\n" + "="*80)
    print("TEST 4: GRADIENT FLOW CHECK")
    print("="*80)
    
    if encoder is None or X_batch is None:
        print("❌ Skipping (no encoder or data)")
        return
    
    # Create a simple classifier on top
    encoder.train()
    classifier = nn.Linear(encoder.get_output_dim(), config.NUM_CLASSES).to(config.DEVICE)
    
    # Forward pass
    X_batch = X_batch.to(config.DEVICE)
    context = encoder(X_batch)
    logits = classifier(context)
    
    # Dummy loss
    dummy_target = torch.randint(0, config.NUM_CLASSES, (X_batch.size(0),)).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, dummy_target)
    
    print(f"\n📊 Forward Pass:")
    print(f"   Context shape: {context.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"\n📊 Gradient Check:")
    has_grad = 0
    no_grad = 0
    total_grad_norm = 0.0
    
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            has_grad += 1
            status = "✅" if grad_norm > 0 else "⚠️"
            print(f"   {status} {name:30s} | Grad norm: {grad_norm:.6f}")
        else:
            no_grad += 1
            print(f"   ❌ {name:30s} | No gradient")
    
    print(f"\n✅ Gradient Summary:")
    print(f"   Parameters with gradients: {has_grad}")
    print(f"   Parameters without gradients: {no_grad}")
    print(f"   Total gradient norm: {total_grad_norm:.6f}")
    
    if total_grad_norm > 0:
        print(f"   ✅ Gradients are flowing correctly!")
    else:
        print(f"   ⚠️  Warning: No gradients detected!")


def test_feature_visualization(X_batch, encoder):
    """Test 5: Visualize encoder features."""
    print("\n" + "="*80)
    print("TEST 5: FEATURE VISUALIZATION")
    print("="*80)
    
    if encoder is None or X_batch is None:
        print("❌ Skipping (no encoder or data)")
        return
    
    encoder.eval()
    X_batch = X_batch.to(config.DEVICE)
    
    with torch.no_grad():
        context, features = encoder(X_batch, return_sequence=True)
    
    # Convert to numpy
    X_np = X_batch.cpu().numpy()
    features_np = features.cpu().numpy()
    context_np = context.cpu().numpy()
    
    # Visualize first sample
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Raw BVP signal
    time = np.arange(X_np.shape[1]) / config.BVP_FS
    axes[0].plot(time, X_np[0, :, 0], 'b-', linewidth=0.8)
    axes[0].set_title('Raw BVP Signal (First Sample)', fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Encoded temporal features (heatmap)
    im = axes[1].imshow(features_np[0].T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title('Encoded Temporal Features [T, 64]', fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Feature Dimension')
    plt.colorbar(im, ax=axes[1])
    
    # Plot 3: Global context vector
    axes[2].bar(range(len(context_np[0])), context_np[0], color='steelblue', alpha=0.7)
    axes[2].set_title('Global Context Vector [64]', fontweight='bold')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('bvp_encoder_features.png', dpi=300, bbox_inches='tight')
    print("\n✅ Feature visualization saved: bvp_encoder_features.png")
    plt.close()
    
    # Statistics
    print(f"\n📊 Feature Statistics:")
    print(f"   Temporal features shape: {features_np.shape}")
    print(f"   Context vector shape: {context_np.shape}")
    print(f"   Context mean: {context_np.mean():.3f}")
    print(f"   Context std: {context_np.std():.3f}")
    print(f"   Context range: [{context_np.min():.3f}, {context_np.max():.3f}]")


def test_batch_processing():
    """Test 6: Test encoder with different batch sizes."""
    print("\n" + "="*80)
    print("TEST 6: BATCH PROCESSING")
    print("="*80)
    
    encoder = BVPEncoder(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    encoder.eval()
    
    time_steps = config.BVP_WINDOW_SEC * config.BVP_FS
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    print(f"\n📊 Testing different batch sizes:")
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, time_steps, config.INPUT_SIZE).to(config.DEVICE)
        
        with torch.no_grad():
            context = encoder(dummy_input)
        
        status = "✅" if context.shape == (batch_size, encoder.get_output_dim()) else "❌"
        print(f"   {status} Batch size {batch_size:3d}: Output shape {context.shape}")


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """Run all BVP encoder tests."""
    print("="*80)
    print("BVP ENCODER TEST SUITE")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Input size: {config.INPUT_SIZE}")
    print(f"Hidden size: {config.HIDDEN_SIZE}")
    print(f"Output size: {config.HIDDEN_SIZE * 2}")
    print("="*80)
    
    # Test 1: Architecture
    encoder, encoder_attn = test_encoder_architecture()
    
    # Test 2: Parameter count
    test_parameter_count()
    
    # Test 3: Real data
    X_batch, encoder, encoder_attn = test_with_real_data()
    
    # Test 4: Gradient flow
    if encoder is not None and X_batch is not None:
        test_gradient_flow(encoder, X_batch)
    
    # Test 5: Feature visualization
    if encoder is not None and X_batch is not None:
        test_feature_visualization(X_batch, encoder)
    
    # Test 6: Batch processing
    test_batch_processing()
    
    print("\n" + "="*80)
    print("🎉 ALL ENCODER TESTS COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - bvp_encoder_features.png (feature visualization)")
    print("="*80)


if __name__ == "__main__":
    main()
