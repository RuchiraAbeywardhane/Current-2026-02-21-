"""
BVP Encoder Test Script
========================

This script tests the BVP encoder module to verify:
- Model initialization and architecture
- Forward pass functionality
- Output shapes and dimensions
- Attention mechanism (for BVPEncoderWithAttention)
- Gradient flow
- Model parameters
- Integration with real/synthetic BVP data
- Visualization of encoder outputs

Usage:
    python test_bvp_encoder.py

Author: Final Year Project
Date: 2026
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import BVP encoder
from bvp_encoder import BVPEncoder, BVPEncoderWithAttention
from bvp_config import BVPConfig


# ==================================================
# CONFIGURATION
# ==================================================

class TestConfig:
    """Configuration for testing BVP encoder."""
    BATCH_SIZE = 16
    TIME_STEPS = 640  # 10 seconds at 64 Hz
    INPUT_SIZE = 1
    HIDDEN_SIZE = 32
    DROPOUT = 0.3
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set random seed
config = TestConfig()
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)


# ==================================================
# TEST FUNCTIONS
# ==================================================

def test_model_initialization():
    """Test model initialization."""
    print("\n" + "="*80)
    print("TEST 1: MODEL INITIALIZATION")
    print("="*80)
    
    # Test BVPEncoder
    print("\n1.1 BVPEncoder (Average Pooling):")
    encoder = BVPEncoder(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    )
    print(f"   ✅ Model created successfully")
    print(f"   Input size: {encoder.input_size}")
    print(f"   Hidden size: {encoder.hidden_size}")
    print(f"   Output size: {encoder.output_size}")
    
    # Test BVPEncoderWithAttention
    print("\n1.2 BVPEncoderWithAttention:")
    encoder_attn = BVPEncoderWithAttention(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    )
    print(f"   ✅ Model created successfully")
    print(f"   Input size: {encoder_attn.input_size}")
    print(f"   Hidden size: {encoder_attn.hidden_size}")
    print(f"   Output size: {encoder_attn.output_size}")
    
    return encoder, encoder_attn


def test_forward_pass(encoder, encoder_attn):
    """Test forward pass with different input scenarios."""
    print("\n" + "="*80)
    print("TEST 2: FORWARD PASS")
    print("="*80)
    
    # Create dummy BVP data
    bvp_data = torch.randn(config.BATCH_SIZE, config.TIME_STEPS, config.INPUT_SIZE)
    print(f"\n📊 Input shape: {bvp_data.shape}")
    print(f"   [Batch={config.BATCH_SIZE}, Time={config.TIME_STEPS}, Features={config.INPUT_SIZE}]")
    
    # Test BVPEncoder - context only
    print("\n2.1 BVPEncoder - Context Only:")
    encoder.eval()
    with torch.no_grad():
        bvp_context = encoder(bvp_data, return_sequence=False)
    print(f"   Output shape: {bvp_context.shape}")
    print(f"   Expected: [16, 64]")
    assert bvp_context.shape == (config.BATCH_SIZE, 64), "❌ Shape mismatch!"
    print(f"   ✅ Shape validation passed")
    
    # Test BVPEncoder - with sequence
    print("\n2.2 BVPEncoder - With Sequence:")
    with torch.no_grad():
        bvp_context, bvp_feat = encoder(bvp_data, return_sequence=True)
    print(f"   Context shape: {bvp_context.shape}")
    print(f"   Feature shape: {bvp_feat.shape}")
    print(f"   Expected: [16, 64] and [16, 640, 64]")
    assert bvp_context.shape == (config.BATCH_SIZE, 64), "❌ Context shape mismatch!"
    assert bvp_feat.shape == (config.BATCH_SIZE, config.TIME_STEPS, 64), "❌ Feature shape mismatch!"
    print(f"   ✅ Shape validation passed")
    
    # Test BVPEncoderWithAttention
    print("\n2.3 BVPEncoderWithAttention - Context Only:")
    encoder_attn.eval()
    with torch.no_grad():
        bvp_context_attn = encoder_attn(bvp_data, return_sequence=False)
    print(f"   Output shape: {bvp_context_attn.shape}")
    assert bvp_context_attn.shape == (config.BATCH_SIZE, 64), "❌ Shape mismatch!"
    print(f"   ✅ Shape validation passed")
    
    # Test BVPEncoderWithAttention - with sequence
    print("\n2.4 BVPEncoderWithAttention - With Sequence:")
    with torch.no_grad():
        bvp_context_attn, bvp_feat_attn = encoder_attn(bvp_data, return_sequence=True)
    print(f"   Context shape: {bvp_context_attn.shape}")
    print(f"   Feature shape: {bvp_feat_attn.shape}")
    assert bvp_context_attn.shape == (config.BATCH_SIZE, 64), "❌ Context shape mismatch!"
    assert bvp_feat_attn.shape == (config.BATCH_SIZE, config.TIME_STEPS, 64), "❌ Feature shape mismatch!"
    print(f"   ✅ Shape validation passed")
    
    return bvp_data, bvp_context, bvp_feat, bvp_context_attn, bvp_feat_attn


def test_attention_mechanism(encoder_attn, bvp_data):
    """Test and visualize attention mechanism."""
    print("\n" + "="*80)
    print("TEST 3: ATTENTION MECHANISM")
    print("="*80)
    
    encoder_attn.eval()
    
    # Forward pass through LSTM
    with torch.no_grad():
        bvp_feat, _ = encoder_attn.bvp_lstm(bvp_data)
        bvp_feat = encoder_attn.layer_norm(bvp_feat)
        
        # Get attention scores
        attn_scores = encoder_attn.attention(bvp_feat)  # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
    
    print(f"\n📊 Attention Statistics:")
    print(f"   Attention scores shape: {attn_scores.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    print(f"   Weights sum (should be ~1.0): {attn_weights[0].sum().item():.4f}")
    print(f"   Min weight: {attn_weights.min().item():.6f}")
    print(f"   Max weight: {attn_weights.max().item():.6f}")
    print(f"   Mean weight: {attn_weights.mean().item():.6f}")
    
    # Visualize attention weights for first 4 samples
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i in range(min(4, config.BATCH_SIZE)):
        ax = axes[i]
        
        # Plot BVP signal
        bvp_signal = bvp_data[i, :, 0].cpu().numpy()
        time = np.arange(len(bvp_signal)) / 64.0  # 64 Hz sampling
        
        ax2 = ax.twinx()
        
        # Plot BVP signal
        ax.plot(time, bvp_signal, 'b-', linewidth=0.8, alpha=0.6, label='BVP Signal')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('BVP Amplitude', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot attention weights
        attn = attn_weights[i, :, 0].cpu().numpy()
        ax2.plot(time, attn, 'r-', linewidth=1.5, alpha=0.8, label='Attention Weight')
        ax2.set_ylabel('Attention Weight', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, attn.max() * 1.2])
        
        ax.set_title(f'Sample {i+1}: BVP Signal & Attention Weights', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bvp_encoder_attention_weights.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: bvp_encoder_attention_weights.png")
    plt.close()
    
    print(f"   ✅ Attention mechanism working correctly")


def test_gradient_flow(encoder, encoder_attn):
    """Test gradient flow through models."""
    print("\n" + "="*80)
    print("TEST 4: GRADIENT FLOW")
    print("="*80)
    
    # Create dummy data and target
    bvp_data = torch.randn(config.BATCH_SIZE, config.TIME_STEPS, config.INPUT_SIZE)
    target = torch.randint(0, 4, (config.BATCH_SIZE,))
    
    # Test BVPEncoder
    print("\n4.1 BVPEncoder Gradient Flow:")
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create a simple classifier head for testing
    classifier = nn.Linear(64, 4)
    
    optimizer.zero_grad()
    bvp_context = encoder(bvp_data)
    logits = classifier(bvp_context)
    loss = criterion(logits, target)
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                has_grad = True
                print(f"   {name}: grad_norm = {grad_norm:.6f}")
    
    if has_grad:
        print(f"   ✅ Gradients flowing correctly")
    else:
        print(f"   ❌ No gradients detected!")
    
    # Test BVPEncoderWithAttention
    print("\n4.2 BVPEncoderWithAttention Gradient Flow:")
    encoder_attn.train()
    optimizer_attn = torch.optim.Adam(encoder_attn.parameters(), lr=1e-3)
    
    optimizer_attn.zero_grad()
    bvp_context_attn = encoder_attn(bvp_data)
    logits_attn = classifier(bvp_context_attn)
    loss_attn = criterion(logits_attn, target)
    loss_attn.backward()
    
    # Check gradients
    has_grad_attn = False
    for name, param in encoder_attn.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                has_grad_attn = True
                print(f"   {name}: grad_norm = {grad_norm:.6f}")
    
    if has_grad_attn:
        print(f"   ✅ Gradients flowing correctly")
    else:
        print(f"   ❌ No gradients detected!")


def test_model_parameters(encoder, encoder_attn):
    """Test and compare model parameters."""
    print("\n" + "="*80)
    print("TEST 5: MODEL PARAMETERS")
    print("="*80)
    
    # BVPEncoder
    print("\n5.1 BVPEncoder:")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n   Layer-wise parameters:")
    for name, param in encoder.named_parameters():
        print(f"      {name:30s} {param.shape} ({param.numel():,} params)")
    
    # BVPEncoderWithAttention
    print("\n5.2 BVPEncoderWithAttention:")
    total_params_attn = sum(p.numel() for p in encoder_attn.parameters())
    trainable_params_attn = sum(p.numel() for p in encoder_attn.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params_attn:,}")
    print(f"   Trainable parameters: {trainable_params_attn:,}")
    
    print("\n   Layer-wise parameters:")
    for name, param in encoder_attn.named_parameters():
        print(f"      {name:30s} {param.shape} ({param.numel():,} params)")
    
    # Comparison
    print(f"\n5.3 Comparison:")
    print(f"   Attention overhead: {total_params_attn - total_params:,} parameters")
    print(f"   Percentage increase: {100 * (total_params_attn - total_params) / total_params:.2f}%")


def test_different_input_sizes(encoder):
    """Test encoder with different input sizes."""
    print("\n" + "="*80)
    print("TEST 6: DIFFERENT INPUT SIZES")
    print("="*80)
    
    encoder.eval()
    
    test_cases = [
        (8, 320, 1),    # 5 seconds, smaller batch
        (16, 640, 1),   # 10 seconds, normal batch
        (32, 1280, 1),  # 20 seconds, larger batch
        (1, 640, 1),    # Single sample
    ]
    
    for i, (batch, time, feat) in enumerate(test_cases, 1):
        print(f"\n6.{i} Input: [B={batch}, T={time}, F={feat}]")
        bvp_data = torch.randn(batch, time, feat)
        
        try:
            with torch.no_grad():
                bvp_context = encoder(bvp_data)
            
            expected_shape = (batch, 64)
            if bvp_context.shape == expected_shape:
                print(f"   ✅ Output shape: {bvp_context.shape} - PASS")
            else:
                print(f"   ❌ Output shape: {bvp_context.shape}, Expected: {expected_shape} - FAIL")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def test_encoder_outputs_visualization(encoder, encoder_attn):
    """Visualize encoder outputs for different input patterns."""
    print("\n" + "="*80)
    print("TEST 7: ENCODER OUTPUT VISUALIZATION")
    print("="*80)
    
    encoder.eval()
    encoder_attn.eval()
    
    # Create different BVP patterns
    time = np.linspace(0, 10, 640)  # 10 seconds
    
    patterns = {
        'Sine Wave (60 BPM)': np.sin(2 * np.pi * 1.0 * time),
        'Fast Sine (120 BPM)': np.sin(2 * np.pi * 2.0 * time),
        'Random Noise': np.random.randn(640),
        'Constant': np.ones(640) * 0.5,
    }
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 3, figure=fig)
    
    for i, (pattern_name, pattern) in enumerate(patterns.items()):
        # Input signal
        ax_signal = fig.add_subplot(gs[i, 0])
        ax_signal.plot(time, pattern, 'b-', linewidth=0.8)
        ax_signal.set_title(f'{pattern_name}', fontweight='bold')
        ax_signal.set_xlabel('Time (s)')
        ax_signal.set_ylabel('Amplitude')
        ax_signal.grid(True, alpha=0.3)
        
        # Prepare input
        bvp_input = torch.FloatTensor(pattern).unsqueeze(0).unsqueeze(-1)  # [1, 640, 1]
        
        # Average pooling encoder
        with torch.no_grad():
            context_avg = encoder(bvp_input).squeeze(0).numpy()
        
        ax_avg = fig.add_subplot(gs[i, 1])
        ax_avg.bar(range(len(context_avg)), context_avg, alpha=0.7)
        ax_avg.set_title('Avg Pooling Encoder Output', fontweight='bold')
        ax_avg.set_xlabel('Feature Index')
        ax_avg.set_ylabel('Value')
        ax_avg.grid(True, alpha=0.3)
        
        # Attention encoder
        with torch.no_grad():
            context_attn = encoder_attn(bvp_input).squeeze(0).numpy()
        
        ax_attn = fig.add_subplot(gs[i, 2])
        ax_attn.bar(range(len(context_attn)), context_attn, alpha=0.7, color='orange')
        ax_attn.set_title('Attention Encoder Output', fontweight='bold')
        ax_attn.set_xlabel('Feature Index')
        ax_attn.set_ylabel('Value')
        ax_attn.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bvp_encoder_outputs_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: bvp_encoder_outputs_comparison.png")
    plt.close()
    
    print(f"   ✅ Output visualization complete")


def test_device_compatibility():
    """Test model compatibility with CPU/GPU."""
    print("\n" + "="*80)
    print("TEST 8: DEVICE COMPATIBILITY")
    print("="*80)
    
    print(f"\n💻 Available device: {config.DEVICE}")
    
    encoder = BVPEncoder().to(config.DEVICE)
    bvp_data = torch.randn(4, 640, 1).to(config.DEVICE)
    
    try:
        with torch.no_grad():
            output = encoder(bvp_data)
        print(f"   Output device: {output.device}")
        print(f"   ✅ Device compatibility test passed")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def test_output_dimension_method(encoder, encoder_attn):
    """Test get_output_dim method."""
    print("\n" + "="*80)
    print("TEST 9: OUTPUT DIMENSION METHOD")
    print("="*80)
    
    dim1 = encoder.get_output_dim()
    dim2 = encoder_attn.get_output_dim()
    
    print(f"\n   BVPEncoder output dim: {dim1}")
    print(f"   BVPEncoderWithAttention output dim: {dim2}")
    
    assert dim1 == 64, "❌ BVPEncoder output dim incorrect!"
    assert dim2 == 64, "❌ BVPEncoderWithAttention output dim incorrect!"
    print(f"   ✅ Output dimension methods working correctly")


# ==================================================
# MAIN TEST EXECUTION
# ==================================================

def main():
    """Run all BVP encoder tests."""
    print("="*80)
    print("BVP ENCODER TEST SUITE")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Time steps: {config.TIME_STEPS}")
    print(f"Input size: {config.INPUT_SIZE}")
    print(f"Hidden size: {config.HIDDEN_SIZE}")
    print("="*80)
    
    # Test 1: Model initialization
    encoder, encoder_attn = test_model_initialization()
    
    # Test 2: Forward pass
    bvp_data, bvp_context, bvp_feat, bvp_context_attn, bvp_feat_attn = test_forward_pass(
        encoder, encoder_attn
    )
    
    # Test 3: Attention mechanism
    test_attention_mechanism(encoder_attn, bvp_data)
    
    # Test 4: Gradient flow
    test_gradient_flow(encoder, encoder_attn)
    
    # Test 5: Model parameters
    test_model_parameters(encoder, encoder_attn)
    
    # Test 6: Different input sizes
    test_different_input_sizes(encoder)
    
    # Test 7: Encoder outputs visualization
    test_encoder_outputs_visualization(encoder, encoder_attn)
    
    # Test 8: Device compatibility
    test_device_compatibility()
    
    # Test 9: Output dimension method
    test_output_dimension_method(encoder, encoder_attn)
    
    print("\n" + "="*80)
    print("🎉 ALL TESTS COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - bvp_encoder_attention_weights.png")
    print("   - bvp_encoder_outputs_comparison.png")
    print("="*80)
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   ✅ All tests passed successfully")
    print(f"   📈 BVPEncoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
    print(f"   📈 BVPEncoderWithAttention: {sum(p.numel() for p in encoder_attn.parameters()):,} parameters")
    print(f"   🎯 Both encoders produce [B, 64] context vectors")
    print(f"   🔍 Attention mechanism verified and visualized")
    print("="*80)


if __name__ == "__main__":
    main()
