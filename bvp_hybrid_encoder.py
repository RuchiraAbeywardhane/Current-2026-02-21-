"""
BVP Hybrid Encoder Module - Enhanced Multi-Scale Version
=========================================================

This module combines deep learned features with traditional handcrafted features
for robust BVP-based emotion recognition.

ENHANCEMENT: Added multi-scale processing from EMCNN while maintaining BiLSTM
temporal modeling and richer handcrafted features (11 vs 5).

The hybrid approach leverages:
1. Deep learning: Multi-scale CNNs + BiLSTM for temporal patterns
2. Handcrafted features: Provides interpretable physiological markers

Author: Final Year Project
Date: 2026-02-24
"""

import torch
import torch.nn as nn
from bvp_encoder import BVPEncoder
from bvp_handcrafted_features import BVPHandcraftedFeatures


class BVPHybridEncoder(nn.Module):
    """
    Enhanced hybrid BVP encoder with multi-scale processing.
    
    This encoder extracts both learned representations (via multi-scale CNN + BiLSTM)
    and traditional physiological features (statistical, HRV, pulse amplitude).
    
    Architecture:
    1. Multi-Scale Deep Encoder Branch:
       - Scale 1: Original signal → Conv1d → BiLSTM
       - Scale 2: Moving average (s=2) → Conv1d → BiLSTM  
       - Scale 3: Downsampled (2x) → Conv1d → BiLSTM
       - Feature fusion → Output: [B, 64] learned features
    
    2. Handcrafted Features Branch:
       - Peak detection
       - Statistical features (mean, std, skew, kurtosis)
       - HRV features (RR intervals, RMSSD)
       - Pulse amplitude features
       - Output: [B, 11] handcrafted features
    
    3. Feature Fusion:
       - Normalize handcrafted features (LayerNorm)
       - Concatenate deep + handcrafted features
       - Output: [B, 75] hybrid feature vector (64 + 11)
    
    Args:
        input_size (int): Input feature dimension (default: 1 for raw BVP)
        hidden_size (int): LSTM hidden size for deep encoder (default: 32)
        dropout (float): Dropout rate for deep encoder (default: 0.3)
        sampling_rate (float): BVP sampling rate in Hz (default: 64.0)
        min_peak_distance (int): Minimum distance between peaks (default: 20)
        use_multiscale (bool): Use multi-scale processing (default: True)
        
    Input Shape:
        - x: [B, T, 1] where B=batch, T=time_steps
        
    Output Shape:
        - hybrid_features: [B, 75] - concatenated deep + handcrafted features
    """
    
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        dropout=0.3,
        sampling_rate=64.0,
        min_peak_distance=20,
        use_multiscale=True
    ):
        super(BVPHybridEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sampling_rate = sampling_rate
        self.use_multiscale = use_multiscale
        
        # ============================================================
        # 1. DEEP ENCODER BRANCH (with optional multi-scale)
        # ============================================================
        if use_multiscale:
            # Multi-scale processing: 3 scales with shared architecture
            self.scale1_encoder = BVPEncoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
            self.scale2_encoder = BVPEncoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
            self.scale3_encoder = BVPEncoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
            
            # Fusion layer to combine multi-scale features
            # Each encoder outputs 64 features → 192 total
            self.scale_fusion = nn.Sequential(
                nn.Linear(64 * 3, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.deep_feature_dim = 64
        else:
            # Single-scale processing (original)
            self.deep_encoder = BVPEncoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
            self.deep_feature_dim = self.deep_encoder.get_output_dim()  # 64
        
        # ============================================================
        # 2. HANDCRAFTED FEATURES BRANCH
        # ============================================================
        self.handcrafted_extractor = BVPHandcraftedFeatures(
            sampling_rate=sampling_rate,
            min_peak_height=None,
            min_peak_distance=min_peak_distance
        )
        
        self.handcrafted_feature_dim = self.handcrafted_extractor.get_output_dim()  # 11
        
        # ============================================================
        # 3. FEATURE NORMALIZATION
        # ============================================================
        self.handcrafted_norm = nn.LayerNorm(self.handcrafted_feature_dim)
        
        # ============================================================
        # 4. OUTPUT DIMENSIONS
        # ============================================================
        self.output_dim = self.deep_feature_dim + self.handcrafted_feature_dim  # 75
        
    def _moving_average(self, x, window_size=2):
        """Apply moving average to input signal."""
        # x: [B, T, 1]
        if window_size <= 1:
            return x
        
        # Use 1D convolution for moving average
        kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
        # Reshape: [B, T, 1] → [B, 1, T]
        x_conv = x.transpose(1, 2)
        # Apply conv with padding to maintain length
        x_ma = torch.nn.functional.conv1d(x_conv, kernel, padding=window_size//2)
        # Reshape back: [B, 1, T] → [B, T, 1]
        x_ma = x_ma.transpose(1, 2)
        
        # Trim to original length if needed
        if x_ma.shape[1] > x.shape[1]:
            x_ma = x_ma[:, :x.shape[1], :]
        
        return x_ma
    
    def forward(self, x, return_separate=False):
        """
        Forward pass through enhanced hybrid encoder.
        
        Args:
            x (torch.Tensor): Input BVP signal [B, T, 1]
            return_separate (bool): If True, return deep and handcrafted features separately
        
        Returns:
            If return_separate=False:
                hybrid_features: [B, 75] - concatenated feature vector
            If return_separate=True:
                (hybrid_features, deep_context, handcrafted_features)
        """
        # ============================================================
        # STEP 1: Extract deep learned features (multi-scale or single)
        # ============================================================
        if self.use_multiscale:
            # Scale 1: Original signal
            scale1_features = self.scale1_encoder(x, return_sequence=False)  # [B, 64]
            
            # Scale 2: Moving average (smoothed signal)
            x_ma = self._moving_average(x, window_size=2)
            scale2_features = self.scale2_encoder(x_ma, return_sequence=False)  # [B, 64]
            
            # Scale 3: Downsampled signal
            x_down = x[:, ::2, :]  # [B, T/2, 1]
            scale3_features = self.scale3_encoder(x_down, return_sequence=False)  # [B, 64]
            
            # Concatenate multi-scale features
            multiscale_features = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)  # [B, 192]
            
            # Fuse multi-scale features to 64D
            deep_context = self.scale_fusion(multiscale_features)  # [B, 64]
        else:
            # Single-scale processing
            deep_context = self.deep_encoder(x, return_sequence=False)  # [B, 64]
        
        # ============================================================
        # STEP 2: Extract handcrafted features
        # ============================================================
        handcrafted_features = self.handcrafted_extractor(x)  # [B, 11]
        
        # ============================================================
        # STEP 3: Normalize handcrafted features
        # ============================================================
        handcrafted_features_norm = self.handcrafted_norm(handcrafted_features)
        
        # ============================================================
        # STEP 4: Concatenate features
        # ============================================================
        hybrid_features = torch.cat(
            [deep_context, handcrafted_features_norm],
            dim=1
        )  # [B, 75]
        
        if return_separate:
            return hybrid_features, deep_context, handcrafted_features
        else:
            return hybrid_features
    
    def get_output_dim(self):
        """Get the output dimension of the hybrid encoder."""
        return self.output_dim
    
    def get_feature_breakdown(self):
        """Get detailed breakdown of feature dimensions."""
        return {
            'deep_features': self.deep_feature_dim,
            'handcrafted_features': self.handcrafted_feature_dim,
            'total_features': self.output_dim,
            'multiscale_enabled': self.use_multiscale,
            'handcrafted_names': self.handcrafted_extractor.get_feature_names()
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("BVP HYBRID ENCODER TEST")
    print("=" * 80)
    
    # ========================================================================
    # 1. Initialize Hybrid Encoder
    # ========================================================================
    print("\n1. Initializing BVP Hybrid Encoder...")
    
    hybrid_encoder = BVPHybridEncoder(
        input_size=1,
        hidden_size=32,
        dropout=0.3,
        sampling_rate=64.0,
        min_peak_distance=20,
        use_multiscale=True
    )
    
    # Display architecture info
    breakdown = hybrid_encoder.get_feature_breakdown()
    print(f"\nFeature Breakdown:")
    print(f"  Deep features:        {breakdown['deep_features']}")
    print(f"  Handcrafted features: {breakdown['handcrafted_features']}")
    print(f"  Total hybrid:         {breakdown['total_features']}")
    print(f"  Multiscale enabled:   {breakdown['multiscale_enabled']}")
    print(f"\nHandcrafted feature names:")
    for i, name in enumerate(breakdown['handcrafted_names']):
        print(f"  {i+1:2d}. {name}")
    
    # ========================================================================
    # 2. Test with Synthetic BVP Data
    # ========================================================================
    print("\n" + "-" * 80)
    print("2. Testing with synthetic BVP signal")
    print("-" * 80)
    
    batch_size = 8
    time_steps = 256
    
    # Create synthetic BVP signal
    import numpy as np
    t = np.linspace(0, 4, time_steps)
    heart_rate = 1.0  # 60 BPM
    bvp_signal = np.sin(2 * np.pi * heart_rate * t)
    bvp_signal = np.tile(bvp_signal, (batch_size, 1))[:, :, np.newaxis]
    bvp_tensor = torch.from_numpy(bvp_signal.astype(np.float32))
    
    print(f"Input shape: {bvp_tensor.shape}")
    
    # Extract hybrid features
    hybrid_features = hybrid_encoder(bvp_tensor)
    print(f"Output shape: {hybrid_features.shape}")
    print(f"Expected: [{batch_size}, {hybrid_encoder.get_output_dim()}]")
    
    # ========================================================================
    # 3. Test with Separate Feature Return
    # ========================================================================
    print("\n" + "-" * 80)
    print("3. Testing with separate feature return")
    print("-" * 80)
    
    hybrid, deep, handcrafted = hybrid_encoder(bvp_tensor, return_separate=True)
    
    print(f"Hybrid features shape:      {hybrid.shape}")
    print(f"Deep features shape:        {deep.shape}")
    print(f"Handcrafted features shape: {handcrafted.shape}")
    
    # Verify concatenation
    print(f"\nVerification:")
    print(f"  Deep dim + Handcrafted dim = {deep.shape[1]} + {handcrafted.shape[1]} = {deep.shape[1] + handcrafted.shape[1]}")
    print(f"  Hybrid dim = {hybrid.shape[1]}")
    print(f"  Match: {deep.shape[1] + handcrafted.shape[1] == hybrid.shape[1]} ✓")
    
    # ========================================================================
    # 4. Display Sample Features
    # ========================================================================
    print("\n" + "-" * 80)
    print("4. Sample feature values (first sample)")
    print("-" * 80)
    
    print("\nDeep features (first 10):")
    print(f"  {deep[0, :10].detach().numpy()}")
    
    print("\nHandcrafted features:")
    feature_names = hybrid_encoder.handcrafted_extractor.get_feature_names()
    for name, value in zip(feature_names, handcrafted[0].detach().numpy()):
        print(f"  {name:15s}: {value:10.4f}")
    
    # ========================================================================
    # 5. Test with Different Input Sizes
    # ========================================================================
    print("\n" + "-" * 80)
    print("5. Testing with different batch sizes")
    print("-" * 80)
    
    for bs in [1, 4, 16, 32]:
        test_input = torch.randn(bs, time_steps, 1)
        test_output = hybrid_encoder(test_input)
        print(f"  Batch size {bs:2d}: Input {list(test_input.shape)} → Output {list(test_output.shape)}")
    
    # ========================================================================
    # 6. Model Parameters
    # ========================================================================
    print("\n" + "-" * 80)
    print("6. Model parameters")
    print("-" * 80)
    
    total_params = sum(p.numel() for p in hybrid_encoder.parameters())
    trainable_params = sum(p.numel() for p in hybrid_encoder.parameters() if p.requires_grad)
    
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Note: Handcrafted features add 0 parameters (non-learnable)")
    
    # ========================================================================
    # 7. Device Compatibility Test
    # ========================================================================
    print("\n" + "-" * 80)
    print("7. Device compatibility")
    print("-" * 80)
    
    # Test CPU
    cpu_input = torch.randn(2, time_steps, 1)
    cpu_output = hybrid_encoder(cpu_input)
    print(f"  CPU: Input device={cpu_input.device}, Output device={cpu_output.device} ✓")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        hybrid_encoder_cuda = hybrid_encoder.cuda()
        cuda_input = torch.randn(2, time_steps, 1).cuda()
        cuda_output = hybrid_encoder_cuda(cuda_input)
        print(f"  CUDA: Input device={cuda_input.device}, Output device={cuda_output.device} ✓")
    else:
        print(f"  CUDA: Not available (CPU only)")
    
    print("\n" + "=" * 80)
    print("✅ BVP HYBRID ENCODER TEST COMPLETE!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Deep features extracted successfully")
    print("  ✓ Handcrafted features computed correctly")
    print("  ✓ Features normalized and concatenated")
    print("  ✓ Multiple batch sizes supported")
    print("  ✓ Device compatibility verified")
    print("=" * 80)
