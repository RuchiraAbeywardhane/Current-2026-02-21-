"""
BVP Encoder Module
==================

This module contains the BVP (Blood Volume Pulse) encoder for extracting
physiological features from BVP signals using a BiLSTM architecture.

BVP is a slower physiological signal compared to EEG, so we use a smaller
network architecture for efficiency.

Author: Final Year Project
Date: 2026
"""

import torch
import torch.nn as nn


class BVPEncoder(nn.Module):
    """
    BVP Encoder using Bidirectional LSTM.
    
    This encoder processes BVP signals to extract global physiological state features.
    Uses a smaller network compared to EEG since BVP is a slower physiological signal.
    
    Architecture:
    - Conv1d layer for local feature extraction
    - BatchNorm1d and ReLU activation
    - BiLSTM with hidden_size=32 (64 total with bidirectional)
    - Global average pooling for temporal compression
    - Output: global physiological context vector
    
    Args:
        input_size (int): Input feature dimension (default: 1 for raw BVP)
        hidden_size (int): LSTM hidden size (default: 32)
        dropout (float): Dropout rate (default: 0.3)
        
    Input Shape:
        - x: [B, T, 1] where B=batch, T=time_steps
        
    Output Shape:
        - bvp_context: [B, 64] - global physiological state vector
        - bvp_feat: [B, T, 64] - temporal features (optional)
    """
    
    def __init__(self, input_size=1, hidden_size=32, dropout=0.3):
        super(BVPEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2  # Bidirectional
        
        # Conv1d layer for local feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.batch_norm = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        # Bidirectional LSTM for BVP encoding
        self.bvp_lstm = nn.LSTM(
            input_size=32,  # Changed from input_size to 32 (conv output channels)
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0  # No dropout for single layer
        )
        
        # Optional: Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Optional: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.output_size)
        
    def forward(self, x, return_sequence=False):
        """
        Forward pass through BVP encoder.
        
        Args:
            x (torch.Tensor): Input BVP signal [B, T, input_size]
            return_sequence (bool): If True, return full sequence features
                                   If False, return only global context
        
        Returns:
            If return_sequence=False:
                bvp_context: [B, 64] - global physiological state
            If return_sequence=True:
                (bvp_context, bvp_feat): tuple of global and temporal features
        """
        # Conv1d expects [B, C, T], so transpose from [B, T, C]
        x = x.transpose(1, 2)  # [B, T, input_size] -> [B, input_size, T]
        
        # Conv1d + BatchNorm + ReLU
        x = self.conv1d(x)  # [B, 32, T]
        x = self.batch_norm(x)  # [B, 32, T]
        x = self.relu(x)  # [B, 32, T]
        
        # Transpose back for LSTM [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)  # [B, T, 32]
        
        # LSTM encoding
        # bvp_feat: [B, T, 64]
        bvp_feat, (hidden, cell) = self.bvp_lstm(x)
        
        # Apply layer normalization
        bvp_feat = self.layer_norm(bvp_feat)
        
        # Apply dropout
        bvp_feat = self.dropout(bvp_feat)
        
        # Compress to global physiological state
        # Global average pooling over time dimension
        bvp_context = bvp_feat.mean(dim=1)  # [B, 64]
        
        if return_sequence:
            return bvp_context, bvp_feat
        else:
            return bvp_context
    
    def get_output_dim(self):
        """Return the output dimension of the encoder."""
        return self.output_size


class BVPEncoderWithAttention(nn.Module):
    """
    BVP Encoder with Attention mechanism.
    
    Enhanced version with attention pooling instead of average pooling
    for better temporal feature aggregation.
    
    Args:
        input_size (int): Input feature dimension (default: 1)
        hidden_size (int): LSTM hidden size (default: 32)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_size=1, hidden_size=32, dropout=0.3):
        super(BVPEncoderWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2  # Bidirectional
        
        # Bidirectional LSTM
        self.bvp_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.output_size, self.output_size // 2),
            nn.Tanh(),
            nn.Linear(self.output_size // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_size)
        
    def forward(self, x, return_sequence=False):
        """
        Forward pass with attention pooling.
        
        Args:
            x (torch.Tensor): Input BVP signal [B, T, input_size]
            return_sequence (bool): Whether to return sequence features
        
        Returns:
            bvp_context: [B, 64] - attention-weighted global state
            (optional) bvp_feat: [B, T, 64] - temporal features
        """
        # LSTM encoding
        bvp_feat, _ = self.bvp_lstm(x)  # [B, T, 64]
        bvp_feat = self.layer_norm(bvp_feat)
        bvp_feat = self.dropout(bvp_feat)
        
        # Attention pooling
        attn_scores = self.attention(bvp_feat)  # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
        
        # Weighted sum
        bvp_context = (bvp_feat * attn_weights).sum(dim=1)  # [B, 64]
        
        if return_sequence:
            return bvp_context, bvp_feat
        else:
            return bvp_context
    
    def get_output_dim(self):
        """Return the output dimension of the encoder."""
        return self.output_size


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("BVP ENCODER TEST")
    print("=" * 80)
    
    # Test basic encoder
    batch_size = 16
    time_steps = 100
    input_size = 1
    
    # Create dummy BVP data
    bvp_data = torch.randn(batch_size, time_steps, input_size)
    
    print(f"\nInput shape: {bvp_data.shape}")
    
    # Test BVPEncoder
    print("\n1. Testing BVPEncoder (Average Pooling):")
    encoder = BVPEncoder(input_size=1, hidden_size=32, dropout=0.3)
    bvp_context = encoder(bvp_data)
    print(f"   Output context shape: {bvp_context.shape}")
    print(f"   Expected: [{batch_size}, 64]")
    
    # Test with sequence return
    bvp_context, bvp_feat = encoder(bvp_data, return_sequence=True)
    print(f"   Output feature shape: {bvp_feat.shape}")
    print(f"   Expected: [{batch_size}, {time_steps}, 64]")
    
    # Test BVPEncoderWithAttention
    print("\n2. Testing BVPEncoderWithAttention:")
    encoder_attn = BVPEncoderWithAttention(input_size=1, hidden_size=32, dropout=0.3)
    bvp_context_attn = encoder_attn(bvp_data)
    print(f"   Output context shape: {bvp_context_attn.shape}")
    print(f"   Expected: [{batch_size}, 64]")
    
    # Test output dimension method
    print(f"\n3. Output dimension: {encoder.get_output_dim()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n4. Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    print("\n" + "=" * 80)
    print("✅ BVP ENCODER TEST COMPLETE!")
    print("=" * 80)
