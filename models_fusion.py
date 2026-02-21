"""
Fusion Model Architecture
==========================
Multimodal fusion of EEG and BVP for emotion recognition.

Usage:
    from models_fusion import WindowFusionEEGBVP
    from models_eeg import EEGEncoder
    from models_bvp import BVPEncoder
    
    eeg_encoder = EEGEncoder()
    bvp_encoder = BVPEncoder()
    fusion_model = WindowFusionEEGBVP(
        eeg_encoder=eeg_encoder,
        bvp_encoder=bvp_encoder,
        n_classes=4,
        shared_dim=128
    )
"""

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between EEG and BVP.
    
    Uses multi-head attention to allow each modality to attend to the other.
    """
    
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_eeg_to_bvp = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.attn_bvp_to_eeg = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm_eeg = nn.LayerNorm(d_model)
        self.norm_bvp = nn.LayerNorm(d_model)

    def forward(self, h_eeg, h_bvp):
        """
        Apply cross-modal attention.
        
        Args:
            h_eeg (torch.Tensor): EEG features (B, d_model)
            h_bvp (torch.Tensor): BVP features (B, d_model)
        
        Returns:
            tuple: (attended_eeg, attended_bvp) each of shape (B, d_model)
        """
        # Add sequence dimension for attention
        h_eeg_ = h_eeg.unsqueeze(1)  # (B, 1, d_model)
        h_bvp_ = h_bvp.unsqueeze(1)  # (B, 1, d_model)
        
        # EEG attends to BVP
        eeg_ctx, _ = self.attn_eeg_to_bvp(query=h_eeg_, key=h_bvp_, value=h_bvp_)
        
        # BVP attends to EEG
        bvp_ctx, _ = self.attn_bvp_to_eeg(query=h_bvp_, key=h_eeg_, value=h_eeg_)
        
        # Residual connection + normalization
        h_eeg_out = self.norm_eeg(h_eeg_ + eeg_ctx).squeeze(1)
        h_bvp_out = self.norm_bvp(h_bvp_ + bvp_ctx).squeeze(1)
        
        return h_eeg_out, h_bvp_out


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism.
    
    Learns a gating mechanism to dynamically weight EEG and BVP contributions.
    """
    
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )

    def forward(self, h_eeg, h_bvp):
        """
        Fuse EEG and BVP features with learned gating.
        
        Args:
            h_eeg (torch.Tensor): EEG features (B, d)
            h_bvp (torch.Tensor): BVP features (B, d)
        
        Returns:
            torch.Tensor: Fused features (B, d)
        """
        g = self.gate(torch.cat([h_eeg, h_bvp], dim=1))
        return g * h_eeg + (1 - g) * h_bvp


class WindowFusionEEGBVP(nn.Module):
    """
    Complete multimodal fusion model.
    
    Architecture:
    1. EEG encoder extracts features
    2. BVP encoder extracts features
    3. Project both to shared dimension
    4. Cross-modal attention
    5. Gated fusion
    6. Classification head
    
    Args:
        eeg_encoder: Pre-trained EEG encoder
        bvp_encoder: Pre-trained BVP encoder
        n_classes (int): Number of output classes (default: 4)
        shared_dim (int): Shared embedding dimension (default: 128)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(self, eeg_encoder, bvp_encoder, n_classes=4, shared_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.bvp_encoder = bvp_encoder
        
        # Project to shared dimension
        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_encoder.out_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU()
        )
        self.bvp_proj = nn.Sequential(
            nn.Linear(bvp_encoder.out_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU()
        )
        
        # Cross-modal attention
        self.cross_attn = CrossModalAttention(
            d_model=shared_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Gated fusion
        self.fusion = GatedFusion(shared_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, n_classes)
        )

    def forward(self, eeg_x, bvp_x):
        """
        Forward pass.
        
        Args:
            eeg_x (torch.Tensor): EEG input (B, C, dx)
            bvp_x (tuple): BVP inputs (x1, x2, x3, feats)
        
        Returns:
            torch.Tensor: Output logits (B, n_classes)
        """
        # Extract features
        H_eeg = self.eeg_encoder(eeg_x)
        H_bvp = self.bvp_encoder(*bvp_x)
        
        # Project to shared space
        H_eeg = self.eeg_proj(H_eeg)
        H_bvp = self.bvp_proj(H_bvp)
        
        # Cross-modal attention
        H_eeg, H_bvp = self.cross_attn(H_eeg, H_bvp)
        
        # Fuse modalities
        H_fused = self.fusion(H_eeg, H_bvp)
        
        # Classify
        return self.classifier(H_fused)


def load_fusion_model(checkpoint_path, eeg_encoder, bvp_encoder, device='cuda'):
    """
    Load a trained fusion model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to saved fusion model checkpoint
        eeg_encoder: EEG encoder instance
        bvp_encoder: BVP encoder instance
        device (str): Device to load model on
    
    Returns:
        WindowFusionEEGBVP: Loaded fusion model
    """
    fusion_model = WindowFusionEEGBVP(
        eeg_encoder=eeg_encoder,
        bvp_encoder=bvp_encoder
    )
    fusion_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    fusion_model.to(device)
    fusion_model.eval()
    return fusion_model
