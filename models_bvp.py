"""
BVP Model Architecture
======================
EMCNN (Emotion Multi-scale CNN) for BVP emotion recognition.

Usage:
    from models_bvp import EMCNN, BVPEncoder
    
    model = EMCNN(n_classes=4, feat_dim=5)
"""

import torch
import torch.nn as nn


class DWConv(nn.Module):
    """Depthwise Separable Convolution for efficient feature extraction."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


def make_branch():
    """
    Create a CNN branch for multi-scale BVP feature extraction.
    
    Returns:
        nn.Sequential: CNN branch (1 -> 16 -> 32 -> 64 -> Global Pool)
    """
    layers = []
    c = 1
    for out in [16, 32, 64]:
        layers.append(DWConv(c, out))
        c = out
    layers.append(nn.AdaptiveAvgPool1d(1))
    return nn.Sequential(*layers)


class EMCNN(nn.Module):
    """
    Hybrid EMCNN for BVP emotion recognition.
    
    Architecture:
    - 3 parallel CNN branches (different scales)
    - Handcrafted feature branch
    - Concatenation + FC classifier
    
    Args:
        n_classes (int): Number of output classes (default: 4)
        feat_dim (int): Dimension of handcrafted features (default: 5)
    """
    
    def __init__(self, n_classes=4, feat_dim=5):
        super().__init__()
        self.b1 = make_branch()  # Original signal
        self.b2 = make_branch()  # Smoothed signal (MA)
        self.b3 = make_branch()  # Downsampled signal
        
        self.feat_fc = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x1, x2, x3, feats):
        """
        Forward pass.
        
        Args:
            x1 (torch.Tensor): Original signal (B, 1, T)
            x2 (torch.Tensor): Smoothed signal (B, 1, T)
            x3 (torch.Tensor): Downsampled signal (B, 1, T//2)
            feats (torch.Tensor): Handcrafted features (B, feat_dim)
        
        Returns:
            torch.Tensor: Output logits (B, n_classes)
        """
        f1 = self.b1(x1).flatten(1)  # (B, 64)
        f2 = self.b2(x2).flatten(1)  # (B, 64)
        f3 = self.b3(x3).flatten(1)  # (B, 64)
        ff = self.feat_fc(feats)      # (B, 16)
        
        return self.fc(torch.cat([f1, f2, f3, ff], dim=1))


class BVPEncoder(nn.Module):
    """
    BVP encoder for fusion (without classifier head).
    
    This extracts features from BVP signals for multimodal fusion.
    """
    
    def __init__(self, feat_dim=5):
        super().__init__()
        self.b1 = make_branch()
        self.b2 = make_branch()
        self.b3 = make_branch()
        
        self.feat_fc = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.out_dim = 64 * 3 + 16  # Output dimension for fusion
    
    def forward(self, x1, x2, x3, feats):
        """
        Forward pass (returns features, not logits).
        
        Args:
            x1 (torch.Tensor): Original signal (B, 1, T)
            x2 (torch.Tensor): Smoothed signal (B, 1, T)
            x3 (torch.Tensor): Downsampled signal (B, 1, T//2)
            feats (torch.Tensor): Handcrafted features (B, feat_dim)
        
        Returns:
            torch.Tensor: Feature vector (B, 64*3+16)
        """
        f1 = self.b1(x1).flatten(1)
        f2 = self.b2(x2).flatten(1)
        f3 = self.b3(x3).flatten(1)
        ff = self.feat_fc(feats)
        
        return torch.cat([f1, f2, f3, ff], dim=1)


def load_bvp_classifier(checkpoint_path, device='cuda'):
    """
    Load a trained BVP classifier from checkpoint.
    
    Args:
        checkpoint_path (str): Path to saved model checkpoint
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        EMCNN: Loaded model
    """
    model = EMCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def convert_classifier_to_encoder(classifier_model):
    """
    Convert a trained BVP classifier to an encoder.
    
    Args:
        classifier_model (EMCNN): Trained classifier
    
    Returns:
        BVPEncoder: Encoder with loaded weights
    """
    encoder = BVPEncoder(feat_dim=5)
    
    # Copy weights from classifier
    state_dict = classifier_model.state_dict()
    encoder_state = {}
    
    for k, v in state_dict.items():
        if k.startswith('b1.') or k.startswith('b2.') or k.startswith('b3.') or k.startswith('feat_fc.'):
            encoder_state[k] = v
    
    encoder.load_state_dict(encoder_state, strict=False)
    
    # Freeze parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    encoder.eval()
    return encoder
