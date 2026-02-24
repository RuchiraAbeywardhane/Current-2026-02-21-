"""
Multimodal Fusion Module for EEG + BVP Emotion Recognition
============================================================

This module provides multiple fusion strategies to combine EEG and BVP modalities:
1. Early Fusion: Concatenate raw features before classification
2. Late Fusion: Average/weighted prediction scores
3. Hybrid Fusion: Cross-modal attention + gated fusion (RECOMMENDED)

The hybrid fusion uses cross-modal attention to allow EEG and BVP features
to attend to each other, capturing inter-modal dependencies.

Author: Final Year Project
Date: 2026-02-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# CROSS-MODAL ATTENTION MECHANISMS
# ==================================================

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between EEG and BVP.
    
    Allows each modality to attend to the other modality's features,
    capturing complementary information (e.g., EEG cognitive states + BVP arousal).
    
    Args:
        d_model (int): Feature dimension for both modalities (default: 128)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        
        # EEG attends to BVP (EEG queries BVP context)
        self.attn_eeg_to_bvp = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # BVP attends to EEG (BVP queries EEG context)
        self.attn_bvp_to_eeg = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm_eeg = nn.LayerNorm(d_model)
        self.norm_bvp = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_eeg, h_bvp):
        """
        Apply cross-modal attention.
        
        Args:
            h_eeg (torch.Tensor): EEG features [batch_size, d_model]
            h_bvp (torch.Tensor): BVP features [batch_size, d_model]
        
        Returns:
            h_eeg_attended (torch.Tensor): EEG features with BVP context [batch_size, d_model]
            h_bvp_attended (torch.Tensor): BVP features with EEG context [batch_size, d_model]
        """
        # Add sequence dimension for multi-head attention: [B, 1, D]
        h_eeg_ = h_eeg.unsqueeze(1)  # [B, 1, d_model]
        h_bvp_ = h_bvp.unsqueeze(1)  # [B, 1, d_model]
        
        # EEG attends to BVP (Query: EEG, Key/Value: BVP)
        eeg_ctx, _ = self.attn_eeg_to_bvp(
            query=h_eeg_,
            key=h_bvp_,
            value=h_bvp_
        )  # [B, 1, d_model]
        
        # BVP attends to EEG (Query: BVP, Key/Value: EEG)
        bvp_ctx, _ = self.attn_bvp_to_eeg(
            query=h_bvp_,
            key=h_eeg_,
            value=h_eeg_
        )  # [B, 1, d_model]
        
        # Residual connection + layer norm
        h_eeg_attended = self.norm_eeg(h_eeg_ + self.dropout(eeg_ctx))  # [B, 1, d_model]
        h_bvp_attended = self.norm_bvp(h_bvp_ + self.dropout(bvp_ctx))  # [B, 1, d_model]
        
        # Remove sequence dimension: [B, 1, D] → [B, D]
        h_eeg_attended = h_eeg_attended.squeeze(1)
        h_bvp_attended = h_bvp_attended.squeeze(1)
        
        return h_eeg_attended, h_bvp_attended


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining EEG and BVP features.
    
    Learns adaptive weights for each modality based on their content.
    The gate determines how much to trust each modality for the current sample.
    
    Args:
        d_model (int): Feature dimension (default: 128)
    """
    
    def __init__(self, d_model=128):
        super().__init__()
        
        # Gate network: learns to weight modalities
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, h_eeg, h_bvp):
        """
        Apply gated fusion.
        
        Args:
            h_eeg (torch.Tensor): EEG features [batch_size, d_model]
            h_bvp (torch.Tensor): BVP features [batch_size, d_model]
        
        Returns:
            h_fused (torch.Tensor): Fused features [batch_size, d_model]
        """
        # Concatenate features
        h_concat = torch.cat([h_eeg, h_bvp], dim=1)  # [B, 2*d_model]
        
        # Compute gate weights
        g = self.gate(h_concat)  # [B, d_model], values in [0, 1]
        
        # Weighted fusion: g controls EEG weight, (1-g) controls BVP weight
        h_fused = g * h_eeg + (1 - g) * h_bvp  # [B, d_model]
        
        return h_fused


# ==================================================
# EEG ENCODER (Extract features from BiLSTM)
# ==================================================

class EEGEncoder(nn.Module):
    """
    EEG feature encoder extracted from pre-trained BiLSTM.
    
    Removes the classification head and uses the BiLSTM representation.
    
    Args:
        pretrained_model (nn.Module): Pre-trained SimpleBiLSTMClassifier
        freeze_weights (bool): Whether to freeze encoder weights (default: True)
    """
    
    def __init__(self, pretrained_model, freeze_weights=True):
        super().__init__()
        
        # Copy encoder components from pretrained model
        self.input_proj = pretrained_model.input_proj
        self.lstm = pretrained_model.lstm
        self.norm = pretrained_model.norm
        self.drop = pretrained_model.drop
        self.attn = pretrained_model.attn
        
        # Output dimension = 2 * hidden (bidirectional LSTM)
        self.output_dim = pretrained_model.hidden * 2  # 512 (for hidden=256)
        
        # Freeze weights if specified
        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
    
    def forward(self, x):
        """
        Extract EEG features.
        
        Args:
            x (torch.Tensor): EEG features [batch_size, n_channels, n_features]
        
        Returns:
            h_pooled (torch.Tensor): EEG representation [batch_size, 512]
        """
        # Same forward pass as SimpleBiLSTMClassifier, but without classifier head
        B, C, dx = x.shape
        x = self.input_proj(x)
        h, _ = self.lstm(x)
        h = self.drop(self.norm(h))
        
        # Attention pooling
        scores = self.attn(h)
        alpha = torch.softmax(scores, dim=1)
        h_pooled = (alpha * h).sum(dim=1)
        
        return h_pooled


# ==================================================
# MULTIMODAL FUSION MODELS
# ==================================================

class EarlyFusionModel(nn.Module):
    """
    Early fusion: Concatenate EEG and BVP features, then classify.
    
    Simplest fusion strategy - just concatenate features and add a classifier.
    
    Args:
        eeg_encoder (nn.Module): EEG feature encoder
        bvp_encoder (nn.Module): BVP feature encoder
        n_classes (int): Number of emotion classes (default: 4)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, eeg_encoder, bvp_encoder, n_classes=4, dropout=0.3):
        super().__init__()
        
        self.eeg_encoder = eeg_encoder
        self.bvp_encoder = bvp_encoder
        
        # Get encoder output dimensions
        eeg_dim = eeg_encoder.output_dim  # 512
        bvp_dim = bvp_encoder.get_output_dim()  # 75
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(eeg_dim + bvp_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, eeg_x, bvp_x):
        """
        Forward pass.
        
        Args:
            eeg_x (torch.Tensor): EEG input [batch_size, n_channels, n_features]
            bvp_x (torch.Tensor): BVP input [batch_size, time_steps, 1]
        
        Returns:
            logits (torch.Tensor): Class logits [batch_size, n_classes]
        """
        # Extract features
        h_eeg = self.eeg_encoder(eeg_x)  # [B, 512]
        h_bvp = self.bvp_encoder(bvp_x)  # [B, 75]
        
        # Concatenate
        h_fused = torch.cat([h_eeg, h_bvp], dim=1)  # [B, 587]
        
        # Classify
        logits = self.classifier(h_fused)
        
        return logits


class LateFusionModel(nn.Module):
    """
    Late fusion: Train separate classifiers, then combine predictions.
    
    Each modality makes independent predictions, then we fuse at decision level.
    
    Args:
        eeg_encoder (nn.Module): EEG feature encoder
        bvp_encoder (nn.Module): BVP feature encoder
        n_classes (int): Number of emotion classes (default: 4)
        fusion_method (str): 'average', 'weighted', or 'learned' (default: 'learned')
    """
    
    def __init__(self, eeg_encoder, bvp_encoder, n_classes=4, fusion_method='learned'):
        super().__init__()
        
        self.eeg_encoder = eeg_encoder
        self.bvp_encoder = bvp_encoder
        self.fusion_method = fusion_method
        
        eeg_dim = eeg_encoder.output_dim
        bvp_dim = bvp_encoder.get_output_dim()
        
        # Separate classifiers for each modality
        self.eeg_classifier = nn.Sequential(
            nn.Linear(eeg_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
        self.bvp_classifier = nn.Sequential(
            nn.Linear(bvp_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
        # Learned weights for fusion
        if fusion_method == 'learned':
            self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
    
    def forward(self, eeg_x, bvp_x):
        """
        Forward pass.
        
        Args:
            eeg_x (torch.Tensor): EEG input [batch_size, n_channels, n_features]
            bvp_x (torch.Tensor): BVP input [batch_size, time_steps, 1]
        
        Returns:
            logits (torch.Tensor): Fused class logits [batch_size, n_classes]
        """
        # Extract features
        h_eeg = self.eeg_encoder(eeg_x)
        h_bvp = self.bvp_encoder(bvp_x)
        
        # Get predictions from each modality
        logits_eeg = self.eeg_classifier(h_eeg)  # [B, n_classes]
        logits_bvp = self.bvp_classifier(h_bvp)  # [B, n_classes]
        
        # Fuse predictions
        if self.fusion_method == 'average':
            logits = (logits_eeg + logits_bvp) / 2
        elif self.fusion_method == 'weighted':
            # Fixed weights (EEG typically more reliable)
            logits = 0.6 * logits_eeg + 0.4 * logits_bvp
        elif self.fusion_method == 'learned':
            # Learned weights (normalized)
            w = F.softmax(self.fusion_weights, dim=0)
            logits = w[0] * logits_eeg + w[1] * logits_bvp
        
        return logits


class HybridFusionModel(nn.Module):
    """
    Hybrid fusion with cross-modal attention and gated fusion (RECOMMENDED).
    
    This is the most sophisticated fusion strategy:
    1. Project EEG and BVP to shared dimension
    2. Apply cross-modal attention (EEG ↔ BVP)
    3. Use gated fusion to combine modalities
    4. Classify fused representation
    
    Args:
        eeg_encoder (nn.Module): EEG feature encoder
        bvp_encoder (nn.Module): BVP feature encoder
        n_classes (int): Number of emotion classes (default: 4)
        shared_dim (int): Shared representation dimension (default: 128)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        eeg_encoder,
        bvp_encoder,
        n_classes=4,
        shared_dim=128,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.eeg_encoder = eeg_encoder
        self.bvp_encoder = bvp_encoder
        
        eeg_dim = eeg_encoder.output_dim  # 512
        bvp_dim = bvp_encoder.get_output_dim()  # 75
        
        # ============================================================
        # 1. PROJECTION TO SHARED SPACE
        # ============================================================
        # Project both modalities to same dimension for cross-modal attention
        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.bvp_proj = nn.Sequential(
            nn.Linear(bvp_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============================================================
        # 2. CROSS-MODAL ATTENTION
        # ============================================================
        self.cross_attention = CrossModalAttention(
            d_model=shared_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ============================================================
        # 3. GATED FUSION
        # ============================================================
        self.gated_fusion = GatedFusion(d_model=shared_dim)
        
        # ============================================================
        # 4. FINAL CLASSIFIER
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim // 2, n_classes)
        )
    
    def forward(self, eeg_x, bvp_x, return_attention_weights=False):
        """
        Forward pass through hybrid fusion model.
        
        Args:
            eeg_x (torch.Tensor): EEG input [batch_size, n_channels, n_features]
            bvp_x (torch.Tensor): BVP input [batch_size, time_steps, 1]
            return_attention_weights (bool): Return attention analysis (default: False)
        
        Returns:
            logits (torch.Tensor): Class logits [batch_size, n_classes]
            (optional) attention_info: Dict with intermediate representations
        """
        # Extract features from both modalities
        h_eeg_raw = self.eeg_encoder(eeg_x)  # [B, 512]
        h_bvp_raw = self.bvp_encoder(bvp_x)  # [B, 75]
        
        # Project to shared dimension
        h_eeg = self.eeg_proj(h_eeg_raw)  # [B, shared_dim]
        h_bvp = self.bvp_proj(h_bvp_raw)  # [B, shared_dim]
        
        # Apply cross-modal attention
        h_eeg_attended, h_bvp_attended = self.cross_attention(h_eeg, h_bvp)
        
        # Gated fusion
        h_fused = self.gated_fusion(h_eeg_attended, h_bvp_attended)  # [B, shared_dim]
        
        # Classification
        logits = self.classifier(h_fused)  # [B, n_classes]
        
        if return_attention_weights:
            attention_info = {
                'eeg_raw': h_eeg_raw,
                'bvp_raw': h_bvp_raw,
                'eeg_projected': h_eeg,
                'bvp_projected': h_bvp,
                'eeg_attended': h_eeg_attended,
                'bvp_attended': h_bvp_attended,
                'fused': h_fused
            }
            return logits, attention_info
        
        return logits


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def create_fusion_model(
    eeg_model_path,
    bvp_encoder,
    fusion_type='hybrid',
    n_classes=4,
    freeze_encoders=True,
    device='cpu'
):
    """
    Factory function to create fusion models.
    
    Args:
        eeg_model_path (str): Path to pre-trained EEG BiLSTM model
        bvp_encoder (nn.Module): BVP hybrid encoder
        fusion_type (str): 'early', 'late', or 'hybrid' (default: 'hybrid')
        n_classes (int): Number of emotion classes (default: 4)
        freeze_encoders (bool): Freeze encoder weights (default: True)
        device (str): Device to load model on
    
    Returns:
        fusion_model (nn.Module): Initialized fusion model
    """
    from eeg_bilstm_model import SimpleBiLSTMClassifier
    
    # Load pre-trained EEG model
    eeg_model = SimpleBiLSTMClassifier(
        dx=26,
        n_channels=4,
        hidden=256,
        layers=3,
        n_classes=n_classes,
        p_drop=0.4
    )
    eeg_model.load_state_dict(torch.load(eeg_model_path, map_location=device))
    
    # Create EEG encoder
    eeg_encoder = EEGEncoder(eeg_model, freeze_weights=freeze_encoders)
    
    # Optionally freeze BVP encoder
    if freeze_encoders:
        for param in bvp_encoder.parameters():
            param.requires_grad = False
        bvp_encoder.eval()
    
    # Create fusion model
    if fusion_type == 'early':
        fusion_model = EarlyFusionModel(
            eeg_encoder=eeg_encoder,
            bvp_encoder=bvp_encoder,
            n_classes=n_classes
        )
    elif fusion_type == 'late':
        fusion_model = LateFusionModel(
            eeg_encoder=eeg_encoder,
            bvp_encoder=bvp_encoder,
            n_classes=n_classes,
            fusion_method='learned'
        )
    elif fusion_type == 'hybrid':
        fusion_model = HybridFusionModel(
            eeg_encoder=eeg_encoder,
            bvp_encoder=bvp_encoder,
            n_classes=n_classes,
            shared_dim=128,
            num_heads=4,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_model.to(device)


def get_trainable_parameters(model):
    """Count trainable parameters in fusion model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
    }


# ==================================================
# TESTING
# ==================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTIMODAL FUSION MODULE TEST")
    print("=" * 80)
    
    # Create dummy encoders for testing
    from bvp_hybrid_encoder import BVPHybridEncoder
    from eeg_bilstm_model import SimpleBiLSTMClassifier
    
    # Dummy EEG model
    eeg_model = SimpleBiLSTMClassifier(dx=26, n_channels=4, hidden=256, layers=3, n_classes=4)
    eeg_encoder = EEGEncoder(eeg_model, freeze_weights=False)
    
    # Dummy BVP encoder
    bvp_encoder = BVPHybridEncoder(
        input_size=1,
        hidden_size=32,
        dropout=0.3,
        use_multiscale=False
    )
    
    print(f"\n✅ EEG Encoder output dim: {eeg_encoder.output_dim}")
    print(f"✅ BVP Encoder output dim: {bvp_encoder.get_output_dim()}")
    
    # Test all fusion types
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\n{'='*80}")
        print(f"Testing {fusion_type.upper()} Fusion")
        print(f"{'='*80}")
        
        if fusion_type == 'early':
            model = EarlyFusionModel(eeg_encoder, bvp_encoder, n_classes=4)
        elif fusion_type == 'late':
            model = LateFusionModel(eeg_encoder, bvp_encoder, n_classes=4)
        else:
            model = HybridFusionModel(eeg_encoder, bvp_encoder, n_classes=4)
        
        # Test forward pass
        batch_size = 8
        eeg_input = torch.randn(batch_size, 4, 26)  # [B, channels, features]
        bvp_input = torch.randn(batch_size, 640, 1)  # [B, time_steps, 1]
        
        output = model(eeg_input, bvp_input)
        
        print(f"\n   Input shapes:")
        print(f"      EEG: {list(eeg_input.shape)}")
        print(f"      BVP: {list(bvp_input.shape)}")
        print(f"   Output shape: {list(output.shape)}")
        print(f"   Expected: [{batch_size}, 4]")
        
        # Count parameters
        params = get_trainable_parameters(model)
        print(f"\n   Parameters:")
        print(f"      Total:     {params['total']:,}")
        print(f"      Trainable: {params['trainable']:,}")
        print(f"      Frozen:    {params['frozen']:,}")
        print(f"      Ratio:     {params['trainable_ratio']:.2%}")
    
    print("\n" + "="*80)
    print("✅ ALL FUSION MODELS TESTED SUCCESSFULLY!")
    print("="*80)
