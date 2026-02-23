"""
Classification Head Module
===========================

This module contains various classification head architectures that can be used
with EEG and BVP encoders for emotion recognition.

Features:
- Multiple classification head architectures
- Configurable depth and complexity
- Support for different input dimensions
- Dropout and normalization options
- Easy integration with encoder outputs

Author: Final Year Project
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassificationHead(nn.Module):
    """
    Simple classification head with single hidden layer.
    
    Architecture:
    - Linear -> BatchNorm -> ReLU -> Dropout -> Linear
    
    Args:
        input_dim (int): Input feature dimension
        num_classes (int): Number of output classes (default: 4)
        hidden_dim (int): Hidden layer dimension (default: 128)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_dim, num_classes=4, hidden_dim=128, dropout=0.3):
        super(SimpleClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features [B, input_dim]
        
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        return self.classifier(x)


class DeepClassificationHead(nn.Module):
    """
    Deep classification head with multiple hidden layers.
    
    Architecture:
    - Linear -> BN -> ReLU -> Dropout -> ... -> Linear
    
    Args:
        input_dim (int): Input feature dimension
        num_classes (int): Number of output classes (default: 4)
        hidden_dims (list): List of hidden layer dimensions (default: [256, 128])
        dropout (float): Dropout rate (default: 0.4)
    """
    
    def __init__(self, input_dim, num_classes=4, hidden_dims=[256, 128], dropout=0.4):
        super(DeepClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features [B, input_dim]
        
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        return self.classifier(x)


class ResidualClassificationHead(nn.Module):
    """
    Classification head with residual connections.
    
    Uses residual blocks for better gradient flow.
    
    Args:
        input_dim (int): Input feature dimension
        num_classes (int): Number of output classes (default: 4)
        hidden_dim (int): Hidden layer dimension (default: 256)
        num_blocks (int): Number of residual blocks (default: 2)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_dim, num_classes=4, hidden_dim=256, num_blocks=2, dropout=0.3):
        super(ResidualClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features [B, input_dim]
        
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output(x)


class ResidualBlock(nn.Module):
    """Residual block for classification head."""
    
    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward with residual connection."""
        residual = x
        out = self.block(x)
        out = out + residual  # Residual connection
        out = self.relu(out)
        out = self.dropout(out)
        return out


class AttentionClassificationHead(nn.Module):
    """
    Classification head with self-attention mechanism.
    
    Uses multi-head attention before final classification.
    
    Args:
        input_dim (int): Input feature dimension
        num_classes (int): Number of output classes (default: 4)
        hidden_dim (int): Hidden dimension (default: 256)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_dim, num_classes=4, hidden_dim=256, num_heads=4, dropout=0.3):
        super(AttentionClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input projection to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features [B, input_dim]
        
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        # Project to hidden dimension
        x = self.input_proj(x)  # [B, hidden_dim]
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # [B, 1, hidden_dim]
        
        # Residual connection and normalization
        x = self.norm(x + attn_out)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [B, hidden_dim]
        
        # Classification
        return self.classifier(x)


class MultiModalClassificationHead(nn.Module):
    """
    Classification head for multimodal fusion.
    
    Accepts multiple input features and fuses them before classification.
    
    Args:
        input_dims (list): List of input dimensions for each modality
        num_classes (int): Number of output classes (default: 4)
        fusion_dim (int): Dimension after fusion (default: 256)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_dims, num_classes=4, fusion_dim=256, dropout=0.3):
        super(MultiModalClassificationHead, self).__init__()
        
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim
        
        # Individual projections for each modality
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            for dim in input_dims
        ])
        
        # Fusion layer
        total_dim = fusion_dim * len(input_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, *features):
        """
        Forward pass.
        
        Args:
            *features: Variable number of feature tensors, one per modality
                      Each should be [B, input_dim_i]
        
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        if len(features) != len(self.input_dims):
            raise ValueError(f"Expected {len(self.input_dims)} features, got {len(features)}")
        
        # Project each modality
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Concatenate
        fused = torch.cat(projected, dim=1)
        
        # Fusion
        fused = self.fusion(fused)
        
        # Classification
        return self.classifier(fused)


class GatedClassificationHead(nn.Module):
    """
    Gated classification head for multimodal fusion.
    
    Uses gating mechanism to weight different modalities dynamically.
    
    Args:
        input_dims (list): List of input dimensions for each modality
        num_classes (int): Number of output classes (default: 4)
        hidden_dim (int): Hidden dimension (default: 256)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_dims, num_classes=4, hidden_dim=256, dropout=0.3):
        super(GatedClassificationHead, self).__init__()
        
        self.input_dims = input_dims
        self.num_classes = num_classes
        
        # Projections for each modality
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Gating network
        total_dim = hidden_dim * len(input_dims)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, len(input_dims)),
            nn.Softmax(dim=1)
        )
        
        # Classification network
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, *features):
        """
        Forward pass with gated fusion.
        
        Args:
            *features: Variable number of feature tensors
        
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        # Project each modality
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Stack for gating
        stacked = torch.stack(projected, dim=1)  # [B, num_modalities, hidden_dim]
        
        # Compute gate weights
        concat_features = torch.cat(projected, dim=1)
        gate_weights = self.gate(concat_features)  # [B, num_modalities]
        
        # Apply gating
        gate_weights = gate_weights.unsqueeze(2)  # [B, num_modalities, 1]
        fused = (stacked * gate_weights).sum(dim=1)  # [B, hidden_dim]
        
        # Classification
        return self.classifier(fused)


def get_classification_head(head_type, input_dim, num_classes=4, **kwargs):
    """
    Factory function to get classification head by type.
    
    Args:
        head_type (str): Type of classification head
            Options: 'simple', 'deep', 'residual', 'attention', 'multimodal', 'gated'
        input_dim (int or list): Input dimension(s)
        num_classes (int): Number of output classes
        **kwargs: Additional arguments for specific head types
    
    Returns:
        nn.Module: Classification head instance
    
    Example:
        >>> # Simple head
        >>> head = get_classification_head('simple', input_dim=64, num_classes=4)
        >>> 
        >>> # Deep head with custom hidden layers
        >>> head = get_classification_head('deep', input_dim=64, num_classes=4, 
        ...                                 hidden_dims=[256, 128])
        >>> 
        >>> # Multimodal head
        >>> head = get_classification_head('multimodal', input_dim=[64, 128], 
        ...                                 num_classes=4)
    """
    head_type = head_type.lower()
    
    if head_type == 'simple':
        return SimpleClassificationHead(input_dim, num_classes, **kwargs)
    
    elif head_type == 'deep':
        return DeepClassificationHead(input_dim, num_classes, **kwargs)
    
    elif head_type == 'residual':
        return ResidualClassificationHead(input_dim, num_classes, **kwargs)
    
    elif head_type == 'attention':
        return AttentionClassificationHead(input_dim, num_classes, **kwargs)
    
    elif head_type == 'multimodal':
        if not isinstance(input_dim, list):
            raise ValueError("Multimodal head requires list of input_dims")
        return MultiModalClassificationHead(input_dim, num_classes, **kwargs)
    
    elif head_type == 'gated':
        if not isinstance(input_dim, list):
            raise ValueError("Gated head requires list of input_dims")
        return GatedClassificationHead(input_dim, num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown head_type: {head_type}. "
                        f"Choose from: simple, deep, residual, attention, multimodal, gated")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("CLASSIFICATION HEAD MODULE TEST")
    print("=" * 80)
    
    batch_size = 16
    num_classes = 4
    
    # Test 1: Simple Classification Head
    print("\n1. Simple Classification Head:")
    simple_head = SimpleClassificationHead(input_dim=64, num_classes=num_classes)
    x = torch.randn(batch_size, 64)
    out = simple_head(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in simple_head.parameters()):,}")
    
    # Test 2: Deep Classification Head
    print("\n2. Deep Classification Head:")
    deep_head = DeepClassificationHead(input_dim=64, num_classes=num_classes, 
                                       hidden_dims=[256, 128, 64])
    out = deep_head(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in deep_head.parameters()):,}")
    
    # Test 3: Residual Classification Head
    print("\n3. Residual Classification Head:")
    residual_head = ResidualClassificationHead(input_dim=64, num_classes=num_classes)
    out = residual_head(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in residual_head.parameters()):,}")
    
    # Test 4: Attention Classification Head
    print("\n4. Attention Classification Head:")
    attention_head = AttentionClassificationHead(input_dim=64, num_classes=num_classes)
    out = attention_head(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in attention_head.parameters()):,}")
    
    # Test 5: Multimodal Classification Head
    print("\n5. Multimodal Classification Head:")
    multimodal_head = MultiModalClassificationHead(input_dims=[64, 128], 
                                                   num_classes=num_classes)
    x1 = torch.randn(batch_size, 64)
    x2 = torch.randn(batch_size, 128)
    out = multimodal_head(x1, x2)
    print(f"   Input: {x1.shape}, {x2.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in multimodal_head.parameters()):,}")
    
    # Test 6: Gated Classification Head
    print("\n6. Gated Classification Head:")
    gated_head = GatedClassificationHead(input_dims=[64, 128], 
                                         num_classes=num_classes)
    out = gated_head(x1, x2)
    print(f"   Input: {x1.shape}, {x2.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in gated_head.parameters()):,}")
    
    # Test 7: Factory function
    print("\n7. Factory Function Test:")
    head = get_classification_head('simple', input_dim=64, num_classes=4, hidden_dim=128)
    out = head(x)
    print(f"   Created: {type(head).__name__}")
    print(f"   Output: {out.shape}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
