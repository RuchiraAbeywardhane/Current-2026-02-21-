"""
Example: BVP Hybrid Encoder for Emotion Classification
========================================================

This script demonstrates how to use the BVPHybridEncoder for emotion recognition.

The hybrid encoder combines:
1. Deep learned features (Conv1d + BiLSTM): 64 dimensions
2. Handcrafted features (statistical + HRV + pulse): 11 dimensions
3. Total: 75-dimensional hybrid feature vector

Author: Final Year Project
Date: 2026-02-24
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from bvp_hybrid_encoder import BVPHybridEncoder
from classification_head import get_classification_head


# ==================================================
# COMPLETE HYBRID MODEL
# ==================================================

class BVPHybridModel(nn.Module):
    """
    Complete BVP emotion recognition model using hybrid features.
    
    Architecture:
    - BVP Hybrid Encoder (deep + handcrafted features) → [B, 75]
    - Classification Head → [B, num_classes]
    
    This model leverages both:
    - Learned patterns from deep learning
    - Interpretable physiological markers
    """
    
    def __init__(self, num_classes=4, hidden_size=32, dropout=0.3, 
                 sampling_rate=64.0, head_type='simple'):
        super(BVPHybridModel, self).__init__()
        
        # Hybrid encoder: combines deep + handcrafted features
        self.encoder = BVPHybridEncoder(
            input_size=1,
            hidden_size=hidden_size,
            dropout=dropout,
            sampling_rate=sampling_rate,
            min_peak_distance=20
        )
        
        # Get encoder output dimension (75 = 64 deep + 11 handcrafted)
        encoder_dim = self.encoder.get_output_dim()
        
        # Classification head
        self.classifier = get_classification_head(
            head_type=head_type,
            input_dim=encoder_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        print(f"✅ BVP Hybrid Model initialized:")
        print(f"   Encoder output: {encoder_dim} features")
        print(f"   Classification head: {head_type}")
        print(f"   Number of classes: {num_classes}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: BVP signal [B, T, 1]
        
        Returns:
            logits: Class predictions [B, num_classes]
        """
        # Extract hybrid features
        hybrid_features = self.encoder(x)  # [B, 75]
        
        # Classify
        logits = self.classifier(hybrid_features)  # [B, num_classes]
        
        return logits
    
    def get_features(self, x, return_separate=False):
        """
        Get features without classification.
        
        Args:
            x: BVP signal [B, T, 1]
            return_separate: If True, return deep and handcrafted separately
        
        Returns:
            If return_separate=False: hybrid_features [B, 75]
            If return_separate=True: (hybrid, deep, handcrafted)
        """
        return self.encoder(x, return_separate=return_separate)


# ==================================================
# EXAMPLE USAGE
# ==================================================

def main():
    print("=" * 80)
    print("BVP HYBRID ENCODER - EMOTION CLASSIFICATION EXAMPLE")
    print("=" * 80)
    
    # ========================================================================
    # 1. Create synthetic BVP data for demonstration
    # ========================================================================
    print("\n1. Creating synthetic BVP data...")
    
    num_samples = 100
    time_steps = 256  # 4 seconds at 64 Hz
    num_classes = 4  # HVHA, HVLA, LVHA, LVLA
    
    # Generate synthetic BVP signals with some structure
    np.random.seed(42)
    bvp_data = []
    labels = []
    
    for i in range(num_samples):
        # Different patterns for different emotions
        class_id = i % num_classes
        t = np.linspace(0, 4, time_steps)
        
        if class_id == 0:  # HVHA - High arousal (faster heart rate)
            signal = np.sin(2 * np.pi * 1.3 * t) + 0.1 * np.random.randn(time_steps)
        elif class_id == 1:  # HVLA - Moderate heart rate
            signal = np.sin(2 * np.pi * 1.1 * t) + 0.1 * np.random.randn(time_steps)
        elif class_id == 2:  # LVHA - High arousal
            signal = np.sin(2 * np.pi * 1.2 * t) + 0.15 * np.random.randn(time_steps)
        else:  # LVLA - Low arousal (slower heart rate)
            signal = np.sin(2 * np.pi * 0.9 * t) + 0.1 * np.random.randn(time_steps)
        
        bvp_data.append(signal)
        labels.append(class_id)
    
    bvp_data = np.array(bvp_data)[:, :, np.newaxis]  # [100, 256, 1]
    labels = np.array(labels)
    
    print(f"   BVP data shape: {bvp_data.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Class distribution: {np.bincount(labels)}")
    
    # ========================================================================
    # 2. Create model
    # ========================================================================
    print("\n2. Creating BVP Hybrid Model...")
    
    model = BVPHybridModel(
        num_classes=4,
        hidden_size=32,
        dropout=0.3,
        sampling_rate=64.0,
        head_type='deep'  # Use deep classification head
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # 3. Test forward pass
    # ========================================================================
    print("\n3. Testing forward pass...")
    
    # Convert to PyTorch tensors
    bvp_tensor = torch.FloatTensor(bvp_data[:8])  # First 8 samples
    labels_tensor = torch.LongTensor(labels[:8])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(bvp_tensor)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"   Input shape: {bvp_tensor.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Predictions: {predictions.numpy()}")
    print(f"   True labels: {labels_tensor.numpy()}")
    
    # ========================================================================
    # 4. Extract and analyze features
    # ========================================================================
    print("\n4. Extracting hybrid features...")
    
    with torch.no_grad():
        # Get all features
        hybrid, deep, handcrafted = model.get_features(bvp_tensor[:1], return_separate=True)
    
    print(f"\n   Hybrid features shape: {hybrid.shape}")
    print(f"   Deep features shape: {deep.shape}")
    print(f"   Handcrafted features shape: {handcrafted.shape}")
    
    print(f"\n   Deep features (first 10): {deep[0, :10].numpy()}")
    
    print(f"\n   Handcrafted features:")
    feature_names = model.encoder.handcrafted_extractor.get_feature_names()
    for name, value in zip(feature_names, handcrafted[0].numpy()):
        print(f"      {name:15s}: {value:8.4f}")
    
    # ========================================================================
    # 5. Simple training example
    # ========================================================================
    print("\n5. Simple training example (5 epochs)...")
    
    # Create dataset and dataloader
    dataset = TensorDataset(torch.FloatTensor(bvp_data), torch.LongTensor(labels))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Setup training
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            # Forward pass
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        print(f"   Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
    
    # ========================================================================
    # 6. Feature breakdown analysis
    # ========================================================================
    print("\n6. Feature breakdown analysis...")
    
    breakdown = model.encoder.get_feature_breakdown()
    print(f"\n   Deep features: {breakdown['deep_features']}")
    print(f"   Handcrafted features: {breakdown['handcrafted_features']}")
    print(f"   Total hybrid features: {breakdown['total_features']}")
    
    print(f"\n   Handcrafted feature names:")
    for i, name in enumerate(breakdown['handcrafted_names'], 1):
        print(f"      {i:2d}. {name}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ EXAMPLE COMPLETE!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  ✓ Hybrid encoder combines 64 deep + 11 handcrafted features")
    print("  ✓ Deep features capture complex temporal patterns")
    print("  ✓ Handcrafted features provide interpretable HRV markers")
    print("  ✓ Model is end-to-end trainable (except handcrafted part)")
    print("  ✓ Suitable for emotion recognition with physiological signals")
    print("\nTo use with real data:")
    print("  python bvp_pipeline.py --encoder hybrid --head deep --baseline_reduction")
    print("=" * 80)


if __name__ == "__main__":
    main()
