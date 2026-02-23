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
    USE_BVP_BASELINE_CORRECTION = False
    
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


def test_classification_head():
    """Test 7: Test encoder with classification head for emotion recognition."""
    print("\n" + "="*80)
    print("TEST 7: EMOTION CLASSIFICATION")
    print("="*80)
    
    try:
        # Load real BVP data
        print("\n📂 Loading BVP data for classification test...")
        X_raw, y_labels, subject_ids, label_to_id = load_bvp_data(config.DATA_ROOT, config)
        
        # Use MORE samples for better training (500 or all available)
        n_samples = min(500, len(X_raw))  # Increased from 200
        
        # Shuffle data to avoid bias
        indices = np.random.permutation(len(X_raw))[:n_samples]
        X_subset = X_raw[indices]
        y_subset = y_labels[indices]
        
        # Convert to PyTorch tensors
        X_tensor = torch.from_numpy(X_subset).float().unsqueeze(-1)  # (N, T, 1)
        y_tensor = torch.from_numpy(y_subset).long()
        
        print(f"\n📊 Classification Dataset:")
        print(f"   Samples: {n_samples}")
        print(f"   Input shape: {X_tensor.shape}")
        print(f"   Classes: {config.NUM_CLASSES}")
        class_counts = torch.bincount(y_tensor)
        print(f"   Class distribution: {class_counts.numpy()}")
        for i in range(config.NUM_CLASSES):
            class_name = [k for k, v in label_to_id.items() if v == i][0]
            print(f"      Class {i} ({class_name}): {class_counts[i]} samples ({100*class_counts[i]/n_samples:.1f}%)")
        
        # Create encoder + classifier model with BETTER architecture
        class BVPClassifier(nn.Module):
            def __init__(self, encoder, num_classes):
                super(BVPClassifier, self).__init__()
                self.encoder = encoder
                # Deeper classifier with BatchNorm
                self.classifier = nn.Sequential(
                    nn.Linear(encoder.get_output_dim(), 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                context = self.encoder(x)
                logits = self.classifier(context)
                return logits
        
        # Initialize model with ATTENTION encoder for better performance
        encoder = BVPEncoderWithAttention(  # Changed from BVPEncoder
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            dropout=config.DROPOUT
        )
        model = BVPClassifier(encoder, config.NUM_CLASSES).to(config.DEVICE)
        
        # Setup training with CLASS WEIGHTS to handle imbalance
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Calculate class weights for imbalanced dataset
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
        class_weights = class_weights.to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"\n⚖️  Class weights: {class_weights.cpu().numpy()}")
        
        # Subject-independent train/val/test split
        subject_ids_subset = subject_ids[indices]
        
        if config.SUBJECT_INDEPENDENT:
            print(f"\n📊 Creating SUBJECT-INDEPENDENT split...")
            unique_subjects = np.unique(subject_ids_subset)
            np.random.shuffle(unique_subjects)
            
            n_subjects = len(unique_subjects)
            n_test_subj = max(1, int(n_subjects * 0.15))
            n_val_subj = max(1, int(n_subjects * 0.15))
            
            test_subjects = unique_subjects[:n_test_subj]
            val_subjects = unique_subjects[n_test_subj:n_test_subj+n_val_subj]
            train_subjects = unique_subjects[n_test_subj+n_val_subj:]
            
            # Get indices for each split
            train_mask = np.isin(subject_ids_subset, train_subjects)
            val_mask = np.isin(subject_ids_subset, val_subjects)
            test_mask = np.isin(subject_ids_subset, test_subjects)
            
            X_train = X_tensor[train_mask]
            y_train = y_tensor[train_mask]
            X_val = X_tensor[val_mask]
            y_val = y_tensor[val_mask]
            X_test = X_tensor[test_mask]
            y_test = y_tensor[test_mask]
            
            print(f"   Train subjects: {len(train_subjects)} → {sorted(train_subjects)}")
            print(f"   Val subjects:   {len(val_subjects)} → {sorted(val_subjects)}")
            print(f"   Test subjects:  {len(test_subjects)} → {sorted(test_subjects)}")
            
            # Verify no overlap
            overlap = set(train_subjects) & set(val_subjects) & set(test_subjects)
            if len(overlap) == 0:
                print(f"   ✅ Subject-independent split verified (no overlap)")
            else:
                print(f"   ⚠️  WARNING: Subject overlap detected: {overlap}")
        else:
            print(f"\n📊 Creating RANDOM split (subject-dependent)...")
            # Random split (70/15/15)
            n_train = int(0.7 * n_samples)
            n_val = int(0.15 * n_samples)
            
            X_train = X_tensor[:n_train]
            y_train = y_tensor[:n_train]
            X_val = X_tensor[n_train:n_train+n_val]
            y_val = y_tensor[n_train:n_train+n_val]
            X_test = X_tensor[n_train+n_val:]
            y_test = y_tensor[n_train+n_val:]
        
        print(f"\n📊 Split Summary:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
        # Training loop with MORE epochs and early stopping
        print(f"\n🔧 Training classifier (with early stopping)...")
        model.train()
        batch_size = 16
        n_epochs = 50  # Increased from 10
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        train_losses = []
        val_accs = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle training data each epoch
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i+batch_size].to(config.DEVICE)
                batch_y = y_train_shuffled[i:i+batch_size].to(config.DEVICE)
                
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val.to(config.DEVICE))
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = (val_preds.cpu() == y_val).float().mean().item()
                val_accs.append(val_acc)
            model.train()
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:2d}/{n_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.3f} ({val_acc*100:.1f}%) | Best: {best_val_acc:.3f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n   ⏹️  Early stopping at epoch {epoch+1} (patience reached)")
                break
        
        # Restore best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        print(f"\n📊 Final Evaluation:")
        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_logits = model(X_train.to(config.DEVICE))
            train_preds = torch.argmax(train_logits, dim=1)
            train_acc = (train_preds.cpu() == y_train).float().mean().item()
            
            # Val accuracy
            val_logits = model(X_val.to(config.DEVICE))
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds.cpu() == y_val).float().mean().item()
            
            # Test accuracy
            test_logits = model(X_test.to(config.DEVICE))
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = (test_preds.cpu() == y_test).float().mean().item()
            
            print(f"   Train Accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
            print(f"   Val Accuracy:   {val_acc:.3f} ({val_acc*100:.1f}%)")
            print(f"   Test Accuracy:  {test_acc:.3f} ({test_acc*100:.1f}%)")
            print(f"   Random Chance:  {1.0/config.NUM_CLASSES:.3f} ({100.0/config.NUM_CLASSES:.1f}%)")
        
        # Confusion matrix
        from collections import Counter
        y_test_np = y_test.numpy()
        y_pred_np = test_preds.cpu().numpy()
        
        print(f"\n📊 Per-Class Performance (Test Set):")
        for class_id in range(config.NUM_CLASSES):
            mask = y_test_np == class_id
            if mask.sum() > 0:
                class_acc = (y_pred_np[mask] == class_id).mean()
                class_name = [k for k, v in label_to_id.items() if v == class_id][0]
                print(f"   Class {class_id} ({class_name}): {class_acc:.3f} ({class_acc*100:.1f}%) [{mask.sum()} samples]")
        
        # Confusion matrix
        conf_matrix = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int)
        for true_label, pred_label in zip(y_test_np, y_pred_np):
            conf_matrix[true_label, pred_label] += 1
        
        print(f"\n📊 Confusion Matrix:")
        print(f"   Rows: True labels, Columns: Predicted labels")
        header = "       " + "  ".join([f"P{i}" for i in range(config.NUM_CLASSES)])
        print(f"   {header}")
        for i in range(config.NUM_CLASSES):
            row = f"   T{i}:  " + "  ".join([f"{conf_matrix[i, j]:3d}" for j in range(config.NUM_CLASSES)])
            print(row)
        
        # Visualize predictions
        print(f"\n📊 Sample Predictions:")
        for i in range(min(10, len(y_test))):
            true_label = y_test[i].item()
            pred_label = test_preds[i].item()
            true_name = [k for k, v in label_to_id.items() if v == true_label][0]
            pred_name = [k for k, v in label_to_id.items() if v == pred_label][0]
            status = "✅" if true_label == pred_label else "❌"
            confidence = torch.softmax(test_logits[i], dim=0)[pred_label].item()
            print(f"   {status} Sample {i+1}: True={true_name}, Pred={pred_name} (conf: {confidence:.2f})")
        
        # Analysis
        print(f"\n📈 Analysis:")
        if test_acc > (1.0/config.NUM_CLASSES + 0.05):
            print(f"   ✅ Model performs BETTER than random chance!")
            print(f"   🎯 The BVP encoder CAN classify emotions (though accuracy is modest)")
        elif test_acc > (1.0/config.NUM_CLASSES):
            print(f"   ⚠️  Model performs slightly better than random")
            print(f"   💡 BVP alone may have limited discriminative power for 4-class emotions")
        else:
            print(f"   ❌ Model performs WORSE than or equal to random chance")
            print(f"   💡 BVP alone is insufficient - consider multimodal fusion (EEG+BVP)")
        
        print(f"\n💡 Recommendations:")
        print(f"   1. Use BVP as part of MULTIMODAL fusion (EEG + BVP)")
        print(f"   2. BVP provides complementary physiological signals to EEG")
        print(f"   3. Extract handcrafted BVP features (HR, HRV) alongside raw signals")
        print(f"   4. Consider binary classification (valence/arousal) instead of 4-class")
        
        # Save model info
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        
        print(f"\n📊 Model Size:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Encoder parameters: {encoder_params:,} ({100*encoder_params/total_params:.1f}%)")
        print(f"   Classifier parameters: {classifier_params:,} ({100*classifier_params/total_params:.1f}%)")
        
        return model, test_acc
        
    except Exception as e:
        print(f"\n❌ Error during classification test: {e}")
        import traceback
        traceback.print_exc()
        return None, None


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
    
    # Test 7: Classification head
    test_classification_head()
    
    print("\n" + "="*80)
    print("🎉 ALL ENCODER TESTS COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - bvp_encoder_features.png (feature visualization)")
    print("="*80)


if __name__ == "__main__":
    main()
