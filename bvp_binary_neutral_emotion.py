"""
BVP Binary Classification: Neutral vs Emotional States
======================================================

Binary classification to detect if a person is in a neutral state or 
experiencing emotions (happy, sad, fear) using BVP signals.

Strategy:
- Neutral (0) vs Emotional (1): Happy, Sad, Fear combined
- Undersample emotional class to balance with neutral
- Use BVP handcrafted features + simple classifier

Author: Final Year Project
Date: 2026-02-24
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Import BVP components
from bvp_config import BVPConfig
from bvp_data_loader import load_bvp_data
from bvp_handcrafted_features import BVPHandcraftedFeatures


# ==================================================
# BINARY CLASSIFIER FOR BVP FEATURES
# ==================================================

class BVPBinaryClassifier(nn.Module):
    """
    Simple MLP classifier for binary neutral vs emotional classification.
    """
    
    def __init__(self, input_dim=23, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Binary output
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] - BVP features
        
        Returns:
            logits: [batch, 1] - Binary logits
        """
        return self.network(x)


# ==================================================
# DATASET
# ==================================================

class BVPBinaryDataset(Dataset):
    """Dataset for binary BVP classification."""
    
    def __init__(self, bvp_X, labels, sampling_rate=64.0):
        self.bvp_X = bvp_X
        self.labels = labels
        
        # Initialize BVP feature extractor
        self.bvp_feature_extractor = BVPHandcraftedFeatures(
            sampling_rate=sampling_rate,
            min_peak_distance=20
        )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        bvp_signal = torch.from_numpy(self.bvp_X[idx]).unsqueeze(-1).float()
        
        # Extract BVP features
        with torch.no_grad():
            bvp_features = self.bvp_feature_extractor(bvp_signal.unsqueeze(0))
            bvp_features = bvp_features.squeeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return bvp_features, label


# ==================================================
# DATA PREPARATION
# ==================================================

def prepare_binary_data(bvp_X, bvp_y, label_map, undersample=True, random_state=42):
    """
    Convert multi-class to binary (neutral vs emotional) and balance classes.
    
    Args:
        bvp_X: Raw BVP signals
        bvp_y: Original labels
        label_map: Dictionary mapping label names to indices
        undersample: Whether to undersample emotional class
        random_state: Random seed
    
    Returns:
        bvp_X_binary, bvp_y_binary: Prepared binary data
    """
    print("\n" + "="*80)
    print("PREPARING BINARY DATA (NEUTRAL vs EMOTIONAL)")
    print("="*80)
    
    # Debug: Print label_map
    print(f"   Label map: {label_map}")
    
    # Find neutral label index - try multiple approaches
    neutral_idx = None
    
    # Approach 1: Check for 'neutral' in lowercase keys
    for label_name, idx in label_map.items():
        if 'neutral' in str(label_name).lower():
            neutral_idx = idx
            print(f"   Found neutral label: '{label_name}' -> {neutral_idx}")
            break
    
    # Approach 2: If not found, check if label_map values are strings
    if neutral_idx is None:
        for idx, label_name in label_map.items():
            if 'neutral' in str(label_name).lower():
                neutral_idx = idx
                print(f"   Found neutral label: {idx} -> '{label_name}'")
                break
    
    # Approach 3: Check for label index 0 as neutral (common convention)
    if neutral_idx is None:
        print(f"   ⚠️  'Neutral' not found in label names. Checking label indices...")
        print(f"   Unique labels in data: {np.unique(bvp_y)}")
        
        # Assume the first/lowest index might be neutral
        if 0 in label_map.values() or 0 in label_map.keys():
            neutral_idx = 0
            print(f"   Assuming index 0 is neutral (common convention)")
        elif len(label_map) > 0:
            # Use the first label as neutral
            neutral_idx = list(label_map.values())[0] if isinstance(list(label_map.keys())[0], str) else list(label_map.keys())[0]
            print(f"   Using first label as neutral: {neutral_idx}")
    
    if neutral_idx is None:
        raise ValueError(
            f"Cannot identify neutral label in label_map: {label_map}\n"
            f"Please ensure label_map contains a label with 'neutral' in its name, "
            f"or modify the script to specify the correct neutral label index."
        )
    
    print(f"   Using neutral label index: {neutral_idx}")
    
    # Create binary labels: 0=neutral, 1=emotional
    binary_labels = np.where(bvp_y == neutral_idx, 0, 1)
    
    # Count samples
    n_neutral = (binary_labels == 0).sum()
    n_emotional = (binary_labels == 1).sum()
    
    print(f"\n📊 Original class distribution:")
    print(f"   Neutral (0):   {n_neutral:4d} samples")
    print(f"   Emotional (1): {n_emotional:4d} samples")
    print(f"   Total:         {len(binary_labels):4d} samples")
    print(f"   Imbalance ratio: {max(n_neutral, n_emotional) / min(n_neutral, n_emotional):.2f}:1")
    
    # Undersample if needed
    if undersample and n_neutral != n_emotional:
        print(f"\n🔄 Undersampling emotional class to balance with neutral...")
        
        # Reshape for undersampling
        X_indices = np.arange(len(bvp_X)).reshape(-1, 1)
        
        undersampler = RandomUnderSampler(random_state=random_state)
        X_indices_resampled, binary_labels_resampled = undersampler.fit_resample(
            X_indices, binary_labels
        )
        
        # Apply resampling
        indices = X_indices_resampled.flatten()
        bvp_X_binary = bvp_X[indices]
        bvp_y_binary = binary_labels_resampled
        
        n_neutral_new = (bvp_y_binary == 0).sum()
        n_emotional_new = (bvp_y_binary == 1).sum()
        
        print(f"\n📊 Balanced class distribution:")
        print(f"   Neutral (0):   {n_neutral_new:4d} samples")
        print(f"   Emotional (1): {n_emotional_new:4d} samples")
        print(f"   Total:         {len(bvp_y_binary):4d} samples")
        print(f"   Balance ratio: 1:1 ✅")
    else:
        bvp_X_binary = bvp_X
        bvp_y_binary = binary_labels
    
    return bvp_X_binary, bvp_y_binary


# ==================================================
# TRAINING & EVALUATION
# ==================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for bvp_features, labels in loader:
        bvp_features = bvp_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(bvp_features).squeeze()
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    return avg_loss, accuracy, f1


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for bvp_features, labels in loader:
            bvp_features = bvp_features.to(device)
            labels = labels.to(device)
            
            # Forward
            logits = model(bvp_features).squeeze()
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    return avg_loss, accuracy, f1, np.array(all_targets), np.array(all_preds), np.array(all_probs)


# ==================================================
# MAIN PIPELINE
# ==================================================

def main(args):
    print("=" * 80)
    print("BVP BINARY CLASSIFICATION: NEUTRAL vs EMOTIONAL")
    print("=" * 80)
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📋 Configuration:")
    print(f"   Device: {device}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Undersample: {args.undersample}")
    
    # Load BVP data
    print("\n" + "="*80)
    print("LOADING BVP DATA")
    print("="*80)
    
    bvp_config = BVPConfig()
    bvp_config.USE_BVP_BASELINE_REDUCTION = False
    bvp_config.USE_BVP_BASELINE_CORRECTION = False
    
    bvp_X_raw, bvp_y, bvp_subjects, bvp_label_map = load_bvp_data(
        bvp_config.DATA_ROOT, 
        bvp_config
    )
    
    print(f"✅ Loaded {bvp_X_raw.shape[0]} BVP samples")
    print(f"   Signal length: {bvp_X_raw.shape[1]}")
    print(f"   Classes: {bvp_label_map}")
    
    # Prepare binary data
    bvp_X_binary, bvp_y_binary = prepare_binary_data(
        bvp_X_raw, 
        bvp_y, 
        bvp_label_map,
        undersample=args.undersample,
        random_state=args.seed
    )
    
    # Train/Val/Test split (70/15/15)
    print("\n" + "="*80)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("="*80)
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        bvp_X_binary, bvp_y_binary,
        test_size=0.3,
        random_state=args.seed,
        stratify=bvp_y_binary
    )
    
    # Second split: 50% val, 50% test (15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=args.seed,
        stratify=y_temp
    )
    
    print(f"   Train: {len(y_train)} samples (Neutral: {(y_train==0).sum()}, Emotional: {(y_train==1).sum()})")
    print(f"   Val:   {len(y_val)} samples (Neutral: {(y_val==0).sum()}, Emotional: {(y_val==1).sum()})")
    print(f"   Test:  {len(y_test)} samples (Neutral: {(y_test==0).sum()}, Emotional: {(y_test==1).sum()})")
    
    # Create datasets and loaders
    train_dataset = BVPBinaryDataset(X_train, y_train, sampling_rate=bvp_config.BVP_FS)
    val_dataset = BVPBinaryDataset(X_val, y_val, sampling_rate=bvp_config.BVP_FS)
    test_dataset = BVPBinaryDataset(X_test, y_test, sampling_rate=bvp_config.BVP_FS)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = BVPBinaryClassifier(
        input_dim=23,  # BVP enhanced features
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: BVPBinaryClassifier")
    print(f"   Hidden dims: {args.hidden_dims}")
    print(f"   Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Print progress
        if epoch % 5 == 0 or epoch <= 10:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            torch.save(model.state_dict(), args.checkpoint)
            print(f"   💾 Saved best model (Val F1: {val_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"   Early stopping at epoch {epoch}")
                break
    
    # Load best model for testing
    print("\n" + "="*80)
    print("TEST EVALUATION")
    print("="*80)
    
    if best_state:
        model.load_state_dict(best_state)
    
    test_loss, test_acc, test_f1, y_true, y_pred, y_probs = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📊 Test Results:")
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"   Test F1:       {test_f1:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              Neutral  Emotional")
    print(f"   Neutral    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"   Emotional  {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Neutral', 'Emotional'],
        digits=3
    ))
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 BINARY CLASSIFICATION COMPLETE!")
    print("=" * 80)
    print(f"✅ Best Val F1: {best_val_f1:.3f}")
    print(f"✅ Test Accuracy: {test_acc:.3f}")
    print(f"✅ Test F1: {test_f1:.3f}")
    print(f"✅ Model saved: {args.checkpoint}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BVP Binary Classification: Neutral vs Emotional"
    )
    
    # Data arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--undersample', action='store_true', default=True,
                        help='Undersample emotional class to balance with neutral')
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--checkpoint', type=str, 
                        default='best_bvp_binary_neutral_emotion.pt',
                        help='Output checkpoint path')
    
    args = parser.parse_args()
    main(args)
