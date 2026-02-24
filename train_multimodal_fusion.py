"""
Complete Multimodal Fusion Training Pipeline
=============================================

This script trains EEG + BVP multimodal fusion models for emotion recognition.

Pipeline:
1. Load preprocessed EEG data (with baseline reduction)
2. Load preprocessed BVP data (with baseline reduction)
3. Train individual EEG and BVP models (optional)
4. Create multimodal fusion model
5. Train fusion model with aligned EEG-BVP pairs
6. Evaluate on test set

Author: Final Year Project
Date: 2026-02-24
"""

import os
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Import configurations
from eeg_config import Config as EEGConfig
from bvp_config import BVPConfig

# Import data loaders
from eeg_data_loader import load_eeg_data, extract_eeg_features, create_data_splits as create_eeg_splits
from bvp_data_loader import load_bvp_data

# Import models
from eeg_bilstm_model import SimpleBiLSTMClassifier
from bvp_hybrid_encoder import BVPHybridEncoder
from multimodal_fusion import (
    EEGEncoder,
    EarlyFusionModel,
    LateFusionModel,
    HybridFusionModel,
    get_trainable_parameters
)


# ==================================================
# MULTIMODAL DATASET
# ==================================================

class MultimodalDataset(Dataset):
    """
    Dataset for aligned EEG and BVP samples.
    
    Ensures that each sample has both EEG and BVP data from the same
    subject and emotional state.
    """
    
    def __init__(self, eeg_X, eeg_y, eeg_subjects, bvp_X, bvp_y, bvp_subjects):
        """
        Initialize multimodal dataset.
        
        Args:
            eeg_X: EEG features [N, channels, features]
            eeg_y: EEG labels [N]
            eeg_subjects: EEG subject IDs [N]
            bvp_X: BVP signals [M, time_steps]
            bvp_y: BVP labels [M]
            bvp_subjects: BVP subject IDs [M]
        """
        # Find common subject-label pairs
        self.samples = []
        
        # Create mappings for quick lookup
        eeg_samples = {}
        for i, (subj, label) in enumerate(zip(eeg_subjects, eeg_y)):
            key = f"{subj}_{label}"
            if key not in eeg_samples:
                eeg_samples[key] = []
            eeg_samples[key].append(i)
        
        bvp_samples = {}
        for i, (subj, label) in enumerate(zip(bvp_subjects, bvp_y)):
            key = f"{subj}_{label}"
            if key not in bvp_samples:
                bvp_samples[key] = []
            bvp_samples[key].append(i)
        
        # Match EEG and BVP samples
        for key in eeg_samples:
            if key in bvp_samples:
                eeg_indices = eeg_samples[key]
                bvp_indices = bvp_samples[key]
                
                # Pair up samples (use minimum length)
                n_pairs = min(len(eeg_indices), len(bvp_indices))
                
                for i in range(n_pairs):
                    eeg_idx = eeg_indices[i]
                    bvp_idx = bvp_indices[i]
                    self.samples.append((eeg_idx, bvp_idx, eeg_y[eeg_idx]))
        
        self.eeg_X = eeg_X
        self.bvp_X = bvp_X
        
        print(f"   Multimodal dataset created: {len(self.samples)} aligned pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        eeg_idx, bvp_idx, label = self.samples[idx]
        
        eeg_sample = torch.from_numpy(self.eeg_X[eeg_idx]).float()
        bvp_sample = torch.from_numpy(self.bvp_X[bvp_idx]).unsqueeze(-1).float()  # Add channel dim
        label = torch.tensor(label, dtype=torch.long)
        
        return eeg_sample, bvp_sample, label


# ==================================================
# DATA ALIGNMENT & SPLITTING
# ==================================================

def align_eeg_bvp_data(eeg_data, bvp_data):
    """
    Align EEG and BVP datasets by subject and emotion.
    
    Returns:
        aligned_eeg: Dict with aligned EEG data
        aligned_bvp: Dict with aligned BVP data
        common_subjects: List of subjects present in both datasets
    """
    eeg_X, eeg_y, eeg_subjects, _, eeg_label_map = eeg_data
    bvp_X, bvp_y, bvp_subjects, _, bvp_label_map = bvp_data
    
    print("\n" + "="*80)
    print("ALIGNING EEG AND BVP DATASETS")
    print("="*80)
    
    # Find common subjects
    eeg_subj_set = set(eeg_subjects)
    bvp_subj_set = set(bvp_subjects)
    common_subjects = list(eeg_subj_set & bvp_subj_set)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   EEG subjects: {len(eeg_subj_set)}")
    print(f"   BVP subjects: {len(bvp_subj_set)}")
    print(f"   Common subjects: {len(common_subjects)}")
    print(f"   EEG-only: {len(eeg_subj_set - bvp_subj_set)}")
    print(f"   BVP-only: {len(bvp_subj_set - eeg_subj_set)}")
    
    # Filter to common subjects
    eeg_mask = np.isin(eeg_subjects, common_subjects)
    bvp_mask = np.isin(bvp_subjects, common_subjects)
    
    aligned_eeg = {
        'X': eeg_X[eeg_mask],
        'y': eeg_y[eeg_mask],
        'subjects': eeg_subjects[eeg_mask],
        'label_map': eeg_label_map
    }
    
    aligned_bvp = {
        'X': bvp_X[bvp_mask],
        'y': bvp_y[bvp_mask],
        'subjects': bvp_subjects[bvp_mask],
        'label_map': bvp_label_map
    }
    
    print(f"\n✅ Aligned datasets:")
    print(f"   EEG samples: {len(aligned_eeg['X'])}")
    print(f"   BVP samples: {len(aligned_bvp['X'])}")
    print(f"   EEG label dist: {Counter(aligned_eeg['y'])}")
    print(f"   BVP label dist: {Counter(aligned_bvp['y'])}")
    
    return aligned_eeg, aligned_bvp, common_subjects


def create_multimodal_splits(aligned_eeg, aligned_bvp, config, test_ratio=0.15, val_ratio=0.15):
    """
    Create subject-independent train/val/test splits for multimodal data.
    """
    print("\n" + "="*80)
    print("CREATING MULTIMODAL SPLITS")
    print("="*80)
    
    subjects = np.unique(aligned_eeg['subjects'])
    np.random.shuffle(subjects)
    
    n_test = max(1, int(len(subjects) * test_ratio))
    n_val = max(1, int(len(subjects) * val_ratio))
    
    test_subjects = subjects[:n_test]
    val_subjects = subjects[n_test:n_test+n_val]
    train_subjects = subjects[n_test+n_val:]
    
    print(f"\n📋 Split by subjects:")
    print(f"   Train: {len(train_subjects)} subjects")
    print(f"   Val:   {len(val_subjects)} subjects")
    print(f"   Test:  {len(test_subjects)} subjects")
    
    # Get sample indices for each split
    train_mask = np.isin(aligned_eeg['subjects'], train_subjects)
    val_mask = np.isin(aligned_eeg['subjects'], val_subjects)
    test_mask = np.isin(aligned_eeg['subjects'], test_subjects)
    
    splits = {
        'train': {'eeg': train_mask, 'bvp': np.isin(aligned_bvp['subjects'], train_subjects)},
        'val': {'eeg': val_mask, 'bvp': np.isin(aligned_bvp['subjects'], val_subjects)},
        'test': {'eeg': test_mask, 'bvp': np.isin(aligned_bvp['subjects'], test_subjects)}
    }
    
    print(f"\n📊 Sample distribution:")
    for split_name, masks in splits.items():
        n_eeg = masks['eeg'].sum()
        n_bvp = masks['bvp'].sum()
        print(f"   {split_name.capitalize():5s}: {n_eeg:4d} EEG, {n_bvp:4d} BVP")
    
    return splits


# ==================================================
# TRAINING FUNCTIONS
# ==================================================

def train_fusion_epoch(model, train_loader, criterion, optimizer, device):
    """Train fusion model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for eeg_x, bvp_x, labels in train_loader:
        eeg_x = eeg_x.to(device)
        bvp_x = bvp_x.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(eeg_x, bvp_x)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate_fusion(model, loader, criterion, device):
    """Evaluate fusion model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for eeg_x, bvp_x, labels in loader:
            eeg_x = eeg_x.to(device)
            bvp_x = bvp_x.to(device)
            labels = labels.to(device)
            
            logits = model(eeg_x, bvp_x)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


# ==================================================
# MAIN TRAINING PIPELINE
# ==================================================

def main(args):
    """Main multimodal fusion training pipeline."""
    
    print("\n" + "="*80)
    print("MULTIMODAL EEG + BVP FUSION PIPELINE")
    print("="*80)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # ============================================================
    # STEP 1: LOAD EEG DATA
    # ============================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING EEG DATA")
    print("="*80)
    
    eeg_config = EEGConfig()
    eeg_config.USE_BASELINE_REDUCTION = args.use_baseline_reduction
    
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(
        args.eeg_data_root, eeg_config
    )
    eeg_X_features = extract_eeg_features(eeg_X_raw, eeg_config)
    
    print(f"✅ EEG data: {eeg_X_features.shape}")
    
    # ============================================================
    # STEP 2: LOAD BVP DATA
    # ============================================================
    print("\n" + "="*80)
    print("STEP 2: LOADING BVP DATA")
    print("="*80)
    
    bvp_config = BVPConfig()
    bvp_config.USE_BVP_BASELINE_REDUCTION = args.use_baseline_reduction
    bvp_config.USE_BVP_BASELINE_CORRECTION = False
    
    # Fix: load_bvp_data returns 4 values, not 5
    bvp_X_raw, bvp_y, bvp_subjects, bvp_label_map = load_bvp_data(
        args.bvp_data_root, bvp_config
    )
    
    print(f"✅ BVP data: {bvp_X_raw.shape}")
    
    # ============================================================
    # STEP 3: ALIGN DATASETS
    # ============================================================
    eeg_data = (eeg_X_features, eeg_y, eeg_subjects, None, eeg_label_map)
    bvp_data = (bvp_X_raw, bvp_y, bvp_subjects, None, bvp_label_map)
    
    aligned_eeg, aligned_bvp, common_subjects = align_eeg_bvp_data(eeg_data, bvp_data)
    
    # ============================================================
    # STEP 4: CREATE SPLITS
    # ============================================================
    splits = create_multimodal_splits(aligned_eeg, aligned_bvp, eeg_config)
    
    # Create datasets
    train_dataset = MultimodalDataset(
        aligned_eeg['X'][splits['train']['eeg']],
        aligned_eeg['y'][splits['train']['eeg']],
        aligned_eeg['subjects'][splits['train']['eeg']],
        aligned_bvp['X'][splits['train']['bvp']],
        aligned_bvp['y'][splits['train']['bvp']],
        aligned_bvp['subjects'][splits['train']['bvp']]
    )
    
    val_dataset = MultimodalDataset(
        aligned_eeg['X'][splits['val']['eeg']],
        aligned_eeg['y'][splits['val']['eeg']],
        aligned_eeg['subjects'][splits['val']['eeg']],
        aligned_bvp['X'][splits['val']['bvp']],
        aligned_bvp['y'][splits['val']['bvp']],
        aligned_bvp['subjects'][splits['val']['bvp']]
    )
    
    test_dataset = MultimodalDataset(
        aligned_eeg['X'][splits['test']['eeg']],
        aligned_eeg['y'][splits['test']['eeg']],
        aligned_eeg['subjects'][splits['test']['eeg']],
        aligned_bvp['X'][splits['test']['bvp']],
        aligned_bvp['y'][splits['test']['bvp']],
        aligned_bvp['subjects'][splits['test']['bvp']]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\n✅ DataLoaders created:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # ============================================================
    # STEP 5: CREATE FUSION MODEL
    # ============================================================
    print("\n" + "="*80)
    print("STEP 5: CREATING FUSION MODEL")
    print("="*80)
    
    # Create EEG encoder
    if args.eeg_checkpoint and os.path.exists(args.eeg_checkpoint):
        print(f"Loading pre-trained EEG model from {args.eeg_checkpoint}")
        eeg_model = SimpleBiLSTMClassifier(dx=26, n_channels=4, hidden=256, layers=3, n_classes=4)
        eeg_model.load_state_dict(torch.load(args.eeg_checkpoint, map_location=device))
    else:
        print("⚠️  No pre-trained EEG model provided, using random initialization")
        eeg_model = SimpleBiLSTMClassifier(dx=26, n_channels=4, hidden=256, layers=3, n_classes=4)
    
    eeg_encoder = EEGEncoder(eeg_model, freeze_weights=args.freeze_encoders)
    
    # Create BVP encoder (even if disabled, needed for model structure)
    bvp_encoder = BVPHybridEncoder(
        input_size=1,
        hidden_size=32,
        dropout=0.3,
        use_multiscale=args.use_multiscale_bvp
    )
    
    if args.freeze_encoders:
        for param in bvp_encoder.parameters():
            param.requires_grad = False
        bvp_encoder.eval()
    
    # Check if BVP is disabled
    if args.disable_bvp:
        print("\n⚠️  BVP MODALITY DISABLED - Running in EEG-ONLY mode")
        print("   This will show baseline EEG performance without BVP fusion")
    
    # Create fusion model
    if args.fusion_type == 'early':
        if args.disable_bvp:
            print("⚠️  Early fusion requires BVP. Switching to hybrid fusion with use_bvp=False")
            model = HybridFusionModel(eeg_encoder, bvp_encoder, n_classes=4, shared_dim=128, num_heads=4, use_bvp=False)
        else:
            model = EarlyFusionModel(eeg_encoder, bvp_encoder, n_classes=4)
    elif args.fusion_type == 'late':
        if args.disable_bvp:
            print("⚠️  Late fusion requires BVP. Switching to hybrid fusion with use_bvp=False")
            model = HybridFusionModel(eeg_encoder, bvp_encoder, n_classes=4, shared_dim=128, num_heads=4, use_bvp=False)
        else:
            model = LateFusionModel(eeg_encoder, bvp_encoder, n_classes=4)
    elif args.fusion_type == 'hybrid':
        model = HybridFusionModel(
            eeg_encoder, 
            bvp_encoder, 
            n_classes=4, 
            shared_dim=128, 
            num_heads=4,
            use_bvp=not args.disable_bvp  # Disable BVP if flag is set
        )
    else:
        raise ValueError(f"Unknown fusion type: {args.fusion_type}")
    
    model = model.to(device)
    
    # Print model info
    params = get_trainable_parameters(model)
    mode_str = "EEG-ONLY" if args.disable_bvp else f"{args.fusion_type.capitalize()} Fusion"
    print(f"\n🧠 {mode_str} Model:")
    print(f"   Total parameters:     {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    print(f"   Frozen parameters:    {params['frozen']:,}")
    print(f"   Trainable ratio:      {params['trainable_ratio']:.2%}")
    
    # ============================================================
    # STEP 6: TRAIN FUSION MODEL
    # ============================================================
    print("\n" + "="*80)
    print("STEP 6: TRAINING FUSION MODEL")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_fusion_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate_fusion(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch:03d} | Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                  f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.3f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), args.checkpoint)
            if epoch % 5 == 0:
                print(f"   ✅ Best model saved (F1: {val_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break
    
    # ============================================================
    # STEP 7: TEST EVALUATION
    # ============================================================
    print("\n" + "="*80)
    print("STEP 7: FINAL EVALUATION")
    print("="*80)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate_fusion(
        model, test_loader, criterion, device
    )
    
    print(f"\n🎯 Test Results:")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   Macro-F1: {test_f1:.3f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=eeg_config.IDX_TO_LABEL,
                                digits=3, zero_division=0))
    
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    print("\n" + "="*80)
    print("✅ MULTIMODAL FUSION PIPELINE COMPLETE!")
    print("="*80)
    print(f"   Model saved: {args.checkpoint}")
    print(f"   Best Val F1: {best_val_f1:.3f}")
    print(f"   Test F1:     {test_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal EEG+BVP Fusion Training")
    
    # Data paths
    parser.add_argument('--eeg_data_root', type=str, 
                        default="/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR",
                        help='EEG data root directory')
    parser.add_argument('--bvp_data_root', type=str,
                        default="/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined",
                        help='BVP data root directory')
    
    # Model configuration
    parser.add_argument('--fusion_type', type=str, default='hybrid',
                        choices=['early', 'late', 'hybrid'],
                        help='Fusion strategy')
    parser.add_argument('--eeg_checkpoint', type=str, default='best_eeg_model.pt',
                        help='Pre-trained EEG model checkpoint')
    parser.add_argument('--freeze_encoders', action='store_true',
                        help='Freeze encoder weights during fusion training')
    parser.add_argument('--use_multiscale_bvp', action='store_true',
                        help='Use multi-scale BVP encoder')
    parser.add_argument('--use_baseline_reduction', action='store_true',
                        help='Apply baseline reduction to both modalities')
    parser.add_argument('--disable_bvp', action='store_true',
                        help='Disable BVP modality for EEG-only mode')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--checkpoint', type=str, default='best_fusion_model.pt',
                        help='Output checkpoint path')
    
    args = parser.parse_args()
    main(args)
