"""
Simple EEG + BVP Handcrafted Features Pipeline
================================================

This script uses a simplified approach:
1. Load EEG features (104 features from 4 channels × 26 features)
2. Extract BVP handcrafted features (11 features: stats + HRV + pulse amplitude)
3. Concatenate EEG + BVP features → 115 total features
4. Train a simple classifier (same BiLSTM as EEG-only)

This is MUCH simpler than complex fusion with attention mechanisms.

Author: Final Year Project
Date: 2026-02-24
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report

# Import EEG components
from eeg_config import Config
from eeg_bilstm_model import SimpleBiLSTMClassifier
from eeg_data_loader import load_eeg_data, extract_eeg_features, create_data_splits
from eeg_trainer import mixup_data, mixup_criterion

# Import BVP components
from bvp_config import BVPConfig
from bvp_data_loader import load_bvp_data
from bvp_handcrafted_features import BVPHandcraftedFeatures


# ==================================================
# SIMPLE MULTIMODAL DATASET
# ==================================================

class SimpleEEGBVPDataset(Dataset):
    """
    Simple dataset that pairs EEG features with BVP handcrafted features.
    """
    
    def __init__(self, eeg_X, bvp_X, labels, subjects):
        """
        Args:
            eeg_X: EEG features [N, channels, features]
            bvp_X: BVP raw signals [N, time_steps] for feature extraction
            labels: Class labels [N]
            subjects: Subject IDs [N]
        """
        self.eeg_X = eeg_X
        self.bvp_X = bvp_X
        self.labels = labels
        self.subjects = subjects
        
        # Initialize BVP feature extractor (11 handcrafted features)
        self.bvp_feature_extractor = BVPHandcraftedFeatures(
            sampling_rate=64.0,
            min_peak_distance=20
        )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get EEG features
        eeg_sample = torch.from_numpy(self.eeg_X[idx]).float()  # [4, 26]
        
        # Get BVP signal and extract handcrafted features
        bvp_signal = torch.from_numpy(self.bvp_X[idx]).unsqueeze(-1).float()  # [time_steps, 1]
        
        # Extract handcrafted features: [1, time_steps, 1] -> [1, 11]
        with torch.no_grad():
            bvp_features = self.bvp_feature_extractor(bvp_signal.unsqueeze(0))  # Add batch dim
            bvp_features = bvp_features.squeeze(0)  # Remove batch dim -> [11]
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return eeg_sample, bvp_features, label


# ==================================================
# SIMPLE CONCATENATION MODEL
# ==================================================

class SimpleEEGBVPModel(nn.Module):
    """
    Simple model: EEG BiLSTM + BVP handcrafted features concatenation.
    
    Architecture:
    1. EEG: 4 channels × 26 features -> BiLSTM -> 512-dim representation
    2. BVP: 11 handcrafted features (pre-extracted)
    3. Concatenate: [512 + 11] = 523 features
    4. Classify: MLP head
    """
    
    def __init__(self, n_classes=4, eeg_hidden=256, eeg_layers=3, dropout=0.4):
        super().__init__()
        
        # EEG encoder (BiLSTM)
        self.eeg_encoder = SimpleBiLSTMClassifier(
            dx=26, 
            n_channels=4, 
            hidden=eeg_hidden, 
            layers=eeg_layers,
            n_classes=n_classes,  # Dummy, we'll use our own classifier
            p_drop=dropout
        )
        
        # Dimension after BiLSTM (bidirectional)
        eeg_dim = eeg_hidden * 2  # 512
        bvp_dim = 11  # Handcrafted features
        
        # Combined classifier
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
    
    def forward(self, eeg_x, bvp_features):
        """
        Forward pass.
        
        Args:
            eeg_x: [batch_size, 4, 26] - EEG features
            bvp_features: [batch_size, 11] - BVP handcrafted features
        
        Returns:
            logits: [batch_size, n_classes]
        """
        # Extract EEG representation using BiLSTM
        # We need to get the representation before the classifier
        B, C, dx = eeg_x.shape
        x = self.eeg_encoder.input_proj(eeg_x)
        h, _ = self.eeg_encoder.lstm(x)
        h = self.eeg_encoder.drop(self.eeg_encoder.norm(h))
        
        # Attention pooling
        scores = self.eeg_encoder.attn(h)
        alpha = torch.softmax(scores, dim=1)
        eeg_repr = (alpha * h).sum(dim=1)  # [batch_size, 512]
        
        # Concatenate with BVP features
        combined = torch.cat([eeg_repr, bvp_features], dim=1)  # [batch_size, 523]
        
        # Classify
        logits = self.classifier(combined)
        
        return logits


# ==================================================
# MAIN PIPELINE
# ==================================================

def main(args):
    print("=" * 80)
    print("SIMPLE EEG + BVP HANDCRAFTED FEATURES PIPELINE")
    print("=" * 80)
    
    # Configuration
    config = Config()
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    print(f"\n📋 Configuration:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Subject-Independent: {config.SUBJECT_INDEPENDENT}")
    print(f"   EEG Baseline Reduction: {config.USE_BASELINE_REDUCTION}")
    
    # ============================================================
    # STEP 1: LOAD EEG DATA
    # ============================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING EEG DATA")
    print("="*80)
    
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(config.DATA_ROOT, config)
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    print(f"✅ EEG data: {eeg_X_features.shape}")
    
    # ============================================================
    # STEP 2: LOAD BVP DATA
    # ============================================================
    print("\n" + "="*80)
    print("STEP 2: LOADING BVP DATA")
    print("="*80)
    
    bvp_config = BVPConfig()
    bvp_config.USE_BVP_BASELINE_REDUCTION = False
    bvp_config.USE_BVP_BASELINE_CORRECTION = False
    
    bvp_X_raw, bvp_y, bvp_subjects, bvp_label_map = load_bvp_data(args.bvp_data_root, bvp_config)
    
    print(f"✅ BVP data: {bvp_X_raw.shape}")
    
    # ============================================================
    # STEP 3: ALIGN DATASETS BY COMMON SUBJECTS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 3: ALIGNING DATASETS")
    print("="*80)
    
    eeg_subj_set = set(eeg_subjects)
    bvp_subj_set = set(bvp_subjects)
    common_subjects = list(eeg_subj_set & bvp_subj_set)
    
    print(f"   EEG subjects: {len(eeg_subj_set)}")
    print(f"   BVP subjects: {len(bvp_subj_set)}")
    print(f"   Common subjects: {len(common_subjects)}")
    
    # Filter to common subjects
    eeg_mask = np.isin(eeg_subjects, common_subjects)
    bvp_mask = np.isin(bvp_subjects, common_subjects)
    
    eeg_X_features = eeg_X_features[eeg_mask]
    eeg_y = eeg_y[eeg_mask]
    eeg_subjects = eeg_subjects[eeg_mask]
    
    bvp_X_raw = bvp_X_raw[bvp_mask]
    bvp_y = bvp_y[bvp_mask]
    bvp_subjects = bvp_subjects[bvp_mask]
    
    print(f"   Aligned EEG: {eeg_X_features.shape[0]} samples")
    print(f"   Aligned BVP: {bvp_X_raw.shape[0]} samples")
    
    # ============================================================
    # STEP 4: CREATE DATA SPLITS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 4: CREATING DATA SPLITS")
    print("="*80)
    
    split_indices = create_data_splits(eeg_y, eeg_subjects, config)
    
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    print(f"\n📋 Split sizes:")
    print(f"   Train: {len(train_idx)}")
    print(f"   Val: {len(val_idx)}")
    print(f"   Test: {len(test_idx)}")
    
    # Get splits
    Xtr, Xva, Xte = eeg_X_features[train_idx], eeg_X_features[val_idx], eeg_X_features[test_idx]
    ytr, yva, yte = eeg_y[train_idx], eeg_y[val_idx], eeg_y[test_idx]
    
    bvp_tr = bvp_X_raw[train_idx]
    bvp_va = bvp_X_raw[val_idx]
    bvp_te = bvp_X_raw[test_idx]
    
    subj_tr = eeg_subjects[train_idx]
    subj_va = eeg_subjects[val_idx]
    subj_te = eeg_subjects[test_idx]
    
    # ============================================================
    # STEP 5: STANDARDIZE EEG FEATURES
    # ============================================================
    print("\n" + "="*80)
    print("STEP 5: STANDARDIZING EEG FEATURES")
    print("="*80)
    
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    
    print(f"   Train: {Xtr.shape}")
    print(f"   Val: {Xva.shape}")
    print(f"   Test: {Xte.shape}")
    
    # ============================================================
    # STEP 6: CREATE DATASETS AND DATALOADERS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 6: CREATING DATASETS")
    print("="*80)
    
    tr_ds = SimpleEEGBVPDataset(Xtr, bvp_tr, ytr, subj_tr)
    va_ds = SimpleEEGBVPDataset(Xva, bvp_va, yva, subj_va)
    te_ds = SimpleEEGBVPDataset(Xte, bvp_te, yte, subj_te)
    
    print(f"   Train dataset: {len(tr_ds)} samples")
    print(f"   Val dataset: {len(va_ds)} samples")
    print(f"   Test dataset: {len(te_ds)} samples")
    
    # Balanced sampling
    class_counts = np.bincount(ytr, minlength=config.NUM_CLASSES).astype(np.float32)
    class_sample_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_sample_weights[ytr]
    sample_weights_tensor = torch.from_numpy(sample_weights.astype(np.float32))
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    tr_loader = DataLoader(tr_ds, batch_size=config.EEG_BATCH_SIZE, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    
    print(f"   Batch size: Train={config.EEG_BATCH_SIZE}, Val/Test=256")
    
    # ============================================================
    # STEP 7: CREATE MODEL
    # ============================================================
    print("\n" + "="*80)
    print("STEP 7: CREATING MODEL")
    print("="*80)
    
    model = SimpleEEGBVPModel(
        n_classes=config.NUM_CLASSES,
        eeg_hidden=256,
        eeg_layers=3,
        dropout=0.4
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model: SimpleEEGBVPModel")
    print(f"   EEG features: 4 × 26 = 104")
    print(f"   BVP features: 11 (handcrafted)")
    print(f"   Combined: 512 (EEG BiLSTM) + 11 (BVP) = 523")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ============================================================
    # STEP 8: TRAINING SETUP
    # ============================================================
    print("\n" + "="*80)
    print("STEP 8: TRAINING")
    print("="*80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_f1, best_state, wait = 0.0, None, 0
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for eeg_xb, bvp_fb, yb in tr_loader:
            eeg_xb = eeg_xb.to(config.DEVICE)
            bvp_fb = bvp_fb.to(config.DEVICE)
            yb = yb.to(config.DEVICE)
            
            # Mixup on EEG only (BVP features are handcrafted, don't augment)
            if config.USE_MIXUP and np.random.rand() < 0.5:
                eeg_xb_mix, ya, yb_m, lam = mixup_data(eeg_xb, yb, alpha=config.MIXUP_ALPHA)
                optimizer.zero_grad()
                logits = model(eeg_xb_mix, bvp_fb)
                loss = mixup_criterion(criterion, logits, ya, yb_m, lam)
            else:
                optimizer.zero_grad()
                logits = model(eeg_xb, bvp_fb)
                loss = criterion(logits, yb)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for eeg_xb, bvp_fb, yb in va_loader:
                eeg_xb = eeg_xb.to(config.DEVICE)
                bvp_fb = bvp_fb.to(config.DEVICE)
                yb = yb.to(config.DEVICE)
                
                logits = model(eeg_xb, bvp_fb)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_acc = (all_preds == all_targets).mean()
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        if epoch % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(model.state_dict(), args.checkpoint)
        else:
            wait += 1
            if wait >= config.EEG_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # ============================================================
    # STEP 9: TEST EVALUATION
    # ============================================================
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for eeg_xb, bvp_fb, yb in te_loader:
            eeg_xb = eeg_xb.to(config.DEVICE)
            bvp_fb = bvp_fb.to(config.DEVICE)
            yb = yb.to(config.DEVICE)
            
            logits = model(eeg_xb, bvp_fb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_acc = (all_preds == all_targets).mean()
    test_f1 = f1_score(all_targets, all_preds, average='macro')
    
    print(f"\n📊 Results:")
    print(f"   Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"   Test Macro-F1: {test_f1:.3f}")
    
    id2lab = {v: k for k, v in eeg_label_map.items()}
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"✅ Best Val F1: {best_f1:.3f}")
    print(f"✅ Test F1: {test_f1:.3f}")
    print(f"✅ Model saved: {args.checkpoint}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple EEG + BVP Handcrafted Features")
    
    parser.add_argument('--bvp_data_root', type=str,
                        default="/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined",
                        help='BVP data root directory')
    parser.add_argument('--checkpoint', type=str, default='best_simple_eeg_bvp_model.pt',
                        help='Output checkpoint path')
    
    args = parser.parse_args()
    main(args)
