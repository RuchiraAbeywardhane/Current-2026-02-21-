"""
EEG + BVP Multimodal Fusion Pipeline
=====================================

This script uses the EXACT SAME EEG pipeline from EEGPipeline.py with BVP fusion added.

ALL EEG components are IDENTICAL:
- Random seeds (SEED=42)
- Data loading (load_eeg_data, extract_eeg_features)
- Baseline reduction (InvBase method)
- Standardization (mu, sd from training set)
- Balanced sampling (WeightedRandomSampler)
- Mixup augmentation (alpha=0.2, 50% probability)
- Optimizer (AdamW, lr=1e-3, weight_decay=1e-4)
- Scheduler (CosineAnnealingWarmRestarts, T_0=10, T_mult=2)
- Loss (CrossEntropyLoss with class weights)
- Batch sizes (train=64, val/test=256)
- Early stopping (patience=20, based on val_f1)

Author: Final Year Project
Date: 2026-02-24
"""

import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import f1_score, classification_report

# Import EEG configuration - EXACT SAME AS EEGPipeline.py
from eeg_config import Config

# Import EEG model - EXACT SAME AS EEGPipeline.py
from eeg_bilstm_model import SimpleBiLSTMClassifier

# Import EEG data loaders - EXACT SAME AS EEGPipeline.py
from eeg_data_loader import (
    load_eeg_data,
    extract_eeg_features,
    create_data_splits
)

# Import EEG training utilities - EXACT SAME AS EEGPipeline.py
from eeg_trainer import mixup_data, mixup_criterion

# Import BVP components
from bvp_config import BVPConfig
from bvp_data_loader import load_bvp_data
from bvp_hybrid_encoder import BVPHybridEncoder

# Import fusion models
from multimodal_fusion import (
    EEGEncoder,
    HybridFusionModel,
    get_trainable_parameters
)


# ==================================================
# MULTIMODAL DATASET (Pairs EEG + BVP)
# ==================================================

class EEGBVPDataset(Dataset):
    """
    Dataset that pairs EEG features with BVP signals.
    Aligns samples by subject and emotion label.
    """
    
    def __init__(self, eeg_X, eeg_y, eeg_subjects, bvp_X, bvp_y, bvp_subjects):
        """
        Args:
            eeg_X: EEG features [N, channels, features]
            eeg_y: EEG labels [N]
            eeg_subjects: EEG subject IDs [N]
            bvp_X: BVP signals [M, time_steps]
            bvp_y: BVP labels [M]
            bvp_subjects: BVP subject IDs [M]
        """
        self.samples = []
        
        # Create lookup dictionaries
        eeg_lookup = {}
        for i, (subj, label) in enumerate(zip(eeg_subjects, eeg_y)):
            key = f"{subj}_{label}"
            if key not in eeg_lookup:
                eeg_lookup[key] = []
            eeg_lookup[key].append(i)
        
        bvp_lookup = {}
        for i, (subj, label) in enumerate(zip(bvp_subjects, bvp_y)):
            key = f"{subj}_{label}"
            if key not in bvp_lookup:
                bvp_lookup[key] = []
            bvp_lookup[key].append(i)
        
        # Pair EEG and BVP samples
        for key in eeg_lookup:
            if key in bvp_lookup:
                eeg_indices = eeg_lookup[key]
                bvp_indices = bvp_lookup[key]
                n_pairs = min(len(eeg_indices), len(bvp_indices))
                
                for i in range(n_pairs):
                    self.samples.append((eeg_indices[i], bvp_indices[i], eeg_y[eeg_indices[i]]))
        
        self.eeg_X = eeg_X
        self.bvp_X = bvp_X
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        eeg_idx, bvp_idx, label = self.samples[idx]
        
        eeg_sample = torch.from_numpy(self.eeg_X[eeg_idx]).float()
        bvp_sample = torch.from_numpy(self.bvp_X[bvp_idx]).unsqueeze(-1).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return eeg_sample, bvp_sample, label_tensor


# ==================================================
# MAIN FUSION TRAINING PIPELINE
# ==================================================

def main(args):
    """Main EEG+BVP fusion pipeline - EXACT SAME EEG setup as EEGPipeline.py"""
    
    print("=" * 80)
    print("EEG + BVP MULTIMODAL FUSION PIPELINE")
    print("=" * 80)
    
    # ============================================================
    # EXACT SAME CONFIG & RANDOM SEEDS AS EEGPipeline.py
    # ============================================================
    config = Config()
    
    # Override with command line args if provided
    if args.disable_bvp:
        print("⚠️  BVP DISABLED - Running EEG-only mode for baseline comparison")
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    print(f"Device: {config.DEVICE}")
    print(f"Mode: {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Baseline Reduction: {config.USE_BASELINE_REDUCTION}")
    print("=" * 80)
    
    # ============================================================
    # STEP 1: LOAD EEG DATA - EXACT SAME AS EEGPipeline.py
    # ============================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING EEG DATA (EXACT SAME AS EEGPipeline.py)")
    print("="*80)
    
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(config.DATA_ROOT, config)
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    print(f"✅ EEG data loaded: {eeg_X_features.shape}")
    
    # ============================================================
    # STEP 2: LOAD BVP DATA (New for fusion)
    # ============================================================
    if not args.disable_bvp:
        print("\n" + "="*80)
        print("STEP 2: LOADING BVP DATA")
        print("="*80)
        
        bvp_config = BVPConfig()
        bvp_config.USE_BVP_BASELINE_REDUCTION = config.USE_BASELINE_REDUCTION
        bvp_config.USE_BVP_BASELINE_CORRECTION = False
        
        bvp_X_raw, bvp_y, bvp_subjects, bvp_label_map = load_bvp_data(
            args.bvp_data_root, bvp_config
        )
        
        print(f"✅ BVP data loaded: {bvp_X_raw.shape}")
        
        # Align datasets by common subjects
        eeg_subj_set = set(eeg_subjects)
        bvp_subj_set = set(bvp_subjects)
        common_subjects = list(eeg_subj_set & bvp_subj_set)
        
        print(f"\n📊 Dataset Alignment:")
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
    # STEP 3: CREATE DATA SPLITS - EXACT SAME AS EEGPipeline.py
    # ============================================================
    print("\n" + "="*80)
    print("CREATING DATA SPLIT (EXACT SAME AS EEGPipeline.py)")
    print("="*80)
    
    split_indices = create_data_splits(eeg_y, eeg_subjects, config)
    
    print(f"\n📋 Split Summary:")
    print(f"   Train samples: {len(split_indices['train'])}")
    print(f"   Val samples: {len(split_indices['val'])}")
    print(f"   Test samples: {len(split_indices['test'])}")
    
    # Get split data
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    Xtr, Xva, Xte = eeg_X_features[train_idx], eeg_X_features[val_idx], eeg_X_features[test_idx]
    ytr, yva, yte = eeg_y[train_idx], eeg_y[val_idx], eeg_y[test_idx]
    
    # ============================================================
    # STEP 4: STANDARDIZATION - EXACT SAME AS eeg_trainer.py
    # ============================================================
    print("\n" + "="*80)
    print("STANDARDIZING EEG FEATURES (EXACT SAME AS eeg_trainer.py)")
    print("="*80)
    
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    
    print(f"Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
    
    # ============================================================
    # STEP 5: CREATE DATALOADERS - EXACT SAME AS eeg_trainer.py
    # ============================================================
    print("\n" + "="*80)
    print("CREATING DATALOADERS (EXACT SAME AS eeg_trainer.py)")
    print("="*80)
    
    # EXACT SAME: Balanced sampling with WeightedRandomSampler
    class_counts = np.bincount(ytr, minlength=config.NUM_CLASSES).astype(np.float32)
    class_sample_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_sample_weights[ytr]
    sample_weights_tensor = torch.from_numpy(sample_weights.astype(np.float32))
    
    if not args.disable_bvp:
        # Create multimodal datasets - FIX: Use correct BVP indices
        bvp_tr = bvp_X_raw[train_idx]
        bvp_va = bvp_X_raw[val_idx]
        bvp_te = bvp_X_raw[test_idx]
        
        # FIX: Use BVP labels and subjects for proper alignment
        bvp_ytr = bvp_y[train_idx]
        bvp_yva = bvp_y[val_idx]
        bvp_yte = bvp_y[test_idx]
        
        bvp_subj_tr = bvp_subjects[train_idx]
        bvp_subj_va = bvp_subjects[val_idx]
        bvp_subj_te = bvp_subjects[test_idx]
        
        subj_tr = eeg_subjects[train_idx]
        subj_va = eeg_subjects[val_idx]
        subj_te = eeg_subjects[test_idx]
        
        # Create properly aligned datasets
        tr_ds = EEGBVPDataset(Xtr, ytr, subj_tr, bvp_tr, bvp_ytr, bvp_subj_tr)
        va_ds = EEGBVPDataset(Xva, yva, subj_va, bvp_va, bvp_yva, bvp_subj_va)
        te_ds = EEGBVPDataset(Xte, yte, subj_te, bvp_te, bvp_yte, bvp_subj_te)
        
        print(f"   Multimodal datasets: Train={len(tr_ds)}, Val={len(va_ds)}, Test={len(te_ds)}")
        
        # FIX: Recompute sample weights for the ALIGNED dataset size
        # The multimodal dataset may be smaller than the original EEG dataset
        aligned_labels = np.array([sample[2] for sample in tr_ds.samples])
        aligned_class_counts = np.bincount(aligned_labels, minlength=config.NUM_CLASSES).astype(np.float32)
        aligned_class_weights = 1.0 / np.clip(aligned_class_counts, 1.0, None)
        aligned_sample_weights = aligned_class_weights[aligned_labels]
        sample_weights_tensor = torch.from_numpy(aligned_sample_weights.astype(np.float32))
        
        print(f"   Class distribution in aligned dataset: {aligned_class_counts}")
        
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    if args.disable_bvp:
        # EEG-only mode
        tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
        te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    
    # EXACT SAME: DataLoader settings from EEG pipeline
    tr_loader = DataLoader(tr_ds, batch_size=config.EEG_BATCH_SIZE, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    
    print(f"   Batch sizes: Train={config.EEG_BATCH_SIZE}, Val/Test=256")
    
    # ============================================================
    # STEP 6: CREATE MODEL - EEG encoder + optional BVP
    # ============================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    # Create EEG model - EXACT SAME AS eeg_trainer.py
    eeg_model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3,
        n_classes=config.NUM_CLASSES, p_drop=0.4
    )
    
    if not args.disable_bvp:
        # Create BVP encoder
        bvp_encoder = BVPHybridEncoder(
            input_size=1,
            hidden_size=32,
            dropout=0.3,
            use_multiscale=args.use_multiscale_bvp
        )
        
        # Wrap in fusion model
        eeg_encoder = EEGEncoder(eeg_model, freeze_weights=False)
        model = HybridFusionModel(
            eeg_encoder,
            bvp_encoder,
            n_classes=config.NUM_CLASSES,
            shared_dim=128,
            num_heads=4,
            use_bvp=True
        )
    else:
        # EEG-only mode
        model = eeg_model
    
    model = model.to(config.DEVICE)
    
    params = get_trainable_parameters(model) if not args.disable_bvp else {'total': sum(p.numel() for p in model.parameters()), 'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    print(f"\n🧠 Model:")
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    
    # ============================================================
    # STEP 7: TRAINING SETUP - EXACT SAME AS eeg_trainer.py
    # ============================================================
    print("\n" + "="*80)
    print("TRAINING SETUP (EXACT SAME AS eeg_trainer.py)")
    print("="*80)
    print(f"Mixup Augmentation: {'ENABLED' if config.USE_MIXUP else 'DISABLED'}")
    if config.USE_MIXUP:
        print(f"Mixup Alpha: {config.MIXUP_ALPHA}")
    print("="*80)
    
    # EXACT SAME: Optimizer from eeg_trainer.py
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    
    # EXACT SAME: Scheduler from eeg_trainer.py
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # EXACT SAME: Loss with class weights from eeg_trainer.py
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # ============================================================
    # STEP 8: TRAINING LOOP - EXACT SAME AS eeg_trainer.py
    # ============================================================
    print("\n" + "="*80)
    print("TRAINING (EXACT SAME LOOP AS eeg_trainer.py)")
    print("="*80)
    
    best_f1, best_state, wait = 0.0, None, 0
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch in tr_loader:
            if not args.disable_bvp:
                xb, bvp_xb, yb = batch
                xb, bvp_xb, yb = xb.to(config.DEVICE), bvp_xb.to(config.DEVICE), yb.to(config.DEVICE)
            else:
                xb, yb = batch
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            
            # EXACT SAME: Mixup augmentation from eeg_trainer.py
            if config.USE_MIXUP and np.random.rand() < 0.5:
                xb_mix, ya, yb_m, lam = mixup_data(xb, yb, alpha=config.MIXUP_ALPHA)
                optimizer.zero_grad()
                
                if not args.disable_bvp:
                    logits = model(xb_mix, bvp_xb)
                else:
                    logits = model(xb_mix)
                
                loss = mixup_criterion(criterion, logits, ya, yb_m, lam)
            else:
                optimizer.zero_grad()
                
                if not args.disable_bvp:
                    logits = model(xb, bvp_xb)
                else:
                    logits = model(xb)
                
                loss = criterion(logits, yb)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # EXACT SAME
            optimizer.step()
            scheduler.step()  # EXACT SAME: called per-batch
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        
        # EXACT SAME: Validation from eeg_trainer.py
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in va_loader:
                if not args.disable_bvp:
                    xb, bvp_xb, yb = batch
                    xb, bvp_xb, yb = xb.to(config.DEVICE), bvp_xb.to(config.DEVICE), yb.to(config.DEVICE)
                    logits = model(xb, bvp_xb)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                    logits = model(xb)
                
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_acc = (all_preds == all_targets).mean()
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        if epoch % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")
        
        # EXACT SAME: Early stopping logic from eeg_trainer.py
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
    # STEP 9: TEST EVALUATION - EXACT SAME AS eeg_trainer.py
    # ============================================================
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in te_loader:
            if not args.disable_bvp:
                xb, bvp_xb, yb = batch
                xb, bvp_xb, yb = xb.to(config.DEVICE), bvp_xb.to(config.DEVICE), yb.to(config.DEVICE)
                logits = model(xb, bvp_xb)
            else:
                xb, yb = batch
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                logits = model(xb)
            
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_acc = (all_preds == all_targets).mean()
    test_f1 = f1_score(all_targets, all_preds, average='macro')
    
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"Test Macro-F1: {test_f1:.3f}")
    
    id2lab = {v: k for k, v in eeg_label_map.items()}
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    print("\n" + "=" * 80)
    print("🎉 FUSION PIPELINE COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {args.checkpoint}")
    print(f"✅ Best Val F1: {best_f1:.3f}")
    print(f"✅ Test F1: {test_f1:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG+BVP Fusion (EXACT EEG pipeline)")
    
    parser.add_argument('--bvp_data_root', type=str,
                        default="/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined",
                        help='BVP data root directory')
    parser.add_argument('--disable_bvp', action='store_true',
                        help='Disable BVP for EEG-only baseline')
    parser.add_argument('--use_multiscale_bvp', action='store_true',
                        help='Use multi-scale BVP encoder')
    parser.add_argument('--checkpoint', type=str, default='best_fusion_model.pt',
                        help='Output checkpoint path')
    
    args = parser.parse_args()
    main(args)
