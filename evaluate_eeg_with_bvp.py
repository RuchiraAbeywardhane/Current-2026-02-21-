"""
Evaluate EEG Model with Optional BVP Post-Processing
=====================================================

This script loads your EXISTING trained EEG model and evaluates it
with and without BVP post-processing adjustment.

Usage:
    python evaluate_eeg_with_bvp.py --eeg_model best_eeg_model.pt

Author: Final Year Project
Date: 2026-02-24
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Import EEG components
from eeg_config import Config
from eeg_bilstm_model import SimpleBiLSTMClassifier
from eeg_data_loader import load_eeg_data, extract_eeg_features, create_data_splits

# Import BVP components
from bvp_config import BVPConfig
from bvp_data_loader import load_bvp_data
from bvp_handcrafted_features import BVPHandcraftedFeatures


# ==================================================
# BVP POST-PROCESSOR
# ==================================================

class BVPPostProcessor:
    """Use BVP to post-process EEG predictions."""
    
    def __init__(self, adjustment_strength=0.2):
        self.adjustment_strength = adjustment_strength
        
        # Q1=Enthusiasm (Pos+High), Q2=Fear (Neg+High)
        # Q3=Sadness (Neg+Low), Q4=Neutral (Pos+Low)
        self.high_arousal_classes = [0, 1]  # Q1, Q2
        self.low_arousal_classes = [2, 3]   # Q3, Q4
    
    def compute_arousal_score(self, bvp_features):
        """Compute arousal from BVP: -1 (low) to +1 (high)."""
        heart_rate = bvp_features[10]  # BPM
        rmssd = bvp_features[6]         # HRV
        
        # Normalize HR (typical: 60-100 bpm)
        hr_norm = np.clip((heart_rate - 70) / 30.0, -1, 1)
        
        # RMSSD: higher = more relaxed (negate for arousal)
        rmssd_norm = np.clip(-rmssd / 50.0, -1, 1)
        
        # Combine
        arousal = 0.6 * hr_norm + 0.4 * rmssd_norm
        return arousal
    
    def adjust_logits(self, eeg_logits, bvp_features):
        """Adjust EEG logits based on BVP arousal."""
        batch_size = eeg_logits.shape[0]
        adjusted = eeg_logits.clone()
        
        for i in range(batch_size):
            arousal = self.compute_arousal_score(bvp_features[i].cpu().numpy())
            adj = self.adjustment_strength * arousal
            
            if arousal > 0:  # High arousal
                adjusted[i, self.high_arousal_classes] += adj
                adjusted[i, self.low_arousal_classes] -= adj * 0.5
            else:  # Low arousal
                adjusted[i, self.low_arousal_classes] += abs(adj)
                adjusted[i, self.high_arousal_classes] -= abs(adj) * 0.5
        
        return adjusted


# ==================================================
# DATASET
# ==================================================

class EEGBVPEvalDataset(Dataset):
    """Dataset for evaluation with both EEG and BVP."""
    
    def __init__(self, eeg_X, bvp_X, labels, subjects):
        self.eeg_X = eeg_X
        self.bvp_X = bvp_X
        self.labels = labels
        self.subjects = subjects
        
        self.bvp_extractor = BVPHandcraftedFeatures(sampling_rate=64.0)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg_X[idx]).float()
        bvp_signal = torch.from_numpy(self.bvp_X[idx]).unsqueeze(-1).float()
        
        with torch.no_grad():
            bvp_feat = self.bvp_extractor(bvp_signal.unsqueeze(0)).squeeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return eeg, bvp_feat, label


# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate_with_and_without_bvp(model, dataloader, post_processor, device, label_map):
    """Evaluate model with and without BVP adjustment."""
    
    model.eval()
    
    # Storage for predictions
    all_labels = []
    eeg_only_preds = []
    eeg_bvp_preds = []
    
    with torch.no_grad():
        for eeg_x, bvp_feat, labels in dataloader:
            eeg_x = eeg_x.to(device)
            bvp_feat = bvp_feat.to(device)
            labels = labels.to(device)
            
            # Get EEG-only predictions
            eeg_logits = model(eeg_x)
            eeg_pred = eeg_logits.argmax(dim=1)
            
            # Get EEG+BVP predictions (with adjustment)
            adjusted_logits = post_processor.adjust_logits(eeg_logits, bvp_feat)
            bvp_pred = adjusted_logits.argmax(dim=1)
            
            # Store
            all_labels.extend(labels.cpu().numpy())
            eeg_only_preds.extend(eeg_pred.cpu().numpy())
            eeg_bvp_preds.extend(bvp_pred.cpu().numpy())
    
    # Convert to arrays
    all_labels = np.array(all_labels)
    eeg_only_preds = np.array(eeg_only_preds)
    eeg_bvp_preds = np.array(eeg_bvp_preds)
    
    # Compute metrics
    eeg_acc = (eeg_only_preds == all_labels).mean()
    eeg_f1 = f1_score(all_labels, eeg_only_preds, average='macro')
    
    bvp_acc = (eeg_bvp_preds == all_labels).mean()
    bvp_f1 = f1_score(all_labels, eeg_bvp_preds, average='macro')
    
    # Get label names
    id2lab = {v: k for k, v in label_map.items()}
    target_names = [id2lab[i] for i in range(len(label_map))]
    
    return {
        'labels': all_labels,
        'eeg_only_preds': eeg_only_preds,
        'eeg_bvp_preds': eeg_bvp_preds,
        'eeg_only_acc': eeg_acc,
        'eeg_only_f1': eeg_f1,
        'eeg_bvp_acc': bvp_acc,
        'eeg_bvp_f1': bvp_f1,
        'target_names': target_names
    }


# ==================================================
# MAIN
# ==================================================

def main(args):
    print("=" * 80)
    print("EVALUATE EEG MODEL WITH OPTIONAL BVP POST-PROCESSING")
    print("=" * 80)
    
    # Config
    config = Config()
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    print(f"\n📋 Configuration:")
    print(f"   EEG Model: {args.eeg_model}")
    print(f"   BVP Adjustment Strength: {args.adjustment_strength}")
    print(f"   Device: {config.DEVICE}")
    
    # Load EEG data
    print("\n" + "="*80)
    print("LOADING EEG DATA")
    print("="*80)
    
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(config.DATA_ROOT, config)
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    # Load BVP data
    print("\n" + "="*80)
    print("LOADING BVP DATA")
    print("="*80)
    
    bvp_config = BVPConfig()
    bvp_config.USE_BVP_BASELINE_REDUCTION = False
    bvp_config.USE_BVP_BASELINE_CORRECTION = False
    
    bvp_X_raw, bvp_y, bvp_subjects, bvp_label_map = load_bvp_data(config.DATA_ROOT, bvp_config)
    
    # Align datasets
    common_subjects = list(set(eeg_subjects) & set(bvp_subjects))
    eeg_mask = np.isin(eeg_subjects, common_subjects)
    bvp_mask = np.isin(bvp_subjects, common_subjects)
    
    eeg_X_features = eeg_X_features[eeg_mask]
    eeg_y = eeg_y[eeg_mask]
    eeg_subjects = eeg_subjects[eeg_mask]
    bvp_X_raw = bvp_X_raw[bvp_mask]
    
    print(f"✅ Aligned: {len(eeg_y)} samples from {len(common_subjects)} subjects")
    
    # Create splits
    split_indices = create_data_splits(eeg_y, eeg_subjects, config)
    test_idx = split_indices['test']
    train_idx = split_indices['train']
    
    # Get test data
    Xte = eeg_X_features[test_idx]
    yte = eeg_y[test_idx]
    bvp_te = bvp_X_raw[test_idx]
    subj_te = eeg_subjects[test_idx]
    
    # Standardize (using train statistics)
    Xtr = eeg_X_features[train_idx]
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xte = (Xte - mu) / sd
    
    # Create test dataset
    test_ds = EEGBVPEvalDataset(Xte, bvp_te, yte, subj_te)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    print(f"\n📊 Test set: {len(test_ds)} samples")
    
    # Load EEG model
    print("\n" + "="*80)
    print("LOADING EEG MODEL")
    print("="*80)
    
    model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3,
        n_classes=config.NUM_CLASSES, p_drop=0.4
    ).to(config.DEVICE)
    
    if os.path.exists(args.eeg_model):
        model.load_state_dict(torch.load(args.eeg_model, map_location=config.DEVICE))
        print(f"✅ Loaded model from {args.eeg_model}")
    else:
        print(f"⚠️  Model file not found: {args.eeg_model}")
        print(f"   Using randomly initialized model for demo")
    
    # Create BVP post-processor
    post_processor = BVPPostProcessor(adjustment_strength=args.adjustment_strength)
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    results = evaluate_with_and_without_bvp(
        model, test_loader, post_processor, config.DEVICE, eeg_label_map
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n📊 EEG-Only Performance:")
    print(f"   Accuracy: {results['eeg_only_acc']:.3f} ({results['eeg_only_acc']*100:.1f}%)")
    print(f"   Macro-F1: {results['eeg_only_f1']:.3f}")
    
    print(f"\n📊 EEG+BVP Performance:")
    print(f"   Accuracy: {results['eeg_bvp_acc']:.3f} ({results['eeg_bvp_acc']*100:.1f}%)")
    print(f"   Macro-F1: {results['eeg_bvp_f1']:.3f}")
    
    # Calculate improvement
    acc_improvement = results['eeg_bvp_acc'] - results['eeg_only_acc']
    f1_improvement = results['eeg_bvp_f1'] - results['eeg_only_f1']
    
    print(f"\n📈 Improvement with BVP:")
    print(f"   Accuracy: {acc_improvement:+.3f} ({acc_improvement*100:+.1f}%)")
    print(f"   Macro-F1: {f1_improvement:+.3f}")
    
    # Detailed reports
    print("\n" + "="*80)
    print("EEG-ONLY CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        results['labels'], 
        results['eeg_only_preds'],
        target_names=results['target_names'],
        digits=3, 
        zero_division=0
    ))
    
    print("\n" + "="*80)
    print("EEG+BVP CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        results['labels'], 
        results['eeg_bvp_preds'],
        target_names=results['target_names'],
        digits=3,
        zero_division=0
    ))
    
    # Show where predictions changed
    changed_mask = results['eeg_only_preds'] != results['eeg_bvp_preds']
    n_changed = changed_mask.sum()
    pct_changed = 100 * n_changed / len(changed_mask)
    
    print("\n" + "="*80)
    print("PREDICTION CHANGES")
    print("="*80)
    print(f"   BVP adjustment changed {n_changed}/{len(changed_mask)} predictions ({pct_changed:.1f}%)")
    
    # Show confusion matrices
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)
    
    cm_eeg = confusion_matrix(results['labels'], results['eeg_only_preds'])
    cm_bvp = confusion_matrix(results['labels'], results['eeg_bvp_preds'])
    
    print("\nEEG-Only:")
    print(cm_eeg)
    
    print("\nEEG+BVP:")
    print(cm_bvp)
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if f1_improvement > 0:
        print(f"   ✅ BVP post-processing IMPROVED F1 by {f1_improvement:.3f}")
        print(f"   💡 Consider using adjustment_strength={args.adjustment_strength}")
        print(f"   📝 Try tuning with values: 0.1, 0.2, 0.3, 0.5")
    else:
        print(f"   ⚠️  BVP post-processing did not improve performance")
        print(f"   💡 Try different adjustment_strength values")
        print(f"   📝 Current: {args.adjustment_strength}, try: 0.1, 0.05, 0.3")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate EEG model with optional BVP post-processing"
    )
    
    parser.add_argument(
        '--eeg_model', 
        type=str, 
        default='best_eeg_model.pt',
        help='Path to trained EEG model checkpoint'
    )
    parser.add_argument(
        '--adjustment_strength', 
        type=float, 
        default=0.2,
        help='BVP adjustment strength (0.0 = no adjustment, 1.0 = full adjustment)'
    )
    
    args = parser.parse_args()
    main(args)
