"""
BVP-Guided EEG Emotion Classification
======================================

This approach uses BVP handcrafted features to GUIDE the EEG model,
rather than simple concatenation.

Key idea: BVP features (heart rate, HRV) indicate arousal/stress level,
which can help the model know which EEG patterns to focus on.

Methods:
1. **BVP-Guided Channel Attention**: Use BVP to weight EEG channels
2. **BVP-Conditional Feature Learning**: Modulate EEG features based on BVP
3. **Multi-Task Learning**: Joint emotion + BVP feature prediction

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
# BVP-GUIDED ATTENTION MODULE
# ==================================================

class BVPGuidedChannelAttention(nn.Module):
    """
    Use BVP features to generate attention weights for EEG channels.
    
    Intuition: Different emotional states may have different brain-body
    coupling patterns. BVP features (arousal indicators) can help determine
    which EEG channels are most relevant for the current physiological state.
    """
    
    def __init__(self, bvp_dim=11, n_channels=4, hidden_dim=32):
        super().__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(bvp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_channels),
            nn.Softmax(dim=1)  # Normalize to get attention weights
        )
    
    def forward(self, eeg_features, bvp_features):
        """
        Args:
            eeg_features: [batch, channels, features]
            bvp_features: [batch, bvp_dim]
        
        Returns:
            weighted_eeg: [batch, channels, features] with channel attention applied
            attention_weights: [batch, channels] for interpretability
        """
        # Generate channel attention weights from BVP
        attention_weights = self.attention_net(bvp_features)  # [batch, channels]
        
        # Apply attention to each channel
        attention_weights = attention_weights.unsqueeze(2)  # [batch, channels, 1]
        weighted_eeg = eeg_features * attention_weights  # Broadcast multiplication
        
        return weighted_eeg, attention_weights.squeeze(2)


class BVPConditionedFeatureModulation(nn.Module):
    """
    Use BVP features to modulate EEG feature processing via FiLM
    (Feature-wise Linear Modulation).
    
    This allows BVP to adaptively scale and shift EEG features based on
    physiological state (arousal, stress, etc.).
    """
    
    def __init__(self, bvp_dim=11, feature_dim=512, hidden_dim=64):
        super().__init__()
        
        # Generate scale and shift parameters from BVP
        self.film_net = nn.Sequential(
            nn.Linear(bvp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim * 2)  # gamma and beta
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, eeg_repr, bvp_features):
        """
        Args:
            eeg_repr: [batch, feature_dim] - EEG representation from BiLSTM
            bvp_features: [batch, bvp_dim]
        
        Returns:
            modulated_repr: [batch, feature_dim] - Modulated EEG representation
        """
        # Generate FiLM parameters
        film_params = self.film_net(bvp_features)  # [batch, feature_dim * 2]
        
        # Split into scale (gamma) and shift (beta)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # Apply FiLM: y = gamma * x + beta
        modulated_repr = gamma * eeg_repr + beta
        
        return modulated_repr


# ==================================================
# BVP-GUIDED EEG MODEL
# ==================================================

class BVPGuidedEEGModel(nn.Module):
    """
    EEG emotion classifier guided by BVP features.
    
    Architecture:
    1. BVP features → Channel attention weights
    2. Apply attention to EEG channels
    3. EEG → BiLSTM → representation
    4. BVP features → FiLM modulation of EEG representation
    5. Classify modulated representation
    
    Optional: Multi-task learning to predict BVP features from EEG
    """
    
    def __init__(
        self, 
        n_classes=4, 
        eeg_hidden=256, 
        eeg_layers=3, 
        dropout=0.4,
        bvp_dim=11,
        use_channel_attention=True,
        use_feature_modulation=True,
        use_multitask=False
    ):
        super().__init__()
        
        self.use_channel_attention = use_channel_attention
        self.use_feature_modulation = use_feature_modulation
        self.use_multitask = use_multitask
        
        # BVP-guided channel attention (optional)
        if use_channel_attention:
            self.channel_attention = BVPGuidedChannelAttention(
                bvp_dim=bvp_dim,
                n_channels=4,
                hidden_dim=32
            )
        
        # EEG encoder (BiLSTM)
        self.eeg_encoder = SimpleBiLSTMClassifier(
            dx=26, 
            n_channels=4, 
            hidden=eeg_hidden, 
            layers=eeg_layers,
            n_classes=n_classes,
            p_drop=dropout
        )
        
        eeg_repr_dim = eeg_hidden * 2  # 512 for bidirectional
        
        # BVP-conditioned feature modulation (optional)
        if use_feature_modulation:
            self.feature_modulation = BVPConditionedFeatureModulation(
                bvp_dim=bvp_dim,
                feature_dim=eeg_repr_dim,
                hidden_dim=64
            )
        
        # Main emotion classifier
        self.classifier = nn.Sequential(
            nn.Linear(eeg_repr_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
        # Multi-task: Auxiliary head to predict BVP features from EEG
        if use_multitask:
            self.bvp_predictor = nn.Sequential(
                nn.Linear(eeg_repr_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, bvp_dim)
            )
    
    def forward(self, eeg_x, bvp_features):
        """
        Forward pass with BVP guidance.
        
        Args:
            eeg_x: [batch, channels, features] - EEG features
            bvp_features: [batch, bvp_dim] - BVP handcrafted features
        
        Returns:
            emotion_logits: [batch, n_classes]
            aux_outputs: dict with auxiliary outputs (attention, predictions, etc.)
        """
        aux_outputs = {}
        
        # Step 1: BVP-guided channel attention (optional)
        if self.use_channel_attention:
            eeg_x, channel_attn = self.channel_attention(eeg_x, bvp_features)
            aux_outputs['channel_attention'] = channel_attn
        
        # Step 2: Extract EEG representation via BiLSTM
        B, C, dx = eeg_x.shape
        x = self.eeg_encoder.input_proj(eeg_x)
        h, _ = self.eeg_encoder.lstm(x)
        h = self.eeg_encoder.drop(self.eeg_encoder.norm(h))
        
        # Attention pooling
        scores = self.eeg_encoder.attn(h)
        alpha = torch.softmax(scores, dim=1)
        eeg_repr = (alpha * h).sum(dim=1)  # [batch, 512]
        
        # Step 3: BVP-conditioned feature modulation (optional)
        if self.use_feature_modulation:
            eeg_repr = self.feature_modulation(eeg_repr, bvp_features)
            aux_outputs['modulated_repr'] = eeg_repr
        
        # Step 4: Classify emotion
        emotion_logits = self.classifier(eeg_repr)
        
        # Step 5: Multi-task BVP prediction (optional)
        if self.use_multitask:
            bvp_pred = self.bvp_predictor(eeg_repr)
            aux_outputs['bvp_prediction'] = bvp_pred
        
        return emotion_logits, aux_outputs


# ==================================================
# DATASET
# ==================================================

class BVPGuidedDataset(Dataset):
    """Dataset for BVP-guided EEG model."""
    
    def __init__(self, eeg_X, bvp_X, labels, subjects):
        self.eeg_X = eeg_X
        self.bvp_X = bvp_X
        self.labels = labels
        self.subjects = subjects
        
        # Initialize BVP feature extractor
        self.bvp_feature_extractor = BVPHandcraftedFeatures(
            sampling_rate=64.0,
            min_peak_distance=20
        )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg_sample = torch.from_numpy(self.eeg_X[idx]).float()
        bvp_signal = torch.from_numpy(self.bvp_X[idx]).unsqueeze(-1).float()
        
        # Extract BVP features
        with torch.no_grad():
            bvp_features = self.bvp_feature_extractor(bvp_signal.unsqueeze(0))
            bvp_features = bvp_features.squeeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return eeg_sample, bvp_features, label


# ==================================================
# MAIN PIPELINE
# ==================================================

def main(args):
    print("=" * 80)
    print("BVP-GUIDED EEG EMOTION CLASSIFICATION")
    print("=" * 80)
    
    # Configuration
    config = Config()
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    print(f"\n📋 Configuration:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Channel Attention: {args.use_channel_attention}")
    print(f"   Feature Modulation: {args.use_feature_modulation}")
    print(f"   Multi-Task Learning: {args.use_multitask}")
    
    # Load data (same as simple approach)
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(config.DATA_ROOT, config)
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    bvp_config = BVPConfig()
    bvp_config.USE_BVP_BASELINE_REDUCTION = False
    bvp_config.USE_BVP_BASELINE_CORRECTION = False
    bvp_X_raw, bvp_y, bvp_subjects, bvp_label_map = load_bvp_data(bvp_config.DATA_ROOT, bvp_config)
    
    # Align datasets
    common_subjects = list(set(eeg_subjects) & set(bvp_subjects))
    eeg_mask = np.isin(eeg_subjects, common_subjects)
    bvp_mask = np.isin(bvp_subjects, common_subjects)
    
    eeg_X_features = eeg_X_features[eeg_mask]
    eeg_y = eeg_y[eeg_mask]
    eeg_subjects = eeg_subjects[eeg_mask]
    bvp_X_raw = bvp_X_raw[bvp_mask]
    
    print(f"✅ Aligned data: {eeg_X_features.shape[0]} samples")
    
    # Create splits
    split_indices = create_data_splits(eeg_y, eeg_subjects, config)
    train_idx, val_idx, test_idx = split_indices['train'], split_indices['val'], split_indices['test']
    
    # Get splits
    Xtr, Xva, Xte = eeg_X_features[train_idx], eeg_X_features[val_idx], eeg_X_features[test_idx]
    ytr, yva, yte = eeg_y[train_idx], eeg_y[val_idx], eeg_y[test_idx]
    bvp_tr, bvp_va, bvp_te = bvp_X_raw[train_idx], bvp_X_raw[val_idx], bvp_X_raw[test_idx]
    subj_tr, subj_va, subj_te = eeg_subjects[train_idx], eeg_subjects[val_idx], eeg_subjects[test_idx]
    
    # Standardize EEG
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    
    # Create datasets
    tr_ds = BVPGuidedDataset(Xtr, bvp_tr, ytr, subj_tr)
    va_ds = BVPGuidedDataset(Xva, bvp_va, yva, subj_va)
    te_ds = BVPGuidedDataset(Xte, bvp_te, yte, subj_te)
    
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
    
    # Create model
    print("\n" + "="*80)
    print("CREATING BVP-GUIDED MODEL")
    print("="*80)
    
    model = BVPGuidedEEGModel(
        n_classes=config.NUM_CLASSES,
        eeg_hidden=256,
        eeg_layers=3,
        dropout=0.4,
        bvp_dim=11,
        use_channel_attention=args.use_channel_attention,
        use_feature_modulation=args.use_feature_modulation,
        use_multitask=args.use_multitask
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Training setup
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Multi-task loss for BVP prediction (MSE)
    if args.use_multitask:
        bvp_criterion = nn.MSELoss()
        bvp_loss_weight = args.multitask_weight
    
    best_f1, best_state, wait = 0.0, None, 0
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for eeg_xb, bvp_fb, yb in tr_loader:
            eeg_xb = eeg_xb.to(config.DEVICE)
            bvp_fb = bvp_fb.to(config.DEVICE)
            yb = yb.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            emotion_logits, aux_outputs = model(eeg_xb, bvp_fb)
            
            # Main emotion classification loss
            emotion_loss = criterion(emotion_logits, yb)
            
            # Multi-task BVP prediction loss (optional)
            if args.use_multitask and 'bvp_prediction' in aux_outputs:
                bvp_pred = aux_outputs['bvp_prediction']
                bvp_loss = bvp_criterion(bvp_pred, bvp_fb)
                total_loss = emotion_loss + bvp_loss_weight * bvp_loss
            else:
                total_loss = emotion_loss
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
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
                
                emotion_logits, _ = model(eeg_xb, bvp_fb)
                preds = emotion_logits.argmax(dim=1)
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
    
    # Test evaluation
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
            
            emotion_logits, _ = model(eeg_xb, bvp_fb)
            preds = emotion_logits.argmax(dim=1)
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
    print("🎉 BVP-GUIDED PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"✅ Best Val F1: {best_f1:.3f}")
    print(f"✅ Test F1: {test_f1:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVP-Guided EEG Emotion Classification")
    
    parser.add_argument('--bvp_data_root', type=str,
                        default="/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined",
                        help='BVP data root directory')
    parser.add_argument('--checkpoint', type=str, default='best_bvp_guided_model.pt',
                        help='Output checkpoint path')
    
    # BVP guidance options
    parser.add_argument('--use_channel_attention', action='store_true', default=True,
                        help='Use BVP to guide EEG channel attention')
    parser.add_argument('--use_feature_modulation', action='store_true', default=True,
                        help='Use BVP to modulate EEG features (FiLM)')
    parser.add_argument('--use_multitask', action='store_true',
                        help='Use multi-task learning to predict BVP from EEG')
    parser.add_argument('--multitask_weight', type=float, default=0.1,
                        help='Weight for multi-task BVP prediction loss')
    
    args = parser.parse_args()
    main(args)
