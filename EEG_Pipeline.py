"""
    EEG-Only Emotion Recognition Pipeline
    ======================================
    
    This script contains a complete EEG emotion recognition pipeline with:
    - Preprocessed EEG data loading (MUSE headband)
    - Baseline reduction (InvBase method)
    - Feature extraction (26 features per channel)
    - BiLSTM classifier with attention
    - Subject-independent or subject-dependent splits
    
    Author: Final Year Project
    Date: 2026
"""

import os
import random

import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import f1_score, classification_report

# Import data loading utilities
from eeg_dataloader import load_eeg_data, extract_eeg_features, prepare_eeg_dataloaders

# Import model architecture
from eeg_model import SimpleBiLSTMClassifier


# ==================================================
# CONFIGURATION
# ==================================================

class Config:
    """EEG-specific configuration."""
    # Paths
    DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"
    
    # Common parameters
    NUM_CLASSES = 4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Baseline reduction (InvBase method)
    USE_BASELINE_REDUCTION = True
    
    # Data split mode
    SUBJECT_INDEPENDENT = True
    CLIP_INDEPENDENT = False
    
    # Stratified split parameters
    USE_STRATIFIED_GROUP_SPLIT = True
    MIN_SAMPLES_PER_CLASS = 10
    
    # Label mappings (4-class emotion quadrants)
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",  # Positive + High Arousal
        "FEAR": "Q2",         # Negative + High Arousal
        "SADNESS": "Q3",      # Negative + Low Arousal
        "NEUTRAL": "Q4",      # Positive + Low Arousal
    }
    
    SUPERCLASS_ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    IDX_TO_LABEL = ["Q1_Positive_Active", "Q2_Negative_Active", 
                    "Q3_Negative_Calm", "Q4_Positive_Calm"]
    
    # EEG parameters
    EEG_FS = 256.0
    EEG_CHANNELS = 4  # TP9, AF7, AF8, TP10
    EEG_FEATURES = 26
    EEG_WINDOW_SEC = 10.0
    EEG_OVERLAP = 0.5 if CLIP_INDEPENDENT else 0.0
    EEG_BATCH_SIZE = 32 if CLIP_INDEPENDENT else 64
    EEG_EPOCHS = 200 if CLIP_INDEPENDENT else 150
    EEG_LR = 5e-4 if CLIP_INDEPENDENT else 1e-3
    EEG_PATIENCE = 30 if CLIP_INDEPENDENT else 20
    EEG_CHECKPOINT = "best_eeg_model.pt"
    
    # Augmentation settings
    USE_MIXUP = CLIP_INDEPENDENT
    LABEL_SMOOTHING = 0.1 if CLIP_INDEPENDENT else 0.0
    
    # Frequency bands for feature extraction
    BANDS = [("delta", (1, 3)), ("theta", (4, 7)), ("alpha", (8, 13)), 
             ("beta", (14, 30)), ("gamma", (31, 45))]


# Global config instance
config = Config()

# Set random seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

print(f"Device: {config.DEVICE}")


# ==================================================
# MODEL ARCHITECTURE
# ==================================================


# ==================================================
# TRAINING FUNCTIONS
# ==================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_eeg_model(loaders, label_mapping, stats=None):
    """
    Train EEG BiLSTM model using prepared data loaders.
    
    Args:
        loaders: dict with 'train', 'val', 'test' DataLoaders
        label_mapping: dict mapping class names to indices
        stats: dict with 'mu' and 'sd' (optional, for reference)
    
    Returns:
        model: trained model
        loaders: same loaders (for convenience)
        stats: standardization statistics
    """
    print("\n" + "="*80)
    print("TRAINING EEG MODEL")
    print("="*80)
    
    tr_loader = loaders['train']
    va_loader = loaders['val']
    te_loader = loaders['test']
    
    # Model
    model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3,
        n_classes=config.NUM_CLASSES, p_drop=0.4
    ).to(config.DEVICE)
    
    print(f"🧠 Model: SimpleBiLSTMClassifier")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Get class weights from first batch
    sample_batch_y = next(iter(tr_loader))[1].numpy()
    class_counts = np.bincount(sample_batch_y, minlength=config.NUM_CLASSES).astype(np.float32)
    class_counts = np.clip(class_counts, 1.0, None)
    class_weights = 1.0 / class_counts
    class_weights = torch.from_numpy(class_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_f1, best_state, wait = 0.0, None, 0
    
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {config.EEG_EPOCHS}")
    print(f"   Learning rate: {config.EEG_LR}")
    print(f"   Early stopping patience: {config.EEG_PATIENCE}")
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for xb, yb in tr_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            
            if config.USE_MIXUP and np.random.rand() < 0.5:
                xb_mix, ya, yb_m, lam = mixup_data(xb, yb, alpha=0.2)
                optimizer.zero_grad()
                logits = model(xb_mix)
                loss = mixup_criterion(criterion, logits, ya, yb_m, lam)
            else:
                optimizer.zero_grad()
                logits = model(xb)
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
            for xb, yb in va_loader:
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
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(model.state_dict(), config.EEG_CHECKPOINT)
        else:
            wait += 1
            if wait >= config.EEG_PATIENCE:
                print(f"⏸️  Early stopping at epoch {epoch}")
                break
    
    # Test evaluation
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_acc = (all_preds == all_targets).mean()
    test_f1 = f1_score(all_targets, all_preds, average='macro')
    
    print("\n" + "="*80)
    print("EEG TEST RESULTS")
    print("="*80)
    print(f"✅ Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"✅ Test Macro-F1: {test_f1:.3f}")
    id2lab = {v: k for k, v in label_mapping.items()}
    print("\n📊 Classification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    return model, loaders, stats


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """EEG-only emotion recognition pipeline."""
    print("=" * 80)
    print("EEG-ONLY EMOTION RECOGNITION PIPELINE")
    print("=" * 80)
    print(f"Dataset: {config.DATA_ROOT}")
    print(f"Mode: {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Baseline Reduction: {config.USE_BASELINE_REDUCTION}")
    print("=" * 80)
    
    # Step 1: Load EEG data
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(config.DATA_ROOT, config)
    
    # Step 2: Extract features
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    # Step 3: Create data splits
    print("\n" + "="*80)
    print("CREATING DATA SPLIT")
    print("="*80)
    
    if config.SUBJECT_INDEPENDENT:
        print("  Strategy: SUBJECT-INDEPENDENT split")
        unique_subjects = np.unique(eeg_subjects)
        np.random.shuffle(unique_subjects)
        
        n_test = int(len(unique_subjects) * 0.15)
        n_val = int(len(unique_subjects) * 0.15)
        
        test_subjects = unique_subjects[:n_test]
        val_subjects = unique_subjects[n_test:n_test+n_val]
        train_subjects = unique_subjects[n_test+n_val:]
        
        train_mask = np.isin(eeg_subjects, train_subjects)
        val_mask = np.isin(eeg_subjects, val_subjects)
        test_mask = np.isin(eeg_subjects, test_subjects)
    else:
        print("  Strategy: RANDOM split")
        n_samples = len(eeg_y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_test = int(n_samples * 0.15)
        n_val = int(n_samples * 0.15)
        
        train_mask = np.zeros(n_samples, dtype=bool)
        val_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)
        
        test_mask[indices[:n_test]] = True
        val_mask[indices[n_test:n_test+n_val]] = True
        train_mask[indices[n_test+n_val:]] = True
    
    split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    print(f"\n📋 Split Summary:")
    print(f"   Train samples: {len(split_indices['train'])}")
    print(f"   Val samples: {len(split_indices['val'])}")
    print(f"   Test samples: {len(split_indices['test'])}")
    
    # Step 4: Prepare data loaders
    loaders, stats = prepare_eeg_dataloaders(eeg_X_features, eeg_y, split_indices, config)
    
    # Step 5: Train EEG model
    eeg_model, loaders, stats = train_eeg_model(loaders, eeg_label_map, stats)
    
    print("\n" + "=" * 80)
    print("🎉 EEG PIPELINE COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {config.EEG_CHECKPOINT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
