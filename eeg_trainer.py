"""
    EEG Model Training Utilities
    =============================
    
    This module contains training and evaluation functions for EEG emotion recognition.
    
    Features:
    - Mixup data augmentation
    - Training loop with early stopping
    - Validation and test evaluation
    - Detailed performance metrics
    
    Author: Final Year Project
    Date: 2026
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from eeg_model import SimpleBiLSTMClassifier


# ==================================================
# DATA AUGMENTATION
# ==================================================

def mixup_data(x, y, alpha=0.2):
    """
    Mixup data augmentation.
    
    Args:
        x: Input tensor (batch_size, n_channels, n_features)
        y: Target labels (batch_size,)
        alpha: Mixup interpolation strength
    
    Returns:
        mixed_x: Mixed input tensor
        y_a, y_b: Original and permuted labels
        lam: Mixing coefficient
    """
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
    """
    Mixup loss calculation.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Original and permuted labels
        lam: Mixing coefficient
    
    Returns:
        loss: Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================================================
# TRAINING FUNCTIONS
# ==================================================

def train_eeg_model(loaders, label_mapping, config, stats=None):
    """
    Train EEG BiLSTM model using prepared data loaders.
    
    Args:
        loaders: dict with 'train', 'val', 'test' DataLoaders
        label_mapping: dict mapping class names to indices
        config: Config object with training settings
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
