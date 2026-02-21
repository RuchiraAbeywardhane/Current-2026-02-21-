"""
    Preprocessed EEG-Only Emotion Recognition (Standalone Test)
    ============================================================
    
    This is a 100% EXACT copy of the EEG training code from BR_WithPP_FourEmotionsEEG.py
    Used to verify that preprocessed EEG training works identically before adding fusion.
    
    Dataset: /kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR
    
    Author: Final Year Project
    Date: 2026
"""

import os
import glob
import json
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, find_peaks, medfilt


# ==================================================
# CONFIGURATION (100% SAME AS BR_WithPP_FourEmotionsEEG.py)
# ==================================================

class Config:
    """Shared configuration for all models."""
    # Paths
    DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"  # Preprocessed dataset
    
    # Common parameters
    NUM_CLASSES = 4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Classification Mode
    USE_DUAL_BINARY = False
    CLASSIFY_WHOLE_CLIPS = False
    CLIP_AGGREGATION_METHOD = "mean"
    
    # Baseline reduction (InvBase method)
    USE_BASELINE_REDUCTION = True
    
    # Data split mode
    SUBJECT_INDEPENDENT = True
    CLIP_INDEPENDENT = False
    
    # LOSO Cross-Validation
    USE_LOSO = False
    LOSO_SAVE_ALL_FOLDS = True
    
    # Stratified split parameters
    USE_STRATIFIED_GROUP_SPLIT = True
    MIN_SAMPLES_PER_CLASS = 10
    
    # Label mappings (4-class system)
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",
        "FEAR": "Q2",
        "SADNESS": "Q3",
        "NEUTRAL": "Q4",
    }
    
    SUPERCLASS_ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    IDX_TO_LABEL = ["Q1_Positive_Active", "Q2_Negative_Active", "Q3_Negative_Calm", "Q4_Positive_Calm"]
    
    # Dual Binary Classification Mappings
    AROUSAL_MAP = {0: 1, 1: 1, 2: 0, 3: 0}
    VALENCE_MAP = {0: 1, 1: 0, 2: 0, 3: 1}
    AROUSAL_LABELS = ["Low_Arousal", "High_Arousal"]
    VALENCE_LABELS = ["Negative_Valence", "Positive_Valence"]
    
    # EEG parameters
    EEG_FS = 256.0
    EEG_CHANNELS = 4
    EEG_FEATURES = 26
    EEG_WINDOW_SEC = 10.0
    EEG_OVERLAP = 0.5 if CLIP_INDEPENDENT else 0.0
    EEG_BATCH_SIZE = 32 if CLIP_INDEPENDENT else 64
    EEG_EPOCHS = 200 if CLIP_INDEPENDENT else 150
    EEG_LR = 5e-4 if CLIP_INDEPENDENT else 1e-3
    EEG_PATIENCE = 30 if CLIP_INDEPENDENT else 20
    EEG_CHECKPOINT = "best_eeg_pp_only.pt"
    
    # Augmentation settings
    USE_MIXUP = CLIP_INDEPENDENT
    USE_LABEL_SMOOTHING = CLIP_INDEPENDENT
    LABEL_SMOOTHING = 0.1 if CLIP_INDEPENDENT else 0.0
    
    # File outputs
    SPLIT_FILE = "data_split_indices.npz"
    
    # Frequency bands for EEG
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
# PART 1: DATA LOADING & PREPROCESSING
# ==================================================

def _to_num(x):
    """Convert to numeric array."""
    if isinstance(x, list):
        if not x:
            return np.array([], np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, np.float64)
    return np.asarray([x], np.float64)


def _interp_nan(a):
    """Interpolate NaN values."""
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def load_preprocessed_eeg_data(data_root):
    """Load preprocessed EEG data from JSON files."""
    print("\n" + "="*80)
    print("LOADING PREPROCESSED EEG DATA")
    print("="*80)
    
    # Search for preprocessed JSON files
    patterns = [
        os.path.join(data_root, "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned", "*_STIMULUS_MUSE_cleaned.json")
    ]
    
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    print(f"Found {len(files)} preprocessed files")
    
    if len(files) == 0:
        print("\n❌ ERROR: No preprocessed files found!")
        print(f"   Searched in: {data_root}")
        print(f"\n   Expected pattern: *_STIMULUS_MUSE_cleaned.json")
        
        # Debug: show what's in the directory
        if os.path.exists(data_root):
            print(f"\n   Directory exists. Contents:")
            contents = os.listdir(data_root)[:10]
            for item in contents:
                print(f"      {item}")
            
            # Check subdirectories
            subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
            if subdirs:
                print(f"\n   Found {len(subdirs)} subdirectories. First one:")
                first_subdir = os.path.join(data_root, subdirs[0])
                subdir_contents = os.listdir(first_subdir)[:10]
                for item in subdir_contents:
                    print(f"      {item}")
        else:
            print(f"   ❌ Directory does not exist!")
        
        raise ValueError("No preprocessed files found. Please check DATA_ROOT path.")
    
    print(f"\n📁 Sample files:")
    for f in files[:3]:
        print(f"   {os.path.basename(f)}")
    
    all_windows = []
    all_labels = []
    all_subjects = []
    
    win_samples = int(config.EEG_WINDOW_SEC * config.EEG_FS)
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))
    
    skipped_reasons = {
        'baseline_file': 0,
        'unknown_emotion': 0,
        'no_data': 0,
        'insufficient_length': 0,
        'parse_error': 0
    }
    
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        
        if len(parts) < 2:
            skipped_reasons['parse_error'] += 1
            continue
        
        # Skip baseline files
        if "BASELINE" in fname:
            skipped_reasons['baseline_file'] += 1
            continue
        
        subject = parts[0]
        emotion = parts[1].upper()
        
        if emotion not in config.SUPERCLASS_MAP:
            skipped_reasons['unknown_emotion'] += 1
            continue
        
        superclass = config.SUPERCLASS_MAP[emotion]
        label_id = config.SUPERCLASS_ID[superclass]
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Extract 4 EEG channels
            tp9 = _interp_nan(_to_num(data.get("RAW_TP9", [])))
            af7 = _interp_nan(_to_num(data.get("RAW_AF7", [])))
            af8 = _interp_nan(_to_num(data.get("RAW_AF8", [])))
            tp10 = _interp_nan(_to_num(data.get("RAW_TP10", [])))
            
            L = min(len(tp9), len(af7), len(af8), len(tp10))
            
            if L == 0:
                skipped_reasons['no_data'] += 1
                continue
            
            if L < win_samples:
                skipped_reasons['insufficient_length'] += 1
                continue
            
            # Stack channels
            signal = np.stack([tp9[:L], af7[:L], af8[:L], tp10[:L]], axis=1)  # (T, 4)
            signal = signal - np.nanmean(signal, axis=0, keepdims=True)
            
            # Create sliding windows
            for start in range(0, L - win_samples + 1, step_samples):
                window = signal[start:start + win_samples]
                if len(window) == win_samples:
                    all_windows.append(window)
                    all_labels.append(label_id)
                    all_subjects.append(subject)
        
        except Exception as e:
            skipped_reasons['parse_error'] += 1
            print(f"   ⚠️ Error processing {fname}: {e}")
            continue
    
    # Print statistics
    print(f"\n📊 File Processing Summary:")
    print(f"   Total files found: {len(files)}")
    print(f"   Successfully processed: {len(all_windows)} windows")
    print(f"\n   Skipped files:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"      {reason}: {count}")
    
    if len(all_windows) == 0:
        print("\n❌ ERROR: No valid EEG windows extracted!")
        print("\n💡 Troubleshooting:")
        print("   1. Check that files contain RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10 fields")
        print("   2. Verify emotion names match SUPERCLASS_MAP")
        print("   3. Ensure signals are long enough (≥10 seconds)")
        print(f"\n   Expected emotions: {list(config.SUPERCLASS_MAP.keys())}")
        raise ValueError("No valid EEG data extracted.")
    
    X_raw = np.stack(all_windows).astype(np.float32)
    y_labels = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects)
    
    print(f"\n✅ EEG data loaded: {X_raw.shape}")
    print(f"   Subjects: {len(set(subjects))}")
    print(f"   Label distribution: {Counter(all_labels)}")
    
    return X_raw, y_labels, subjects


def extract_eeg_features(X_raw, fs=256.0, eps=1e-12):
    """Extract 26 features per channel from EEG windows."""
    print("Extracting EEG features (26 per channel)...")
    N, T, C = X_raw.shape
    
    P = (np.abs(np.fft.rfft(X_raw, axis=1))**2) / T
    freqs = np.fft.rfftfreq(T, d=1/fs)
    
    feature_list = []
    
    # 1) Differential Entropy (5)
    de_feats = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * (bp + eps))
        de_feats.append(de[..., None])
    de_all = np.concatenate(de_feats, axis=2)
    feature_list.append(de_all)
    
    # 2) Log-PSD (5)
    psd_feats = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        log_psd = np.log(bp + eps)
        psd_feats.append(log_psd[..., None])
    psd_all = np.concatenate(psd_feats, axis=2)
    feature_list.append(psd_all)
    
    # 3) Temporal stats (4)
    temp_mean = X_raw.mean(axis=1)[..., None]
    temp_std = X_raw.std(axis=1)[..., None]
    temp_skew = skew(X_raw, axis=1)[..., None]
    temp_kurt = kurtosis(X_raw, axis=1)[..., None]
    temp_all = np.concatenate([temp_mean, temp_std, temp_skew, temp_kurt], axis=2)
    feature_list.append(temp_all)
    
    # 4) DE asymmetry (5)
    de_left = (de_all[:, 0, :] + de_all[:, 1, :]) / 2
    de_right = (de_all[:, 2, :] + de_all[:, 3, :]) / 2
    de_asym = de_left - de_right
    de_asym_full = np.tile(de_asym[:, None, :], (1, C, 1))
    feature_list.append(de_asym_full)
    
    # 5) Bandpower ratios (3)
    band_bp = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        band_bp.append(bp)
    _, theta_bp, alpha_bp, beta_bp, gamma_bp = band_bp
    
    ratio_theta_alpha = (theta_bp + eps) / (alpha_bp + eps)
    ratio_beta_alpha = (beta_bp + eps) / (alpha_bp + eps)
    ratio_gamma_beta = (gamma_bp + eps) / (beta_bp + eps)
    ratio_all = np.stack([ratio_theta_alpha, ratio_beta_alpha, ratio_gamma_beta], axis=2)
    feature_list.append(ratio_all)
    
    # 6) Hjorth parameters (2)
    Xc = X_raw - X_raw.mean(axis=1, keepdims=True)
    dx = np.diff(Xc, axis=1)
    var_x = (Xc**2).mean(axis=1) + eps
    var_dx = (dx**2).mean(axis=1) + eps
    mobility = np.sqrt(var_dx / var_x)
    ddx = np.diff(dx, axis=1)
    var_ddx = (ddx**2).mean(axis=1) + eps
    mobility_dx = np.sqrt(var_ddx / var_dx)
    complexity = mobility_dx / (mobility + eps)
    hjorth_all = np.stack([mobility, complexity], axis=2)
    feature_list.append(hjorth_all)
    
    # 7) Time-domain extras (2)
    log_var = np.log(var_x + eps)
    sign_x = np.sign(Xc)
    zc = (np.diff(sign_x, axis=1) != 0).sum(axis=1) / float(T - 1 + eps)
    td_extras = np.stack([log_var, zc], axis=2)
    feature_list.append(td_extras)
    
    features = np.concatenate(feature_list, axis=2)
    print(f"EEG features extracted: {features.shape}")
    return features.astype(np.float32)


# ==================================================
# PART 2: MODEL ARCHITECTURE
# ==================================================

class SimpleBiLSTMClassifier(nn.Module):
    """3-layer BiLSTM with attention for EEG."""
    
    def __init__(self, dx=26, n_channels=4, hidden=256, layers=3, n_classes=4, p_drop=0.4):
        super().__init__()
        self.n_channels = n_channels
        self.hidden = hidden
        
        self.input_proj = nn.Sequential(
            nn.Linear(dx, hidden),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
            nn.Dropout(p_drop * 0.5)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=p_drop if layers > 1 else 0
        )
        
        d_lstm = 2 * hidden
        self.norm = nn.LayerNorm(d_lstm)
        self.drop = nn.Dropout(p_drop)

        self.attn = nn.Sequential(
            nn.Linear(d_lstm, d_lstm // 2),
            nn.Tanh(),
            nn.Linear(d_lstm // 2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_lstm, d_lstm),
            nn.BatchNorm1d(d_lstm),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_lstm, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B, C, dx = x.shape
        x = self.input_proj(x)
        h, _ = self.lstm(x)
        h = self.drop(self.norm(h))

        scores = self.attn(h)
        alpha = torch.softmax(scores, dim=1)
        h_pooled = (alpha * h).sum(dim=1)

        return self.classifier(h_pooled)


# ==================================================
# PART 3: TRAINING FUNCTIONS
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


def train_eeg_model(X_features, y_labels, split_indices, label_mapping):
    """Train EEG BiLSTM model."""
    print("\n" + "="*80)
    print("TRAINING EEG MODEL")
    print("="*80)
    
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    Xtr, Xva, Xte = X_features[train_idx], X_features[val_idx], X_features[test_idx]
    ytr, yva, yte = y_labels[train_idx], y_labels[val_idx], y_labels[test_idx]
    
    # Standardization
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    
    print(f"Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
    
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
    
    # Data loaders
    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    
    tr_loader = DataLoader(tr_ds, batch_size=config.EEG_BATCH_SIZE, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    
    # Model
    model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3,
        n_classes=config.NUM_CLASSES, p_drop=0.4
    ).to(config.DEVICE)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_f1, best_state, wait = 0.0, None, 0
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for xb, yb in tr_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            
            if np.random.rand() < 0.5:
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
                print(f"Early stopping at epoch {epoch}")
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
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"Test Macro-F1: {test_f1:.3f}")
    id2lab = {v: k for k, v in label_mapping.items()}
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    return model, mu, sd


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """EEG-only training pipeline."""
    print("=" * 80)
    print("PREPROCESSED EEG-ONLY EMOTION RECOGNITION")
    print("=" * 80)
    print(f"Dataset: {config.DATA_ROOT}")
    print(f"Mode: {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Baseline Reduction: {config.USE_BASELINE_REDUCTION}")
    print("=" * 80)
    
    # Step 1: Load data
    eeg_X_raw, eeg_y, eeg_subjects = load_preprocessed_eeg_data(config.DATA_ROOT)
    eeg_X_features = extract_eeg_features(eeg_X_raw)
    
    # Step 2: Create splits
    print("\n" + "="*80)
    print("CREATING EEG DATA SPLIT")
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
    
    eeg_split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    print(f"\n📋 Split Summary:")
    print(f"   Train samples: {len(eeg_split_indices['train'])}")
    print(f"   Val samples: {len(eeg_split_indices['val'])}")
    print(f"   Test samples: {len(eeg_split_indices['test'])}")
    
    # Step 3: Train EEG model
    eeg_model, eeg_mu, eeg_sd = train_eeg_model(eeg_X_features, eeg_y, eeg_split_indices, config.SUPERCLASS_ID)
    
    print("\n" + "=" * 80)
    print("🎉 EEG TRAINING COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {config.EEG_CHECKPOINT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
