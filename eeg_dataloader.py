"""
    EEG Data Loading and Preprocessing Utilities
    =============================================
    
    This module contains functions for:
    - Loading preprocessed EEG data from MUSE headband
    - Baseline reduction (InvBase method)
    - Feature extraction (26 features per channel)
    - Creating PyTorch DataLoaders with balanced sampling
    
    Author: Final Year Project
    Date: 2026
"""

import os
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from scipy.stats import skew, kurtosis


# ==================================================
# DATA LOADING & PREPROCESSING
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


def apply_baseline_reduction(signal, baseline, eps=1e-12):
    """
    Apply InvBase method: divide trial FFT by baseline FFT.
    
    This reduces inter-subject variability by normalizing against 
    each subject's resting state baseline.
    
    Args:
        signal: (T, C) - trial signal
        baseline: (T, C) - baseline signal (same length)
        eps: small constant to prevent division by zero
    
    Returns:
        reduced_signal: (T, C) - baseline-reduced signal in time domain
    """
    # Compute FFT for each channel
    FFT_trial = np.fft.rfft(signal, axis=0)
    FFT_baseline = np.fft.rfft(baseline, axis=0)
    
    # InvBase: divide trial by baseline (element-wise per channel)
    FFT_reduced = FFT_trial / (np.abs(FFT_baseline) + eps)
    
    # Convert back to time domain
    signal_reduced = np.fft.irfft(FFT_reduced, n=len(signal), axis=0)
    
    return signal_reduced.astype(np.float32)


def load_eeg_data(data_root, config):
    """Load EEG data from MUSE files with optional baseline reduction."""
    print("\n" + "="*80)
    print("LOADING EEG DATA (MUSE)")
    print("="*80)
    
    # Search for preprocessed JSON files
    patterns = [
        os.path.join(data_root, "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned", "*_STIMULUS_MUSE_cleaned.json")
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    print(f"Found {len(files)} MUSE files")
    
    if len(files) == 0:
        print("\n❌ ERROR: No MUSE files found!")
        print(f"   Searched in: {data_root}")
        if os.path.exists(data_root):
            print(f"   Directory contents: {os.listdir(data_root)[:10]}")
        raise ValueError("No MUSE files found. Check DATA_ROOT path.")
    
    print(f"\n📁 Sample files:")
    for f in files[:3]:
        print(f"   {os.path.basename(f)}")
    
    # DEBUG: Extract unique emotions from filenames
    print(f"\n🔍 Analyzing file naming convention:")
    emotions_found = set()
    for fpath in files[:10]:  # Sample first 10 files
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        if len(parts) >= 2 and "BASELINE" not in fname:
            emotions_found.add(parts[1])
    
    print(f"   Sample emotions found in filenames: {sorted(emotions_found)}")
    print(f"   Expected emotions in SUPERCLASS_MAP: {sorted(config.SUPERCLASS_MAP.keys())}")
    
    # Check for mismatches
    missing_in_config = emotions_found - set(config.SUPERCLASS_MAP.keys())
    if missing_in_config:
        print(f"\n   ⚠️  WARNING: Emotions in files but NOT in SUPERCLASS_MAP:")
        for em in sorted(missing_in_config):
            print(f"      - {em}")
    
    # Load baseline files for each subject
    baseline_dict = {}
    if config.USE_BASELINE_REDUCTION:
        print(f"\n🔧 Baseline Reduction: ENABLED")
        print("   Loading baseline recordings...")
        
        for fpath in files:
            fname = os.path.basename(fpath)
            parts = fname.split("_")
            
            if len(parts) < 2:
                continue
            
            subject = parts[0]
            
            # Skip if already loaded or if this IS a baseline file
            if subject in baseline_dict or "BASELINE" in fname:
                continue
            
            # Try to find baseline file
            baseline_patterns = [
                os.path.join(data_root, f"{subject}_BASELINE_STIMULUS_MUSE.json"),
                os.path.join(data_root, subject, f"{subject}_BASELINE_STIMULUS_MUSE.json"),
                os.path.join(data_root, subject, f"{subject}_BASELINE_STIMULUS_MUSE_cleaned", 
                           f"{subject}_BASELINE_STIMULUS_MUSE_cleaned.json")
            ]
            
            for baseline_path in baseline_patterns:
                if os.path.exists(baseline_path):
                    try:
                        with open(baseline_path, "r") as f:
                            data = json.load(f)
                        
                        # Extract baseline channels
                        tp9 = _interp_nan(_to_num(data.get("RAW_TP9", [])))
                        af7 = _interp_nan(_to_num(data.get("RAW_AF7", [])))
                        af8 = _interp_nan(_to_num(data.get("RAW_AF8", [])))
                        tp10 = _interp_nan(_to_num(data.get("RAW_TP10", [])))
                        
                        L = min(len(tp9), len(af7), len(af8), len(tp10))
                        if L > 0:
                            baseline_signal = np.stack([tp9[:L], af7[:L], af8[:L], tp10[:L]], axis=1)
                            baseline_signal = baseline_signal - np.nanmean(baseline_signal, axis=0, keepdims=True)
                            baseline_dict[subject] = baseline_signal
                            
                    except Exception as e:
                        print(f"   ⚠️  Failed to load baseline for {subject}: {e}")
                    break
        
        print(f"   ✅ Loaded {len(baseline_dict)} baseline recordings")
    else:
        print(f"\n🔧 Baseline Reduction: DISABLED")
    
    all_windows, all_labels, all_subjects = [], [], []
    win_samples = int(config.EEG_WINDOW_SEC * config.EEG_FS)
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))
    
    # Track statistics
    reduced_count = 0
    not_reduced_count = 0
    skipped_reasons = {
        'baseline_file': 0,
        'unknown_emotion': 0,
        'no_data': 0,
        'insufficient_length': 0,
        'parse_error': 0,
        'quality_filtered': 0
    }
    
    debug_file_details = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        
        if len(parts) < 2:
            skipped_reasons['parse_error'] += 1
            continue
        
        # Skip baseline files themselves
        if "BASELINE" in fname:
            skipped_reasons['baseline_file'] += 1
            continue
            
        subject = parts[0]
        emotion = parts[1].upper()
        
        if emotion not in config.SUPERCLASS_MAP:
            skipped_reasons['unknown_emotion'] += 1
            continue
        
        superclass = config.SUPERCLASS_MAP[emotion]
        
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
            
            # Quality filtering
            hsi_tp9 = _to_num(data.get("HSI_TP9", []))[:L]
            hsi_af7 = _to_num(data.get("HSI_AF7", []))[:L]
            hsi_af8 = _to_num(data.get("HSI_AF8", []))[:L]
            hsi_tp10 = _to_num(data.get("HSI_TP10", []))[:L]
            head_on = _to_num(data.get("HeadBandOn", []))[:L]
            
            mask = np.isfinite(tp9[:L]) & np.isfinite(af7[:L]) & np.isfinite(af8[:L]) & np.isfinite(tp10[:L])
            
            # DEBUG: Check if quality indicators exist
            has_quality_data = len(head_on) == L and len(hsi_tp9) == L and len(hsi_af7) == L and len(hsi_af8) == L and len(hsi_tp10) == L
            
            if has_quality_data:
                # Apply quality filtering only if quality data is available
                quality_mask = (
                    (head_on == 1) &
                    np.isfinite(hsi_tp9) & (hsi_tp9 <= 2) &
                    np.isfinite(hsi_af7) & (hsi_af7 <= 2) &
                    np.isfinite(hsi_af8) & (hsi_af8 <= 2) &
                    np.isfinite(hsi_tp10) & (hsi_tp10 <= 2)
                )
                before_quality = np.sum(mask)
                mask = mask & quality_mask
                after_quality = np.sum(mask)
                
                if before_quality > 0 and after_quality == 0:
                    skipped_reasons['quality_filtered'] += 1
                    debug_file_details.append({
                        'file': fname,
                        'reason': 'quality_filtered',
                        'before_quality': int(before_quality),
                        'after_quality': int(after_quality),
                        'head_on_mean': float(np.mean(head_on[np.isfinite(head_on)])) if np.any(np.isfinite(head_on)) else np.nan,
                        'hsi_mean': float(np.mean([np.mean(h[np.isfinite(h)]) for h in [hsi_tp9, hsi_af7, hsi_af8, hsi_tp10] if np.any(np.isfinite(h))]))
                    })
            
            tp9, af7, af8, tp10 = tp9[:L][mask], af7[:L][mask], af8[:L][mask], tp10[:L][mask]
            L = len(tp9)
            if L < win_samples:
                skipped_reasons['insufficient_length'] += 1
                debug_file_details.append({
                    'file': fname,
                    'reason': 'insufficient_length',
                    'length': L,
                    'required': win_samples
                })
                continue
            
            signal = np.stack([tp9, af7, af8, tp10], axis=1)  # (T, 4)
            signal = signal - np.nanmean(signal, axis=0, keepdims=True)
            
            # Apply baseline reduction if available
            if config.USE_BASELINE_REDUCTION and subject in baseline_dict:
                baseline_signal = baseline_dict[subject]
                
                # Match lengths
                common_len = min(len(signal), len(baseline_signal))
                signal_trim = signal[:common_len]
                baseline_trim = baseline_signal[:common_len]
                
                # Apply InvBase method
                signal = apply_baseline_reduction(signal_trim, baseline_trim)
                L = len(signal)
                
                reduced_count += 1
            else:
                not_reduced_count += 1
            
            # Create windows
            for start in range(0, L - win_samples + 1, step_samples):
                window = signal[start:start + win_samples]
                if len(window) == win_samples:
                    all_windows.append(window)
                    all_labels.append(superclass)
                    all_subjects.append(subject)
        
        except Exception as e:
            skipped_reasons['parse_error'] += 1
            continue
    
    # Print statistics
    print(f"\n📊 File Processing Summary:")
    print(f"   Total files found: {len(files)}")
    print(f"   Successfully processed: {len(all_windows)} windows")
    print(f"\n   Skipped files:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"      {reason}: {count}")
    
    # Print detailed debug info if available
    if debug_file_details:
        print(f"\n🔍 Detailed filtering information (first 10 files):")
        for detail in debug_file_details[:10]:
            if detail['reason'] == 'quality_filtered':
                print(f"   {detail['file']}: quality filtered")
                print(f"      Before quality: {detail['before_quality']}, After: {detail['after_quality']}")
                print(f"      HeadBandOn mean: {detail['head_on_mean']:.2f}, HSI mean: {detail['hsi_mean']:.2f}")
            elif detail['reason'] == 'insufficient_length':
                print(f"   {detail['file']}: insufficient length")
                print(f"      Got {detail['length']} samples, need {detail['required']}")
    
    if len(all_windows) == 0:
        print("\n❌ ERROR: No valid EEG windows extracted!")
        print("\n💡 TROUBLESHOOTING STEPS:")
        print("   1. Check that emotion names in files match SUPERCLASS_MAP")
        print("   2. Verify DATA_ROOT path is correct")
        print("   3. Check that JSON files contain RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10 keys")
        print("   4. Ensure HeadBandOn and quality indicators (HSI_*) are present")
        print("   5. Check if quality_filtered count is high - quality thresholds may be too strict")
        raise ValueError("No valid EEG data extracted.")
    
    X_raw = np.stack(all_windows).astype(np.float32)
    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    y_labels = np.array([label_to_id[lab] for lab in all_labels], dtype=np.int64)
    subject_ids = np.array(all_subjects)
    
    print(f"\n✅ EEG data loaded: {X_raw.shape}")
    print(f"   Label distribution: {Counter(all_labels)}")
    
    if config.USE_BASELINE_REDUCTION:
        total_files = reduced_count + not_reduced_count
        print(f"\n📊 Baseline Reduction Statistics:")
        print(f"   ✅ Files with baseline reduction: {reduced_count}")
        print(f"   ⚠️  Files without baseline: {not_reduced_count}")
        if total_files > 0:
            print(f"   📈 Reduction rate: {100*reduced_count/total_files:.1f}%")
    
    return X_raw, y_labels, subject_ids, label_to_id


def extract_eeg_features(X_raw, config, eps=1e-12):
    """Extract 26 features per channel from EEG windows."""
    print("Extracting EEG features (26 per channel)...")
    N, T, C = X_raw.shape
    
    P = (np.abs(np.fft.rfft(X_raw, axis=1))**2) / T
    freqs = np.fft.rfftfreq(T, d=1/config.EEG_FS)
    
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
# DATALOADER CREATION
# ==================================================

def prepare_eeg_dataloaders(X_features, y_labels, split_indices, config):
    """
    Prepare standardized data loaders for train, validation, and test sets.
    
    Args:
        X_features: (N, C, F) - extracted features
        y_labels: (N,) - class labels
        split_indices: dict with 'train', 'val', 'test' indices
        config: Config object with settings
    
    Returns:
        loaders: dict with 'train', 'val', 'test' DataLoaders
        stats: dict with 'mu' and 'sd' for standardization
    """
    print("\n" + "="*80)
    print("PREPARING DATA LOADERS")
    print("="*80)
    
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    Xtr, Xva, Xte = X_features[train_idx], X_features[val_idx], X_features[test_idx]
    ytr, yva, yte = y_labels[train_idx], y_labels[val_idx], y_labels[test_idx]
    
    # Standardization using training set statistics
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    
    print(f"📊 Data shapes:")
    print(f"   Train: {Xtr.shape}")
    print(f"   Val: {Xva.shape}")
    print(f"   Test: {Xte.shape}")
    
    # Balanced sampling for training
    class_counts = np.bincount(ytr, minlength=config.NUM_CLASSES).astype(np.float32)
    class_sample_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_sample_weights[ytr]
    sample_weights_tensor = torch.from_numpy(sample_weights.astype(np.float32))
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    print(f"\n⚖️  Class balancing:")
    for i in range(config.NUM_CLASSES):
        print(f"   Class {i}: {int(class_counts[i])} samples (weight: {class_sample_weights[i]:.3f})")
    
    # Create datasets
    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    
    # Create data loaders
    tr_loader = DataLoader(tr_ds, batch_size=config.EEG_BATCH_SIZE, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    
    print(f"\n✅ Data loaders created:")
    print(f"   Train batches: {len(tr_loader)} (batch_size={config.EEG_BATCH_SIZE})")
    print(f"   Val batches: {len(va_loader)} (batch_size=256)")
    print(f"   Test batches: {len(te_loader)} (batch_size=256)")
    
    loaders = {
        'train': tr_loader,
        'val': va_loader,
        'test': te_loader
    }
    
    stats = {
        'mu': mu,
        'sd': sd
    }
    
    return loaders, stats
