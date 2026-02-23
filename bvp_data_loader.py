"""
BVP Data Loader Module
======================

This module handles all data loading and preprocessing for BVP (Blood Volume Pulse)
signals from wearable devices (Samsung Watch / Empatica).

Features:
- BVP signal loading from JSON files
- Signal preprocessing (lowpass filtering, wavelet denoising)
- Baseline drift correction and normalization
- Windowing with configurable overlap
- Quality-based filtering
- Subject-independent and subject-dependent data splitting

Author: Final Year Project
Date: 2026
"""

import os
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, find_peaks, medfilt

# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def _to_num(x):
    """
    Convert various input types to numeric numpy array.
    
    Args:
        x: Input (list, scalar, etc.)
    
    Returns:
        numpy array of float64
    """
    if isinstance(x, list):
        if not x:
            return np.array([], np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, np.float64)
    return np.asarray([x], np.float64)


def _interp_nan(a):
    """
    Interpolate NaN values in a 1D array using linear interpolation.
    
    Args:
        a: Input array with potential NaN values
    
    Returns:
        Array with NaN values interpolated
    """
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


# ==================================================
# SIGNAL PREPROCESSING
# ==================================================

def butter_lowpass(cutoff_hz, fs, order=6):
    """
    Design Butterworth lowpass filter.
    
    Args:
        cutoff_hz: Cutoff frequency in Hz
        fs: Sampling frequency
        order: Filter order (default: 6)
    
    Returns:
        b, a: Filter coefficients
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def wavelet_denoise(sig, wavelet="db4", level=4, thresh_scale=1.0):
    """
    Wavelet denoising for BVP signals using soft thresholding.
    
    This method decomposes the signal into wavelet coefficients,
    applies soft thresholding to remove noise, and reconstructs
    the denoised signal.
    
    Args:
        sig: Input signal (1D array)
        wavelet: Wavelet family (default: "db4" - Daubechies 4)
        level: Decomposition level (default: 4)
        thresh_scale: Threshold scaling factor (default: 1.0)
    
    Returns:
        Denoised signal
    
    Reference:
        Wavelet-based denoising for physiological signals
    """
    sig = np.asarray(sig, dtype=float)
    
    # Wavelet decomposition
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    
    # Estimate noise standard deviation from finest scale
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Universal threshold
    uthresh = thresh_scale * sigma * np.sqrt(2 * np.log(len(sig)))
    
    # Apply soft thresholding to detail coefficients
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], uthresh, mode="soft")
    
    # Reconstruct signal
    rec = pywt.waverec(coeffs, wavelet)
    
    # Handle length mismatch due to wavelet transform
    if len(rec) > len(sig):
        rec = rec[:len(sig)]
    elif len(rec) < len(sig):
        rec = np.pad(rec, (0, len(sig) - len(rec)), mode="edge")
    
    return rec


def baseline_correct(sig, fs, return_normalized=True):
    """
    Remove baseline drift from BVP signal using local minima interpolation.
    
    This method identifies local minima (diastolic points) in the BVP signal,
    interpolates a baseline from these points, and subtracts it to remove drift.
    
    Args:
        sig: Input BVP signal (1D array)
        fs: Sampling frequency
        return_normalized: If True, normalize to [0, 1] range (default: True)
    
    Returns:
        Baseline-corrected (and optionally normalized) signal
    
    Reference:
        Baseline wander removal for BVP/PPG signals
    """
    sig = np.asarray(sig).astype(float)
    
    # Find local minima (diastolic points)
    # Minimum distance between peaks ~= 60 bpm max heart rate
    min_dist_samples = max(1, int(fs * 60.0 / 200.0))
    
    # Calculate prominence threshold
    sig_range = np.max(sig) - np.min(sig)
    if np.isclose(sig_range, 0):
        sig_range = 1.0
    prominence = 0.03 * sig_range
    
    # Invert signal to find minima using peak detection
    inv = -sig
    minima_idx, _ = find_peaks(inv, distance=min_dist_samples, prominence=prominence)
    
    # Fallback: use median filtering if not enough minima found
    if len(minima_idx) < 2:
        win = int(1.5 * fs)
        if win % 2 == 0:
            win += 1
        win = max(3, min(win, len(sig)))
        baseline = medfilt(sig, kernel_size=win)
        corrected = sig - baseline
    else:
        # Interpolate baseline from minima
        idx = np.concatenate(([0], minima_idx, [len(sig) - 1]))
        vals = np.concatenate(([sig[0]], sig[minima_idx], [sig[-1]]))
        baseline = np.interp(np.arange(len(sig)), idx, vals)
        corrected = sig - baseline
    
    if not return_normalized:
        return corrected
    
    # Normalize to [0, 1]
    vmin, vmax = np.min(corrected), np.max(corrected)
    if np.isclose(vmin, vmax):
        return np.zeros_like(corrected)
    else:
        return (corrected - vmin) / (vmax - vmin)


def preprocess_bvp_signal(bvp_raw, fs, cutoff_hz=15.0, filter_order=6, 
                          wavelet="db4", denoise_level=4, use_baseline_correction=True, 
                          normalize=True):
    """
    Complete BVP preprocessing pipeline.
    
    Pipeline:
    1. Lowpass filtering (remove high-frequency noise)
    2. Wavelet denoising (remove residual noise)
    3. Baseline drift correction (OPTIONAL - remove low-frequency drift)
    4. Normalization and standardization
    
    Args:
        bvp_raw: Raw BVP signal (1D array)
        fs: Sampling frequency
        cutoff_hz: Lowpass filter cutoff frequency (default: 15 Hz)
        filter_order: Butterworth filter order (default: 6)
        wavelet: Wavelet family for denoising (default: "db4")
        denoise_level: Wavelet decomposition level (default: 4)
        use_baseline_correction: Whether to apply baseline drift correction (default: True)
        normalize: Whether to normalize to [0,1] (default: True)
    
    Returns:
        Preprocessed BVP signal (standardized with mean=0, std=1)
    """
    # Step 1: Lowpass filtering
    b, a = butter_lowpass(cutoff_hz, fs, order=filter_order)
    bvp_filtered = filtfilt(b, a, bvp_raw)
    
    # Step 2: Wavelet denoising
    bvp_denoised = wavelet_denoise(bvp_filtered, wavelet=wavelet, level=denoise_level)
    
    # Step 3: Baseline correction (OPTIONAL)
    if use_baseline_correction:
        bvp_normalized = baseline_correct(bvp_denoised, fs, return_normalized=normalize)
    else:
        # Skip baseline correction - just normalize if requested
        if normalize:
            vmin, vmax = np.min(bvp_denoised), np.max(bvp_denoised)
            if np.isclose(vmin, vmax):
                bvp_normalized = np.zeros_like(bvp_denoised)
            else:
                bvp_normalized = (bvp_denoised - vmin) / (vmax - vmin)
        else:
            bvp_normalized = bvp_denoised
    
    # Step 4: Standardization (z-score)
    bvp_mean = bvp_normalized.mean()
    bvp_std = bvp_normalized.std()
    if bvp_std < 1e-8:
        bvp_std = 1.0
    bvp_standardized = (bvp_normalized - bvp_mean) / bvp_std
    
    return bvp_standardized


# ==================================================
# DATA LOADING
# ==================================================

def load_bvp_data(data_root, config):
    """
    Load BVP data from Samsung Watch/Empatica JSON files.
    
    This function searches for BVP data files (SAMSUNG_WATCH or EMPATICA),
    applies preprocessing, and creates windowed samples for training.
    
    Args:
        data_root: Root directory containing BVP JSON files
        config: Configuration object with parameters (window size, overlap, etc.)
                Must have: BVP_FS, BVP_WINDOW_SEC, BVP_OVERLAP, SUPERCLASS_MAP
                Optional: USE_BVP_BASELINE_CORRECTION (default: True)
    
    Returns:
        X_raw: (N, T) - Raw BVP windows (preprocessed)
        y_labels: (N,) - Class labels as integers
        subject_ids: (N,) - Subject IDs for each window
        label_to_id: Dictionary mapping label names to integers
    """
    print("\n" + "="*80)
    print("LOADING BVP DATA (SAMSUNG WATCH / EMPATICA)")
    print("="*80)
    
    # Search for BVP files (both Samsung Watch and Empatica)
    patterns = [
        os.path.join(data_root, "*_STIMULUS_SAMSUNG_WATCH.json"),
        os.path.join(data_root, "*", "*_STIMULUS_SAMSUNG_WATCH.json"),
        os.path.join(data_root, "*_STIMULUS_EMPATICA.json"),
        os.path.join(data_root, "*", "*_STIMULUS_EMPATICA.json")
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    print(f"Found {len(files)} BVP files")
    
    if len(files) == 0:
        print("\n❌ ERROR: No BVP files found!")
        print(f"   Searched in: {data_root}")
        if os.path.exists(data_root):
            print(f"   Directory contents: {os.listdir(data_root)[:10]}")
        raise ValueError("No BVP files found. Check DATA_ROOT path.")
    
    print(f"\n📁 Sample files:")
    for f in files[:3]:
        print(f"   {os.path.basename(f)}")
    
    # Check if baseline correction is enabled (default: True)
    use_baseline_correction = getattr(config, 'USE_BVP_BASELINE_CORRECTION', True)
    if use_baseline_correction:
        print(f"\n🔧 BVP Baseline Correction: ENABLED")
    else:
        print(f"\n🔧 BVP Baseline Correction: DISABLED")
    
    # Prepare windowing parameters
    all_windows, all_labels, all_subjects = [], [], []
    win_samples = int(config.BVP_WINDOW_SEC * config.BVP_FS)
    step_samples = int(win_samples * (1.0 - config.BVP_OVERLAP))
    
    # Track statistics
    skipped_reasons = {
        'baseline_file': 0,
        'unknown_emotion': 0,
        'no_data': 0,
        'insufficient_length': 0,
        'parse_error': 0
    }
    
    successful_files = 0
    
    # Process each file
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
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Extract BVP data
            # Format can be: [[timestamp, value], ...] or direct array
            bvp_data = data.get("BVP", [])
            
            if not bvp_data:
                skipped_reasons['no_data'] += 1
                continue
            
            # Handle different BVP data formats
            if isinstance(bvp_data[0], list):
                # Format: [[timestamp, value], ...]
                bvp_raw = np.array([row[1] for row in bvp_data], dtype=float)
            else:
                # Format: [value1, value2, ...]
                bvp_raw = _to_num(bvp_data)
            
            # Interpolate any NaN values
            bvp_raw = _interp_nan(bvp_raw)
            
            L = len(bvp_raw)
            if L < win_samples:
                skipped_reasons['insufficient_length'] += 1
                continue
            
            # Apply preprocessing pipeline with configurable baseline correction
            bvp_processed = preprocess_bvp_signal(
                bvp_raw, 
                fs=config.BVP_FS,
                cutoff_hz=15.0,
                filter_order=6,
                wavelet="db4",
                denoise_level=4,
                use_baseline_correction=use_baseline_correction,
                normalize=True
            )
            
            # Create windows with overlap
            for start in range(0, L - win_samples + 1, step_samples):
                window = bvp_processed[start:start + win_samples]
                if len(window) == win_samples:
                    all_windows.append(window)
                    all_labels.append(superclass)
                    all_subjects.append(subject)
            
            successful_files += 1
        
        except Exception as e:
            skipped_reasons['parse_error'] += 1
            continue
    
    # Print statistics
    print(f"\n📊 File Processing Summary:")
    print(f"   Total files found: {len(files)}")
    print(f"   Successfully processed: {successful_files} files")
    print(f"   Total windows extracted: {len(all_windows)}")
    print(f"\n   Skipped files:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"      {reason}: {count}")
    
    if len(all_windows) == 0:
        print("\n❌ ERROR: No valid BVP windows extracted!")
        raise ValueError("No valid BVP data extracted.")
    
    # Convert to arrays
    X_raw = np.stack(all_windows).astype(np.float32)
    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    y_labels = np.array([label_to_id[lab] for lab in all_labels], dtype=np.int64)
    subject_ids = np.array(all_subjects)
    
    print(f"\n✅ BVP data loaded: {X_raw.shape}")
    print(f"   Window size: {win_samples} samples ({config.BVP_WINDOW_SEC}s @ {config.BVP_FS}Hz)")
    print(f"   Label distribution: {Counter(all_labels)}")
    
    return X_raw, y_labels, subject_ids, label_to_id


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_bvp_features(window, fs=64):
    """
    Extract handcrafted features from BVP window.
    
    Features:
    1. Mean value (DC component)
    2. Standard deviation (signal variability)
    3. Derivative std (signal dynamics)
    4. Heart rate proxy (peak count)
    5. Peak-to-peak amplitude (pulse strength)
    
    Args:
        window: BVP signal window (1D array)
        fs: Sampling frequency (default: 64 Hz)
    
    Returns:
        Feature vector (5 features)
    """
    window = np.asarray(window)
    
    # 1. Mean value
    mean_val = np.mean(window)
    
    # 2. Standard deviation
    std_val = np.std(window)
    
    # 3. Derivative standard deviation
    diff_std = np.std(np.diff(window))
    
    # 4. Heart rate proxy from peak counting
    # Minimum distance = 60 bpm max HR (assuming max 200 bpm)
    min_dist = int(fs * 60.0 / 200.0)
    peaks, _ = find_peaks(window, distance=min_dist)
    window_duration = len(window) / fs
    hr_proxy = len(peaks) / (window_duration + 1e-6)
    
    # 5. Peak-to-peak amplitude
    p2p = np.max(window) - np.min(window)
    
    return np.array([mean_val, std_val, diff_std, hr_proxy, p2p], dtype=np.float32)


def extract_all_bvp_features(X_raw, fs=64):
    """
    Extract features from all BVP windows.
    
    Args:
        X_raw: (N, T) - Raw BVP windows
        fs: Sampling frequency
    
    Returns:
        X_features: (N, 5) - Feature matrix
    """
    print("\n" + "="*80)
    print("EXTRACTING BVP FEATURES")
    print("="*80)
    
    N = len(X_raw)
    features = []
    
    for i in range(N):
        feat = extract_bvp_features(X_raw[i], fs=fs)
        features.append(feat)
    
    X_features = np.stack(features).astype(np.float32)
    
    print(f"✅ BVP features extracted: {X_features.shape}")
    print(f"   Features: [mean, std, diff_std, hr_proxy, peak2peak]")
    
    return X_features


# ==================================================
# DATA SPLITTING
# ==================================================

def create_data_splits(y_labels, subject_ids, config, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits with subject-independent or random strategy.
    
    Args:
        y_labels: Array of class labels
        subject_ids: Array of subject IDs
        config: Configuration object
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    
    Returns:
        split_indices: Dictionary with 'train', 'val', 'test' index arrays
    """
    print("\n" + "="*80)
    print("CREATING DATA SPLIT (BVP)")
    print("="*80)
    
    n_samples = len(y_labels)
    
    if config.SUBJECT_INDEPENDENT:
        print("  Strategy: SUBJECT-INDEPENDENT split")
        unique_subjects = np.unique(subject_ids)
        np.random.shuffle(unique_subjects)
        
        n_test = int(len(unique_subjects) * test_ratio)
        n_val = int(len(unique_subjects) * val_ratio)
        
        test_subjects = unique_subjects[:n_test]
        val_subjects = unique_subjects[n_test:n_test+n_val]
        train_subjects = unique_subjects[n_test+n_val:]
        
        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)
        test_mask = np.isin(subject_ids, test_subjects)
        
        print(f"  Train subjects: {len(train_subjects)}")
        print(f"  Val subjects: {len(val_subjects)}")
        print(f"  Test subjects: {len(test_subjects)}")
    else:
        print("  Strategy: RANDOM split")
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_test = int(n_samples * test_ratio)
        n_val = int(n_samples * val_ratio)
        
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
    
    # Print class distribution for each split
    for split_name, indices in split_indices.items():
        labels_split = y_labels[indices]
        dist = Counter(labels_split)
        print(f"   {split_name.capitalize()} class distribution: {dict(dist)}")
    
    return split_indices
