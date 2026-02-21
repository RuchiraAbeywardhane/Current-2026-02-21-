"""
Data Loading Module
===================
Functions for loading and preparing EEG and BVP datasets from disk.

Usage:
    from data_loaders import load_eeg_data, load_bvp_data, create_data_split
"""

import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict, Counter
from config import config
from preprocessing import preprocess_bvp, extract_bvp_features, moving_average_backward


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


def extract_eeg_features_from_signal(signal, fs=256.0):
    """
    Extract features from raw EEG signal.
    
    Args:
        signal (np.ndarray): Raw EEG signal
        fs (float): Sampling frequency
    
    Returns:
        np.ndarray: Feature vector (26 features)
    """
    from scipy.signal import welch
    from scipy.stats import skew, kurtosis
    
    features = []
    
    # Time domain features
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.var(signal))
    features.append(skew(signal))
    features.append(kurtosis(signal))
    features.append(np.max(signal))
    features.append(np.min(signal))
    features.append(np.ptp(signal))  # Peak-to-peak
    
    # Frequency domain features (PSD for each band)
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    
    bands = [
        (1, 3),    # delta
        (4, 7),    # theta
        (8, 13),   # alpha
        (14, 30),  # beta
        (31, 45)   # gamma
    ]
    
    for low, high in bands:
        idx = (freqs >= low) & (freqs <= high)
        band_power = np.sum(psd[idx])
        features.append(band_power)
    
    # Differential Entropy (DE) for each band
    for low, high in bands:
        idx = (freqs >= low) & (freqs <= high)
        band_psd = psd[idx]
        if len(band_psd) > 0 and np.sum(band_psd) > 0:
            de = 0.5 * np.log(2 * np.pi * np.e * np.mean(band_psd))
        else:
            de = 0.0
        features.append(de)
    
    # Additional temporal features
    features.append(np.median(signal))
    features.append(np.percentile(signal, 25))
    features.append(np.percentile(signal, 75))
    features.append(np.sum(np.abs(np.diff(signal))))  # Total variation
    features.append(np.mean(np.abs(np.diff(signal))))  # Mean absolute difference
    features.append(np.sum(np.diff(np.sign(signal)) != 0))  # Zero crossings
    
    return np.array(features, dtype=np.float32)


def load_eeg_data(data_root, use_baseline_reduction=True):
    """
    Load preprocessed EEG data from JSON files.
    
    Args:
        data_root (str): Path to EEG data directory containing *_STIMULUS_MUSE_cleaned.json files
        use_baseline_reduction (bool): Whether baseline reduction was applied
    
    Returns:
        tuple: (features, labels, clip_names, subjects)
            features (np.ndarray): EEG features (N, C, dx) where C=4 channels, dx=26 features
            labels (np.ndarray): Label IDs (N,)
            clip_names (np.ndarray): Clip identifiers (N,)
            subjects (np.ndarray): Subject IDs (N,)
    """
    print(f"\n{'='*80}")
    print(f"LOADING EEG DATA FROM JSON FILES")
    print(f"{'='*80}")
    print(f"Data root: {data_root}")
    print(f"Baseline reduction: {use_baseline_reduction}")
    
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"EEG data directory not found: {data_root}")
    
    # Search for JSON files with multiple patterns
    print(f"\n🔍 Searching for EEG JSON files...")
    search_patterns = [
        "*_STIMULUS_MUSE_cleaned.json",
        "*_STIMULUS_MUSE.json",
        "**/*_STIMULUS_MUSE_cleaned.json",
        "**/*_STIMULUS_MUSE.json"
    ]
    
    json_files = []
    for pattern in search_patterns:
        found = list(data_path.glob(pattern))
        json_files.extend(found)
        if found:
            print(f"   Pattern '{pattern}': Found {len(found)} files")
    
    # Remove duplicates
    json_files = sorted(set(json_files))
    
    print(f"\n📊 Total unique JSON files found: {len(json_files)}")
    
    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No EEG JSON files found in {data_root}\n"
            f"Expected files like: *_STIMULUS_MUSE_cleaned.json or *_STIMULUS_MUSE.json"
        )
    
    # Show first few files
    print(f"   Sample files:")
    for f in json_files[:5]:
        print(f"      {f.name}")
    
    all_features = []
    all_labels = []
    all_clip_names = []
    all_subjects = []
    
    fs = config.EEG_FS
    window_sec = config.EEG_WINDOW_SEC
    window_size = int(window_sec * fs)
    overlap = config.EEG_OVERLAP
    stride = int(window_size * (1 - overlap))
    
    files_processed = 0
    files_skipped = 0
    
    for json_file in json_files:
        fname = json_file.name
        
        # Skip BASELINE files
        if "BASELINE" in fname.upper():
            files_skipped += 1
            continue
        
        # Parse filename: expected format like "s01_ENTHUSIASM_STIMULUS_MUSE_cleaned.json"
        parts = fname.replace("_STIMULUS_MUSE_cleaned.json", "").replace("_STIMULUS_MUSE.json", "").split("_")
        
        if len(parts) < 2:
            print(f"   ⚠️  Skipping file with unexpected name format: {fname}")
            files_skipped += 1
            continue
        
        subject = parts[0]
        emotion = parts[1].upper()
        
        # Map emotion to label
        superclass = config.SUPERCLASS_MAP.get(emotion)
        if superclass is None:
            print(f"   ⚠️  Unknown emotion '{emotion}' in file: {fname}")
            files_skipped += 1
            continue
        
        label = config.SUPERCLASS_ID[superclass]
        
        try:
            # Load JSON file
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Extract EEG channels
            tp9 = _interp_nan(_to_num(data.get("RAW_TP9", [])))
            af7 = _interp_nan(_to_num(data.get("RAW_AF7", [])))
            af8 = _interp_nan(_to_num(data.get("RAW_AF8", [])))
            tp10 = _interp_nan(_to_num(data.get("RAW_TP10", [])))
            
            # Get minimum length
            L = min(len(tp9), len(af7), len(af8), len(tp10))
            
            if L < window_size:
                files_skipped += 1
                continue
            
            # Truncate to same length
            tp9 = tp9[:L]
            af7 = af7[:L]
            af8 = af8[:L]
            tp10 = tp10[:L]
            
            # Quality filtering (if available)
            if 'HSI_TP9' in data and 'HeadBandOn' in data:
                hsi_tp9 = _to_num(data.get("HSI_TP9", []))[:L]
                hsi_af7 = _to_num(data.get("HSI_AF7", []))[:L]
                hsi_af8 = _to_num(data.get("HSI_AF8", []))[:L]
                hsi_tp10 = _to_num(data.get("HSI_TP10", []))[:L]
                head_on = _to_num(data.get("HeadBandOn", []))[:L]
                
                quality_mask = (
                    (head_on == 1) &
                    np.isfinite(hsi_tp9) & (hsi_tp9 <= 2) &
                    np.isfinite(hsi_af7) & (hsi_af7 <= 2) &
                    np.isfinite(hsi_af8) & (hsi_af8 <= 2) &
                    np.isfinite(hsi_tp10) & (hsi_tp10 <= 2)
                )
                
                tp9 = tp9[quality_mask]
                af7 = af7[quality_mask]
                af8 = af8[quality_mask]
                tp10 = tp10[quality_mask]
            
            L_clean = len(tp9)
            
            if L_clean < window_size:
                files_skipped += 1
                continue
            
            # Create sliding windows
            num_windows = (L_clean - window_size) // stride + 1
            
            for i in range(num_windows):
                start = i * stride
                end = start + window_size
                
                if end > L_clean:
                    break
                
                # Extract window for each channel
                tp9_win = tp9[start:end]
                af7_win = af7[start:end]
                af8_win = af8[start:end]
                tp10_win = tp10[start:end]
                
                # Extract features for each channel
                tp9_feats = extract_eeg_features_from_signal(tp9_win, fs)
                af7_feats = extract_eeg_features_from_signal(af7_win, fs)
                af8_feats = extract_eeg_features_from_signal(af8_win, fs)
                tp10_feats = extract_eeg_features_from_signal(tp10_win, fs)
                
                # Stack features: shape (4, 26)
                window_features = np.stack([tp9_feats, af7_feats, af8_feats, tp10_feats], axis=0)
                
                all_features.append(window_features)
                all_labels.append(label)
                all_clip_names.append(f"{subject}_{emotion}")
                all_subjects.append(subject)
            
            files_processed += 1
            
            if files_processed % 50 == 0:
                print(f"   Processed {files_processed}/{len(json_files)} files...")
        
        except Exception as e:
            print(f"   ⚠️  Error loading {fname}: {e}")
            files_skipped += 1
            continue
    
    print(f"\n📊 Loading Summary:")
    print(f"   Files processed: {files_processed}")
    print(f"   Files skipped: {files_skipped}")
    print(f"   Total windows extracted: {len(all_features)}")
    
    if len(all_features) == 0:
        raise ValueError(
            "No EEG data was loaded! Check:\n"
            "  1. JSON files contain RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10 keys\n"
            "  2. Signals are long enough for windowing\n"
            "  3. Emotion names match SUPERCLASS_MAP in config.py"
        )
    
    # Convert to arrays
    features = np.array(all_features, dtype=np.float32)  # (N, 4, 26)
    labels = np.array(all_labels, dtype=np.int64)
    clip_names = np.array(all_clip_names)
    subjects = np.array(all_subjects)
    
    print(f"\n✅ Loaded EEG Data Successfully:")
    print(f"   Total windows: {len(features)}")
    print(f"   Feature shape: {features.shape}")
    print(f"   Unique clips: {len(np.unique(clip_names))}")
    print(f"   Unique subjects: {len(np.unique(subjects))}")
    print(f"   Label distribution: {Counter(labels)}")
    
    return features, labels, clip_names, subjects


def load_bvp_data(data_root):
    """
    Load raw BVP data and preprocess it.
    
    Args:
        data_root (str): Path to raw BVP data directory
    
    Returns:
        tuple: (X1, X2, X3, feats, labels, clip_names, subjects)
            X1 (np.ndarray): Original signal (N, 1, T)
            X2 (np.ndarray): Smoothed signal (N, 1, T)
            X3 (np.ndarray): Downsampled signal (N, 1, T//2)
            feats (np.ndarray): Handcrafted features (N, 5)
            labels (np.ndarray): Label IDs (N,)
            clip_names (np.ndarray): Clip identifiers (N,)
            subjects (np.ndarray): Subject IDs (N,)
    """
    print(f"\n{'='*80}")
    print(f"LOADING BVP DATA")
    print(f"{'='*80}")
    print(f"Data root: {data_root}")
    
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"BVP data directory not found: {data_root}")
    
    X1_list, X2_list, X3_list, feats_list = [], [], [], []
    all_labels, all_clip_names, all_subjects = [], [], []
    
    fs = config.BVP_FS
    window_size = config.BVP_WINDOW_SIZE
    
    # Iterate through subject directories
    for subject_dir in sorted(data_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        
        # Load BVP CSV files
        for csv_file in sorted(subject_dir.glob("*_BVP.csv")):
            # Parse clip name and emotion
            clip_name = csv_file.stem.replace("_BVP", "")
            emotion = clip_name.split("_")[1] if "_" in clip_name else "UNKNOWN"
            
            # Map emotion to label
            superclass = config.SUPERCLASS_MAP.get(emotion.upper())
            if superclass is None:
                continue
            label = config.SUPERCLASS_ID[superclass]
            
            # Load raw BVP signal
            df = pd.read_csv(csv_file)
            bvp_raw = df.iloc[:, 0].values  # First column is BVP
            
            # Preprocess
            bvp_preprocessed = preprocess_bvp(bvp_raw, fs=fs)
            
            # Create windows
            num_windows = len(bvp_preprocessed) // window_size
            
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                window = bvp_preprocessed[start:end]
                
                if len(window) < window_size:
                    continue
                
                # X1: Original window
                X1 = window.reshape(1, -1)
                
                # X2: Smoothed with moving average
                X2 = moving_average_backward(window, s=5).reshape(1, -1)
                
                # X3: Downsampled (every 2nd sample)
                X3 = window[::2].reshape(1, -1)
                
                # Handcrafted features
                feats = extract_bvp_features(window, fs=fs)
                
                X1_list.append(X1)
                X2_list.append(X2)
                X3_list.append(X3)
                feats_list.append(feats)
                all_labels.append(label)
                all_clip_names.append(clip_name)
                all_subjects.append(subject_id)
    
    # Stack all windows
    X1 = np.array(X1_list, dtype=np.float32)
    X2 = np.array(X2_list, dtype=np.float32)
    X3 = np.array(X3_list, dtype=np.float32)
    feats = np.array(feats_list, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    clip_names = np.array(all_clip_names)
    subjects = np.array(all_subjects)
    
    print(f"\n📊 Loaded BVP Data:")
    print(f"   Total windows: {len(labels)}")
    print(f"   X1 shape: {X1.shape}")
    print(f"   X2 shape: {X2.shape}")
    print(f"   X3 shape: {X3.shape}")
    print(f"   Features shape: {feats.shape}")
    print(f"   Unique clips: {len(np.unique(clip_names))}")
    print(f"   Unique subjects: {len(np.unique(subjects))}")
    print(f"   Label distribution: {Counter(labels)}")
    
    return X1, X2, X3, feats, labels, clip_names, subjects


def create_data_split(clip_names, labels, subjects, test_ratio=0.15, val_ratio=0.15):
    """
    Create subject-independent stratified data split.
    
    Args:
        clip_names (np.ndarray): Clip identifiers
        labels (np.ndarray): Label IDs
        subjects (np.ndarray): Subject IDs
        test_ratio (float): Test set ratio
        val_ratio (float): Validation set ratio
    
    Returns:
        dict: {'train_clips', 'val_clips', 'test_clips', 'all_clips'}
    """
    print(f"\n{'='*80}")
    print(f"CREATING DATA SPLIT")
    print(f"{'='*80}")
    
    unique_clips = np.unique(clip_names)
    
    # Map clips to subjects and labels
    clip_to_subject = {}
    clip_to_label = {}
    
    for clip in unique_clips:
        mask = clip_names == clip
        clip_to_subject[clip] = subjects[mask][0]
        clip_to_label[clip] = labels[mask][0]
    
    # Group clips by subject
    subject_to_clips = defaultdict(list)
    for clip, subject in clip_to_subject.items():
        subject_to_clips[subject].append(clip)
    
    # Assign majority label to each subject
    unique_subjects = sorted(subject_to_clips.keys())
    subject_to_label = {}
    
    for subject in unique_subjects:
        subject_clips = subject_to_clips[subject]
        subject_labels = [clip_to_label[c] for c in subject_clips]
        majority_label = max(set(subject_labels), key=subject_labels.count)
        subject_to_label[subject] = majority_label
    
    # Stratified split by subjects
    class_to_subjects = defaultdict(list)
    for subject, label in subject_to_label.items():
        class_to_subjects[label].append(subject)
    
    train_subjects, val_subjects, test_subjects = [], [], []
    
    for class_id in range(config.NUM_CLASSES):
        class_subjects = class_to_subjects[class_id]
        if len(class_subjects) == 0:
            continue
        
        np.random.shuffle(class_subjects)
        n_test = max(1, int(len(class_subjects) * test_ratio))
        n_val = max(1, int(len(class_subjects) * val_ratio))
        
        test_subjects.extend(class_subjects[:n_test])
        val_subjects.extend(class_subjects[n_test:n_test+n_val])
        train_subjects.extend(class_subjects[n_test+n_val:])
    
    # Get clips for each split
    train_clips = [clip for subj in train_subjects for clip in subject_to_clips[subj]]
    val_clips = [clip for subj in val_subjects for clip in subject_to_clips[subj]]
    test_clips = [clip for subj in test_subjects for clip in subject_to_clips[subj]]
    
    print(f"\n📋 Split Summary:")
    print(f"   Train subjects: {len(train_subjects)}, clips: {len(train_clips)}")
    print(f"   Val subjects: {len(val_subjects)}, clips: {len(val_clips)}")
    print(f"   Test subjects: {len(test_subjects)}, clips: {len(test_clips)}")
    
    # Check label distribution
    train_labels = [clip_to_label[c] for c in train_clips]
    val_labels = [clip_to_label[c] for c in val_clips]
    test_labels = [clip_to_label[c] for c in test_clips]
    
    print(f"\n📊 Label Distribution:")
    print(f"   Train: {Counter(train_labels)}")
    print(f"   Val: {Counter(val_labels)}")
    print(f"   Test: {Counter(test_labels)}")
    
    return {
        'train_clips': train_clips,
        'val_clips': val_clips,
        'test_clips': test_clips,
        'all_clips': list(unique_clips)
    }


def filter_data_by_clips(data_tuple, clip_names, target_clips):
    """
    Filter data arrays by clip membership.
    
    Args:
        data_tuple (tuple): Data arrays to filter
        clip_names (np.ndarray): Clip names for all samples
        target_clips (list): Target clip names to keep
    
    Returns:
        tuple: Filtered data arrays
    """
    target_set = set(target_clips)
    mask = np.array([clip in target_set for clip in clip_names])
    
    return tuple(arr[mask] if isinstance(arr, np.ndarray) else arr for arr in data_tuple)


def get_common_clips(eeg_clip_names, bvp_clip_names):
    """
    Find common clips between EEG and BVP datasets.
    
    Args:
        eeg_clip_names (np.ndarray): EEG clip names
        bvp_clip_names (np.ndarray): BVP clip names
    
    Returns:
        list: Common clip names
    """
    eeg_clips = set(eeg_clip_names)
    bvp_clips = set(bvp_clip_names)
    common = eeg_clips & bvp_clips
    
    print(f"\n📊 Dataset Alignment:")
    print(f"   EEG clips: {len(eeg_clips)}")
    print(f"   BVP clips: {len(bvp_clips)}")
    print(f"   Common clips: {len(common)}")
    print(f"   EEG-only clips: {len(eeg_clips - bvp_clips)}")
    print(f"   BVP-only clips: {len(bvp_clips - eeg_clips)}")
    
    if len(common) == 0:
        raise ValueError("No common clips found between EEG and BVP!")
    
    return list(common)
