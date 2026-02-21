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
from pathlib import Path
from collections import defaultdict, Counter
from config import config
from preprocessing import preprocess_bvp, extract_bvp_features, moving_average_backward


def load_eeg_data(data_root, use_baseline_reduction=True):
    """
    Load preprocessed EEG data from disk.
    
    Args:
        data_root (str): Path to EEG data directory
        use_baseline_reduction (bool): Whether baseline reduction was applied
    
    Returns:
        tuple: (features, labels, clip_names, subjects)
            features (np.ndarray): EEG features (N, C, dx)
            labels (np.ndarray): Label IDs (N,)
            clip_names (np.ndarray): Clip identifiers (N,)
            subjects (np.ndarray): Subject IDs (N,)
    """
    print(f"\n{'='*80}")
    print(f"LOADING EEG DATA")
    print(f"{'='*80}")
    print(f"Data root: {data_root}")
    print(f"Baseline reduction: {use_baseline_reduction}")
    
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"EEG data directory not found: {data_root}")
    
    all_features = []
    all_labels = []
    all_clip_names = []
    all_subjects = []
    
    # Iterate through subject directories
    for subject_dir in sorted(data_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        
        # Load features and metadata
        for file_path in sorted(subject_dir.glob("*.npz")):
            data = np.load(file_path)
            
            features = data['features']  # (N, C, dx)
            labels = data['labels']  # (N,)
            clip_name = file_path.stem  # e.g., "s01_ENTHUSIASM_1"
            
            all_features.append(features)
            all_labels.append(labels)
            all_clip_names.extend([clip_name] * len(features))
            all_subjects.extend([subject_id] * len(features))
    
    # Concatenate all data
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    clip_names = np.array(all_clip_names)
    subjects = np.array(all_subjects)
    
    print(f"\n📊 Loaded EEG Data:")
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
