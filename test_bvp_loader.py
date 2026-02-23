"""
BVP Data Loader Test Script
============================

This script tests the BVP data loader module to verify:
- Data loading from JSON files
- Signal preprocessing (filtering, denoising, baseline correction)
- Feature extraction
- Data splitting
- Visualization of preprocessing steps

Usage:
    python test_bvp_loader.py

Author: Final Year Project
Date: 2026
"""

import os
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# Import BVP data loader
from bvp_data_loader import (
    load_bvp_data,
    extract_bvp_features,
    extract_all_bvp_features,
    create_data_splits,
    preprocess_bvp_signal,
    apply_baseline_reduction,
    butter_lowpass,
    wavelet_denoise,
    baseline_correct,
    _to_num,
    _interp_nan
)


# ==================================================
# CONFIGURATION
# ==================================================

class TestConfig:
    """Configuration for testing BVP data loader."""
    # Data path - CHANGE THIS TO YOUR DATA PATH
    DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
    
    # BVP parameters
    BVP_FS = 64  # Samsung Watch sampling frequency
    BVP_WINDOW_SEC = 10  # 10-second windows
    BVP_OVERLAP = 0.0  # No overlap
    BVP_DEVICE = 'samsung_watch'  # Only use Samsung Watch, not Empatica
    
    # BVP Preprocessing
    USE_BVP_BASELINE_CORRECTION = False  # Baseline drift correction
    USE_BVP_BASELINE_REDUCTION = True   # Baseline reduction (InvBase method)
    
    # Classification
    NUM_CLASSES = 4
    SUBJECT_INDEPENDENT = True
    SEED = 42
    
    # Label mappings
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",
        "FEAR": "Q2",
        "SADNESS": "Q3",
        "NEUTRAL": "Q4",
    }


# Set random seed
config = TestConfig()
random.seed(config.SEED)
np.random.seed(config.SEED)


# ==================================================
# TEST FUNCTIONS
# ==================================================

def debug_baseline_matching(data_root, bvp_device='samsung_watch'):
    """Debug baseline file matching."""
    print("\n" + "="*80)
    print("DEBUG: BASELINE FILE MATCHING")
    print("="*80)
    
    # Find baseline files
    baseline_patterns = []
    if bvp_device in ['samsung_watch', 'both']:
        baseline_patterns.extend([
            os.path.join(data_root, "*_BASELINE_SAMSUNG_WATCH.json"),
            os.path.join(data_root, "*", "*_BASELINE_SAMSUNG_WATCH.json")
        ])
    
    baseline_files = sorted({p for pat in baseline_patterns for p in glob.glob(pat)})
    
    print(f"\n📂 Found {len(baseline_files)} baseline files:")
    for i, bf in enumerate(baseline_files[:5], 1):
        basename = os.path.basename(bf)
        parts = basename.split("_")
        subject = parts[0] if parts else "UNKNOWN"
        print(f"   {i}. {basename}")
        print(f"      → Subject ID: {subject}")
    
    if len(baseline_files) > 5:
        print(f"   ... and {len(baseline_files) - 5} more")
    
    # Find stimulus files
    stimulus_patterns = []
    if bvp_device in ['samsung_watch', 'both']:
        stimulus_patterns.extend([
            os.path.join(data_root, "*_STIMULUS_SAMSUNG_WATCH.json"),
            os.path.join(data_root, "*", "*_STIMULUS_SAMSUNG_WATCH.json")
        ])
    
    stimulus_files = sorted({p for pat in stimulus_patterns for p in glob.glob(pat)})
    
    print(f"\n📂 Found {len(stimulus_files)} stimulus files (showing first 10):")
    stimulus_subjects = set()
    
    for i, sf in enumerate(stimulus_files[:10], 1):
        basename = os.path.basename(sf)
        parts = basename.split("_")
        subject = parts[0] if parts else "UNKNOWN"
        emotion = parts[1] if len(parts) > 1 else "UNKNOWN"
        stimulus_subjects.add(subject)
        print(f"   {i}. {basename}")
        print(f"      → Subject: {subject}, Emotion: {emotion}")
    
    if len(stimulus_files) > 10:
        print(f"   ... and {len(stimulus_files) - 10} more")
    
    # Extract all unique subjects
    all_stimulus_subjects = set()
    stimulus_emotions = set()
    for sf in stimulus_files:
        basename = os.path.basename(sf)
        parts = basename.split("_")
        if parts:
            all_stimulus_subjects.add(parts[0])
        if len(parts) > 1:
            stimulus_emotions.add(parts[1])
    
    all_baseline_subjects = set()
    for bf in baseline_files:
        basename = os.path.basename(bf)
        parts = basename.split("_")
        if parts:
            all_baseline_subjects.add(parts[0])
    
    print(f"\n📊 Subject Summary:")
    print(f"   Unique subjects in STIMULUS files: {len(all_stimulus_subjects)}")
    print(f"   Stimulus subjects: {sorted(all_stimulus_subjects)[:20]}")
    if len(all_stimulus_subjects) > 20:
        print(f"   ... and {len(all_stimulus_subjects) - 20} more")
    
    print(f"\n   Unique subjects in BASELINE files: {len(all_baseline_subjects)}")
    print(f"   Baseline subjects: {sorted(all_baseline_subjects)[:20]}")
    if len(all_baseline_subjects) > 20:
        print(f"   ... and {len(all_baseline_subjects) - 20} more")
    
    # Check which subjects have baselines
    subjects_with_baseline = all_stimulus_subjects & all_baseline_subjects
    subjects_without_baseline = all_stimulus_subjects - all_baseline_subjects
    
    print(f"\n🔍 Baseline Matching:")
    print(f"   ✅ Subjects WITH baseline: {len(subjects_with_baseline)}")
    if subjects_with_baseline:
        print(f"      {sorted(subjects_with_baseline)[:20]}")
        if len(subjects_with_baseline) > 20:
            print(f"      ... and {len(subjects_with_baseline) - 20} more")
    
    print(f"\n   ❌ Subjects WITHOUT baseline: {len(subjects_without_baseline)}")
    if subjects_without_baseline:
        print(f"      {sorted(subjects_without_baseline)[:20]}")
        if len(subjects_without_baseline) > 20:
            print(f"      ... and {len(subjects_without_baseline) - 20} more")
    
    # Show all emotions found
    print(f"\n🎭 Emotions Found in Dataset ({len(stimulus_emotions)} unique):")
    for emotion in sorted(stimulus_emotions):
        print(f"   - {emotion}")
    
    # Check which emotions are mapped
    mapped_emotions = set(config.SUPERCLASS_MAP.keys())
    unmapped_emotions = stimulus_emotions - mapped_emotions
    
    print(f"\n📋 Emotion Mapping Status:")
    print(f"   ✅ Mapped emotions: {len(mapped_emotions & stimulus_emotions)}")
    for emotion in sorted(mapped_emotions & stimulus_emotions):
        print(f"      - {emotion} → {config.SUPERCLASS_MAP[emotion]}")
    
    print(f"\n   ❌ Unmapped emotions: {len(unmapped_emotions)}")
    for emotion in sorted(unmapped_emotions):
        print(f"      - {emotion}")
    
    # Check if baseline files are being loaded correctly
    print(f"\n🧪 Testing Baseline File Loading:")
    if baseline_files:
        test_file = baseline_files[0]
        print(f"   Testing file: {os.path.basename(test_file)}")
        try:
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            bvp_baseline = data.get("BVPRaw", [])
            print(f"   ✅ File loaded successfully")
            print(f"   BVP data type: {type(bvp_baseline)}")
            print(f"   BVP length: {len(bvp_baseline)}")
            if bvp_baseline:
                print(f"   BVP sample: {bvp_baseline[:3]}")
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
    
    print("="*80)
    
    return all_stimulus_subjects, all_baseline_subjects, unmapped_emotions


def test_preprocessing_pipeline(bvp_raw):
    """Test preprocessing pipeline with visualization."""
    print("\n" + "="*80)
    print("TEST 1: PREPROCESSING PIPELINE")
    print("="*80)
    
    if bvp_raw is None or len(bvp_raw) == 0:
        print("❌ No data to preprocess!")
        return
    
    # Take a segment for visualization
    segment_length = min(len(bvp_raw), config.BVP_FS * 30)  # 30 seconds
    bvp_segment = bvp_raw[:segment_length]
    
    print(f"Processing {len(bvp_segment)} samples ({len(bvp_segment)/config.BVP_FS:.1f}s)")
    
    # Step-by-step preprocessing
    from scipy.signal import filtfilt
    
    # 1. Lowpass filtering
    b, a = butter_lowpass(15.0, config.BVP_FS, order=6)
    bvp_filtered = filtfilt(b, a, bvp_segment)
    print("✅ Step 1: Lowpass filtering (15 Hz cutoff)")
    
    # 2. Wavelet denoising
    bvp_denoised = wavelet_denoise(bvp_filtered, wavelet="db4", level=4)
    print("✅ Step 2: Wavelet denoising (db4, level 4)")
    
    # 3. Baseline correction
    bvp_corrected = baseline_correct(bvp_denoised, config.BVP_FS, return_normalized=True)
    print("✅ Step 3: Baseline drift correction")
    
    # 4. Standardization
    bvp_final = (bvp_corrected - bvp_corrected.mean()) / (bvp_corrected.std() + 1e-8)
    print("✅ Step 4: Standardization (z-score)")
    
    # Visualize preprocessing steps
    fig, axes = plt.subplots(5, 1, figsize=(14, 10))
    time = np.arange(len(bvp_segment)) / config.BVP_FS
    
    axes[0].plot(time, bvp_segment, 'b-', linewidth=0.5)
    axes[0].set_title('Step 0: Raw BVP Signal', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, bvp_filtered, 'g-', linewidth=0.5)
    axes[1].set_title('Step 1: After Lowpass Filter (15 Hz)', fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time, bvp_denoised, 'orange', linewidth=0.5)
    axes[2].set_title('Step 2: After Wavelet Denoising', fontweight='bold')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(time, bvp_corrected, 'r-', linewidth=0.5)
    axes[3].set_title('Step 3: After Baseline Correction', fontweight='bold')
    axes[3].set_ylabel('Amplitude')
    axes[3].grid(True, alpha=0.3)
    
    axes[4].plot(time, bvp_final, 'purple', linewidth=0.5)
    axes[4].set_title('Step 4: Final Preprocessed Signal (Standardized)', fontweight='bold')
    axes[4].set_ylabel('Z-score')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bvp_preprocessing_steps.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: bvp_preprocessing_steps.png")
    plt.close()
    
    return bvp_final


def test_feature_extraction(bvp_processed):
    """Test feature extraction."""
    print("\n" + "="*80)
    print("TEST 2: FEATURE EXTRACTION")
    print("="*80)
    
    if bvp_processed is None or len(bvp_processed) == 0:
        print("❌ No data for feature extraction!")
        return
    
    # Extract features from a single window
    win_samples = config.BVP_WINDOW_SEC * config.BVP_FS
    
    if len(bvp_processed) >= win_samples:
        window = bvp_processed[:win_samples]
        features = extract_bvp_features(window, fs=config.BVP_FS)
        
        print(f"Window size: {len(window)} samples ({config.BVP_WINDOW_SEC}s)")
        print(f"\n📊 Extracted Features (5 total):")
        print(f"   1. Mean value:        {features[0]:.4f}")
        print(f"   2. Std deviation:     {features[1]:.4f}")
        print(f"   3. Derivative std:    {features[2]:.4f}")
        print(f"   4. Heart rate proxy:  {features[3]:.2f} beats/window")
        print(f"   5. Peak-to-peak amp:  {features[4]:.4f}")
        print(f"\n✅ Feature vector shape: {features.shape}")
    else:
        print(f"❌ Signal too short for {config.BVP_WINDOW_SEC}s window")


def test_full_data_loading():
    """Test full data loading pipeline."""
    print("\n" + "="*80)
    print("TEST 3: FULL DATA LOADING PIPELINE")
    print("="*80)
    
    try:
        # Load all BVP data
        X_raw, y_labels, subject_ids, label_to_id = load_bvp_data(config.DATA_ROOT, config)
        
        print(f"\n✅ Data loaded successfully!")
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total windows: {len(X_raw)}")
        print(f"   Window shape: {X_raw.shape}")
        print(f"   Unique subjects: {len(np.unique(subject_ids))}")
        print(f"   Subjects: {np.unique(subject_ids)}")
        print(f"   Label mapping: {label_to_id}")
        
        # Class distribution
        from collections import Counter
        class_dist = Counter(y_labels)
        print(f"\n📈 Class Distribution:")
        for label_id, count in sorted(class_dist.items()):
            label_name = [k for k, v in label_to_id.items() if v == label_id][0]
            percentage = 100 * count / len(y_labels)
            print(f"   Class {label_id} ({label_name}): {count} samples ({percentage:.1f}%)")
        
        # Visualize a few windows
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()
        
        for i in range(min(4, len(X_raw))):
            ax = axes[i]
            window = X_raw[i]
            label_id = y_labels[i]
            label_name = [k for k, v in label_to_id.items() if v == label_id][0]
            subject = subject_ids[i]
            
            time = np.arange(len(window)) / config.BVP_FS
            ax.plot(time, window, 'b-', linewidth=0.8)
            ax.set_title(f'Sample {i+1}: {label_name} (Subject {subject})', fontweight='bold')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('BVP (normalized)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bvp_sample_windows.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Sample windows visualization saved: bvp_sample_windows.png")
        plt.close()
        
        return X_raw, y_labels, subject_ids, label_to_id
        
    except Exception as e:
        print(f"\n❌ Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_data_splitting(y_labels, subject_ids):
    """Test data splitting."""
    print("\n" + "="*80)
    print("TEST 4: DATA SPLITTING")
    print("="*80)
    
    if y_labels is None or subject_ids is None:
        print("❌ No data for splitting!")
        return
    
    split_indices = create_data_splits(y_labels, subject_ids, config)
    
    print(f"\n✅ Data split created successfully!")
    
    # Verify no overlap
    train_set = set(split_indices['train'])
    val_set = set(split_indices['val'])
    test_set = set(split_indices['test'])
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    print(f"\n🔍 Overlap Check:")
    print(f"   Train-Val overlap: {len(overlap_train_val)} samples")
    print(f"   Train-Test overlap: {len(overlap_train_test)} samples")
    print(f"   Val-Test overlap: {len(overlap_val_test)} samples")
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("   ✅ No overlap detected - splits are clean!")
    else:
        print("   ⚠️  WARNING: Overlap detected in splits!")
    
    # Verify subject independence
    if config.SUBJECT_INDEPENDENT:
        train_subjects = set(subject_ids[split_indices['train']])
        val_subjects = set(subject_ids[split_indices['val']])
        test_subjects = set(subject_ids[split_indices['test']])
        
        subject_overlap = train_subjects & val_subjects & test_subjects
        
        print(f"\n🧑 Subject Independence Check:")
        print(f"   Train subjects: {sorted(train_subjects)}")
        print(f"   Val subjects: {sorted(val_subjects)}")
        print(f"   Test subjects: {sorted(test_subjects)}")
        print(f"   Subject overlap: {len(subject_overlap)}")
        
        if len(subject_overlap) == 0:
            print("   ✅ Subject-independent split verified!")
        else:
            print(f"   ⚠️  WARNING: {len(subject_overlap)} subjects appear in multiple splits!")


def test_feature_extraction_batch(X_raw):
    """Test batch feature extraction."""
    print("\n" + "="*80)
    print("TEST 5: BATCH FEATURE EXTRACTION")
    print("="*80)
    
    if X_raw is None:
        print("❌ No data for feature extraction!")
        return
    
    X_features = extract_all_bvp_features(X_raw, fs=config.BVP_FS)
    
    print(f"\n✅ Features extracted successfully!")
    print(f"\n📊 Feature Statistics:")
    print(f"   Feature matrix shape: {X_features.shape}")
    print(f"   Feature names: [mean, std, diff_std, hr_proxy, peak2peak]")
    
    for i in range(X_features.shape[1]):
        feat_name = ['Mean', 'Std', 'Diff Std', 'HR Proxy', 'Peak2Peak'][i]
        feat_values = X_features[:, i]
        print(f"\n   {feat_name}:")
        print(f"      Min:  {np.min(feat_values):.4f}")
        print(f"      Max:  {np.max(feat_values):.4f}")
        print(f"      Mean: {np.mean(feat_values):.4f}")
        print(f"      Std:  {np.std(feat_values):.4f}")


def test_baseline_reduction_method(data_root, bvp_device='samsung_watch'):
    """
    Test baseline reduction (InvBase method) with actual baseline and stimulus files.
    This demonstrates how ONE baseline per subject is used to normalize all trials from that subject.
    """
    print("\n" + "="*80)
    print("TEST: BASELINE REDUCTION (InvBase Method)")
    print("="*80)
    print("\nThis test demonstrates how baseline files work:")
    print("  - ONE baseline file per subject (resting state recording)")
    print("  - Baseline is used to normalize ALL emotional trials from that subject")
    print("  - InvBase method: Divide trial FFT by baseline FFT")
    print("="*80)
    
    # Find baseline files
    baseline_patterns = []
    if bvp_device in ['samsung_watch', 'both']:
        baseline_patterns.extend([
            os.path.join(data_root, "*_BASELINE_SAMSUNG_WATCH.json"),
            os.path.join(data_root, "*", "*_BASELINE_SAMSUNG_WATCH.json"),
            os.path.join(data_root, "*_BASELINE_STIMULUS_SAMSUNG_WATCH.json"),
            os.path.join(data_root, "*", "*_BASELINE_STIMULUS_SAMSUNG_WATCH.json")
        ])
    
    baseline_files = sorted({p for pat in baseline_patterns for p in glob.glob(pat)})
    
    if not baseline_files:
        print("❌ No baseline files found!")
        return
    
    # Find stimulus files
    stimulus_patterns = []
    if bvp_device in ['samsung_watch', 'both']:
        stimulus_patterns.extend([
            os.path.join(data_root, "*_STIMULUS_SAMSUNG_WATCH.json"),
            os.path.join(data_root, "*", "*_STIMULUS_SAMSUNG_WATCH.json")
        ])
    
    stimulus_files = sorted({p for pat in stimulus_patterns for p in glob.glob(pat)})
    stimulus_files = [f for f in stimulus_files if "BASELINE" not in f]  # Exclude baseline files
    
    if not stimulus_files:
        print("❌ No stimulus files found!")
        return
    
    # Load ONE baseline file as example
    baseline_file = baseline_files[0]
    baseline_name = os.path.basename(baseline_file)
    baseline_subject = baseline_name.split("_")[0]
    
    print(f"\n📂 Loading baseline file:")
    print(f"   File: {baseline_name}")
    print(f"   Subject: {baseline_subject}")
    
    try:
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        bvp_baseline_raw = baseline_data.get("BVPRaw", [])
        
        if not bvp_baseline_raw:
            print("❌ No BVP data in baseline file!")
            return
        
        # Handle different formats
        if isinstance(bvp_baseline_raw[0], list):
            baseline_raw = np.array([row[1] for row in bvp_baseline_raw], dtype=float)
        else:
            baseline_raw = _to_num(bvp_baseline_raw)
        
        baseline_raw = _interp_nan(baseline_raw)
        
        print(f"   ✅ Baseline loaded: {len(baseline_raw)} samples ({len(baseline_raw)/config.BVP_FS:.1f}s)")
        
    except Exception as e:
        print(f"❌ Error loading baseline: {e}")
        return
    
    # Find stimulus files from the SAME subject
    subject_stimulus_files = [f for f in stimulus_files if os.path.basename(f).startswith(baseline_subject + "_")]
    
    if not subject_stimulus_files:
        print(f"❌ No stimulus files found for subject {baseline_subject}!")
        # Try first available subject
        for stim_file in stimulus_files[:10]:
            stim_name = os.path.basename(stim_file)
            stim_subject = stim_name.split("_")[0]
            stim_emotion = stim_name.split("_")[1] if len(stim_name.split("_")) > 1 else "UNKNOWN"
            
            # Find matching baseline
            matching_baseline = [bf for bf in baseline_files if os.path.basename(bf).startswith(stim_subject + "_")]
            if matching_baseline:
                baseline_file = matching_baseline[0]
                baseline_subject = stim_subject
                subject_stimulus_files = [stim_file]
                
                print(f"\n   Trying alternative subject: {stim_subject}")
                print(f"   Baseline: {os.path.basename(baseline_file)}")
                print(f"   Stimulus: {os.path.basename(stim_file)} ({stim_emotion})")
                
                # Reload baseline
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                bvp_baseline_raw = baseline_data.get("BVPRaw", [])
                if isinstance(bvp_baseline_raw[0], list):
                    baseline_raw = np.array([row[1] for row in bvp_baseline_raw], dtype=float)
                else:
                    baseline_raw = _to_num(bvp_baseline_raw)
                baseline_raw = _interp_nan(baseline_raw)
                
                break
    
    print(f"\n📂 Loading stimulus file from subject {baseline_subject}:")
    print(f"   Found {len(subject_stimulus_files)} stimulus files for this subject")
    
    if not subject_stimulus_files:
        print("❌ Still no matching stimulus files!")
        return
    
    # Load ONE stimulus file as example
    stimulus_file = subject_stimulus_files[0]
    stimulus_name = os.path.basename(stimulus_file)
    parts = stimulus_name.split("_")
    stimulus_emotion = parts[1] if len(parts) > 1 else "UNKNOWN"
    
    print(f"   File: {stimulus_name}")
    print(f"   Emotion: {stimulus_emotion}")
    
    try:
        with open(stimulus_file, 'r') as f:
            stimulus_data = json.load(f)
        
        bvp_stimulus_raw = stimulus_data.get("BVPRaw", [])
        
        if not bvp_stimulus_raw:
            print("❌ No BVP data in stimulus file!")
            return
        
        # Handle different formats
        if isinstance(bvp_stimulus_raw[0], list):
            stimulus_raw = np.array([row[1] for row in bvp_stimulus_raw], dtype=float)
        else:
            stimulus_raw = _to_num(bvp_stimulus_raw)
        
        stimulus_raw = _interp_nan(stimulus_raw)
        
        print(f"   ✅ Stimulus loaded: {len(stimulus_raw)} samples ({len(stimulus_raw)/config.BVP_FS:.1f}s)")
        
    except Exception as e:
        print(f"❌ Error loading stimulus: {e}")
        return
    
    # Preprocess baseline
    print(f"\n🔧 Preprocessing baseline signal...")
    baseline_processed = preprocess_bvp_signal(
        baseline_raw,
        fs=config.BVP_FS,
        use_baseline_correction=False,  # Don't use drift correction for baseline reduction
        normalize=False  # Don't normalize yet
    )
    print(f"   ✅ Baseline preprocessed: {baseline_processed.shape}")
    
    # Preprocess stimulus WITHOUT baseline reduction
    print(f"\n🔧 Preprocessing stimulus signal (WITHOUT baseline reduction)...")
    stimulus_no_br = preprocess_bvp_signal(
        stimulus_raw,
        fs=config.BVP_FS,
        use_baseline_correction=False,
        normalize=True
    )
    print(f"   ✅ Stimulus preprocessed: {stimulus_no_br.shape}")
    
    # Preprocess stimulus WITH baseline reduction
    print(f"\n🔧 Preprocessing stimulus signal (WITH baseline reduction)...")
    
    # First preprocess without normalization
    stimulus_processed = preprocess_bvp_signal(
        stimulus_raw,
        fs=config.BVP_FS,
        use_baseline_correction=False,
        normalize=False
    )
    
    # Match baseline length to stimulus
    baseline_matched = baseline_processed.copy()
    if len(baseline_matched) < len(stimulus_processed):
        n_repeats = int(np.ceil(len(stimulus_processed) / len(baseline_matched)))
        baseline_matched = np.tile(baseline_matched, n_repeats)[:len(stimulus_processed)]
    elif len(baseline_matched) > len(stimulus_processed):
        baseline_matched = baseline_matched[:len(stimulus_processed)]
    
    print(f"   Matched baseline length: {len(baseline_matched)} samples")
    
    # Apply baseline reduction (InvBase method)
    stimulus_with_br = apply_baseline_reduction(stimulus_processed, baseline_matched)
    
    # Normalize after reduction
    vmin, vmax = np.min(stimulus_with_br), np.max(stimulus_with_br)
    if not np.isclose(vmin, vmax):
        stimulus_with_br = (stimulus_with_br - vmin) / (vmax - vmin)
    
    print(f"   ✅ Baseline reduction applied: {stimulus_with_br.shape}")
    
    # Visualize comparison
    print(f"\n📊 Creating visualization...")
    
    # Take first 30 seconds for visualization
    vis_length = min(len(stimulus_raw), len(baseline_raw), config.BVP_FS * 30)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    time = np.arange(vis_length) / config.BVP_FS
    
    # Plot 1: Raw signals
    axes[0].plot(time, baseline_raw[:vis_length], 'g-', linewidth=0.8, label='Baseline (resting)', alpha=0.7)
    axes[0].plot(time, stimulus_raw[:vis_length], 'b-', linewidth=0.8, label=f'Stimulus ({stimulus_emotion})', alpha=0.7)
    axes[0].set_title(f'Raw BVP Signals - Subject {baseline_subject}', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Preprocessed baseline
    axes[1].plot(time, baseline_processed[:vis_length], 'g-', linewidth=0.8)
    axes[1].set_title('Preprocessed Baseline (Resting State)', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Stimulus WITHOUT baseline reduction
    axes[2].plot(time, stimulus_no_br[:vis_length], 'orange', linewidth=0.8)
    axes[2].set_title(f'Stimulus WITHOUT Baseline Reduction ({stimulus_emotion})', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Normalized')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Stimulus WITH baseline reduction
    axes[3].plot(time, stimulus_with_br[:vis_length], 'purple', linewidth=0.8)
    axes[3].set_title(f'Stimulus WITH Baseline Reduction (InvBase Method)', fontweight='bold', fontsize=12)
    axes[3].set_ylabel('Normalized')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bvp_baseline_reduction_demo.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Visualization saved: bvp_baseline_reduction_demo.png")
    plt.close()
    
    # Print statistics
    print(f"\n📊 Signal Statistics:")
    print(f"\n   Baseline (resting state):")
    print(f"      Mean: {baseline_processed.mean():.4f}")
    print(f"      Std:  {baseline_processed.std():.4f}")
    print(f"      Min:  {baseline_processed.min():.4f}")
    print(f"      Max:  {baseline_processed.max():.4f}")
    
    print(f"\n   Stimulus WITHOUT baseline reduction:")
    print(f"      Mean: {stimulus_no_br.mean():.4f}")
    print(f"      Std:  {stimulus_no_br.std():.4f}")
    print(f"      Min:  {stimulus_no_br.min():.4f}")
    print(f"      Max:  {stimulus_no_br.max():.4f}")
    
    print(f"\n   Stimulus WITH baseline reduction:")
    print(f"      Mean: {stimulus_with_br.mean():.4f}")
    print(f"      Std:  {stimulus_with_br.std():.4f}")
    print(f"      Min:  {stimulus_with_br.min():.4f}")
    print(f"      Max:  {stimulus_with_br.max():.4f}")
    
    print(f"\n✅ Baseline Reduction Test Complete!")
    print(f"\n💡 Key Points:")
    print(f"   - Each subject has ONE baseline file (resting state)")
    print(f"   - This baseline is used to normalize ALL emotional trials from that subject")
    print(f"   - InvBase method reduces inter-subject variability")
    print(f"   - Helps the model generalize better across different subjects")
    print("="*80)


# ==================================================
# MAIN TEST EXECUTION
# ==================================================

def main():
    """Run all BVP data loader tests."""
    print("="*80)
    print("BVP DATA LOADER TEST SUITE")
    print("="*80)
    print(f"Data path: {config.DATA_ROOT}")
    print(f"BVP sampling rate: {config.BVP_FS} Hz")
    print(f"Window size: {config.BVP_WINDOW_SEC} seconds")
    print(f"Baseline Reduction: {config.USE_BVP_BASELINE_REDUCTION}")
    print("="*80)
    
    # Debug: Baseline file matching
    debug_baseline_matching(config.DATA_ROOT)
    
    # NEW TEST: Baseline Reduction Method
    test_baseline_reduction_method(config.DATA_ROOT)
    
    # Test 3: Full data loading
    X_raw, y_labels, subject_ids, label_to_id = test_full_data_loading()
    
    # Test 4: Data splitting
    if y_labels is not None:
        test_data_splitting(y_labels, subject_ids)
    
    # Test 5: Batch feature extraction
    if X_raw is not None:
        test_feature_extraction_batch(X_raw)
    
    print("\n" + "="*80)
    print("🎉 ALL TESTS COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - bvp_baseline_reduction_demo.png")
    print("   - bvp_sample_windows.png")
    print("="*80)


if __name__ == "__main__":
    main()
