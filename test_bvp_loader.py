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
    butter_lowpass,
    wavelet_denoise,
    baseline_correct
)


# ==================================================
# CONFIGURATION
# ==================================================

class TestConfig:
    """Configuration for testing BVP data loader."""
    # Data path - CHANGE THIS TO YOUR DATA PATH
    DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"  # Path to your BVP JSON files
    
    # BVP parameters
    BVP_FS = 64  # Samsung Watch sampling frequency
    BVP_WINDOW_SEC = 10  # 10-second windows
    BVP_OVERLAP = 0.0  # No overlap
    
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

def test_single_file_loading():
    """Test loading a single BVP JSON file."""
    print("\n" + "="*80)
    print("TEST 1: SINGLE FILE LOADING")
    print("="*80)
    
    # Find first BVP file
    import glob
    patterns = [
        os.path.join(config.DATA_ROOT, "*_STIMULUS_SAMSUNG_WATCH.json"),
        os.path.join(config.DATA_ROOT, "*_STIMULUS_EMPATICA.json")
    ]
    files = [p for pat in patterns for p in glob.glob(pat)]
    
    if not files:
        print("❌ No BVP files found!")
        return None
    
    test_file = files[0]
    print(f"Testing file: {os.path.basename(test_file)}")
    
    # Load JSON
    import json
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    # Check structure
    print(f"\n📋 JSON Keys: {list(data.keys())[:10]}")
    
    if "BVP" in data:
        bvp_data = data["BVP"]
        print(f"✅ BVP data found!")
        print(f"   Type: {type(bvp_data)}")
        print(f"   Length: {len(bvp_data)}")
        
        if len(bvp_data) > 0:
            print(f"   First element: {bvp_data[0]}")
            print(f"   First element type: {type(bvp_data[0])}")
            
            # Extract values
            if isinstance(bvp_data[0], list):
                bvp_values = np.array([row[1] for row in bvp_data], dtype=float)
                print(f"   Format: [[timestamp, value], ...]")
            else:
                bvp_values = np.array(bvp_data, dtype=float)
                print(f"   Format: [value1, value2, ...]")
            
            print(f"   Signal length: {len(bvp_values)} samples")
            print(f"   Duration: {len(bvp_values) / config.BVP_FS:.2f} seconds")
            print(f"   Value range: [{np.min(bvp_values):.3f}, {np.max(bvp_values):.3f}]")
            
            return bvp_values
    else:
        print("❌ No 'BVP' key found in JSON!")
        print(f"   Available keys: {list(data.keys())}")
        return None


def test_preprocessing_pipeline(bvp_raw):
    """Test preprocessing pipeline with visualization."""
    print("\n" + "="*80)
    print("TEST 2: PREPROCESSING PIPELINE")
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
    print("TEST 3: FEATURE EXTRACTION")
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
    print("TEST 4: FULL DATA LOADING PIPELINE")
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
    print("TEST 5: DATA SPLITTING")
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
    print("TEST 6: BATCH FEATURE EXTRACTION")
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
    print("="*80)
    
    # Test 1: Single file loading
    bvp_raw = test_single_file_loading()
    
    # Test 2: Preprocessing pipeline
    bvp_processed = test_preprocessing_pipeline(bvp_raw)
    
    # Test 3: Feature extraction (single window)
    test_feature_extraction(bvp_processed)
    
    # Test 4: Full data loading
    X_raw, y_labels, subject_ids, label_to_id = test_full_data_loading()
    
    # Test 5: Data splitting
    if y_labels is not None:
        test_data_splitting(y_labels, subject_ids)
    
    # Test 6: Batch feature extraction
    if X_raw is not None:
        test_feature_extraction_batch(X_raw)
    
    print("\n" + "="*80)
    print("🎉 ALL TESTS COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - bvp_preprocessing_steps.png")
    print("   - bvp_sample_windows.png")
    print("="*80)


if __name__ == "__main__":
    main()
