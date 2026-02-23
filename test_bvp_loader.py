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
    """Run baseline reduction test only."""
    print("="*80)
    print("BVP BASELINE REDUCTION TEST")
    print("="*80)
    print(f"Data path: {config.DATA_ROOT}")
    print(f"BVP sampling rate: {config.BVP_FS} Hz")
    print(f"Device: {config.BVP_DEVICE.upper()}")
    print(f"Baseline Reduction: {config.USE_BVP_BASELINE_REDUCTION}")
    print("="*80)
    
    # Run baseline reduction test
    test_baseline_reduction_method(config.DATA_ROOT, config.BVP_DEVICE)
    
    print("\n" + "="*80)
    print("🎉 BASELINE REDUCTION TEST COMPLETE!")
    print("="*80)
    print("\n📁 Generated File:")
    print("   - bvp_baseline_reduction_demo.png")
    print("="*80)


if __name__ == "__main__":
    main()
