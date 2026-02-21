"""
    Data Loaders Module
    ===================
    Functions for loading and processing EEG data.
"""

import os
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd


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


def load_preprocessed_eeg_data(data_root, config):
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
