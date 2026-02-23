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


def butter_highpass(cutoff_hz, fs, order=6):
    """
    Design Butterworth highpass filter.
    
    Args:
        cutoff_hz: Cutoff frequency in Hz
        fs: Sampling frequency
        order: Filter order (default: 6)
    
    Returns:
        b, a: Filter coefficients
    
    Reference:
        Used to remove DC offset and very low-frequency drift from BVP signals
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_bandpass(lowcut_hz, highcut_hz, fs, order=6):
    """
    Design Butterworth bandpass filter.
    
    Args:
        lowcut_hz: Lower cutoff frequency in Hz
        highcut_hz: Upper cutoff frequency in Hz
        fs: Sampling frequency
        order: Filter order (default: 6)
    
    Returns:
        b, a: Filter coefficients
    
    Reference:
        Combines highpass and lowpass filtering in one step
    """
    nyq = 0.5 * fs
    low = lowcut_hz / nyq
    high = highcut_hz / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
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
    
    NOTE: This is DIFFERENT from Baseline Reduction (InvBase method).
    - Baseline Correction: Removes drift within the same signal
    - Baseline Reduction: Uses separate baseline recordings to normalize
    
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


def apply_baseline_reduction(signal, baseline, eps=1e-12):
    """
    Apply Baseline Reduction using InvBase method (similar to EEG).
    
    This method uses SEPARATE baseline recordings (resting state) to normalize
    the trial signal and reduce inter-subject variability in BVP patterns.
    
    Method: Divide trial FFT by baseline FFT in frequency domain
    
    Args:
        signal: Trial BVP signal (1D array)
        baseline: Baseline BVP signal from resting state (1D array)
        eps: Small constant to avoid division by zero (default: 1e-12)
    
    Returns:
        Baseline-reduced signal (same length as input)
    
    Reference:
        InvBase method adapted for BVP signals to reduce inter-subject variability
        Similar to EEG baseline reduction for normalizing physiological patterns
    
    Notes:
        - Requires separate baseline recordings per subject
        - Reduces subject-specific cardiovascular patterns
        - Helps model generalize across subjects
    """
    signal = np.asarray(signal, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    
    # FFT of trial signal
    FFT_trial = np.fft.rfft(signal, axis=0)
    
    # FFT of baseline signal
    FFT_baseline = np.fft.rfft(baseline, axis=0)
    
    # Divide trial by baseline (with epsilon to avoid division by zero)
    FFT_reduced = FFT_trial / (np.abs(FFT_baseline) + eps)
    
    # Inverse FFT to get time-domain signal
    signal_reduced = np.fft.irfft(FFT_reduced, n=len(signal), axis=0)
    
    return signal_reduced.astype(np.float32)


def preprocess_bvp_signal(bvp_raw, fs, highcut_hz=15.0, lowcut_hz=0.5, filter_order=6, 
                          wavelet="db4", denoise_level=4, use_baseline_correction=True, 
                          use_highpass=True, normalize=True):
    """
    Complete BVP preprocessing pipeline.
    
    Pipeline:
    0. Highpass filtering (OPTIONAL - remove DC offset and very low-frequency drift)
    1. Lowpass filtering (remove high-frequency noise)
    2. Wavelet denoising (remove residual noise)
    3. Baseline drift correction (OPTIONAL - remove low-frequency drift)
    4. Normalization and standardization
    
    Args:
        bvp_raw: Raw BVP signal (1D array)
        fs: Sampling frequency
        highcut_hz: Lowpass filter cutoff frequency (default: 15 Hz)
        lowcut_hz: Highpass filter cutoff frequency (default: 0.5 Hz)
        filter_order: Butterworth filter order (default: 6)
        wavelet: Wavelet family for denoising (default: "db4")
        denoise_level: Wavelet decomposition level (default: 4)
        use_baseline_correction: Whether to apply baseline drift correction (default: True)
        use_highpass: Whether to apply highpass filter to remove DC offset (default: True)
        normalize: Whether to normalize to [0,1] (default: True)
    
    Returns:
        Preprocessed BVP signal (standardized with mean=0, std=1)
    
    Notes:
        - Highpass at 0.5 Hz removes DC shifts and very slow drift
        - Lowpass at 15 Hz removes high-frequency noise
        - Typical BVP signal range: 0.5-4 Hz (30-240 bpm heart rate)
    """
    # Step 0: Highpass filtering (remove DC offset and very low-frequency drift)
    if use_highpass:
        # Use bandpass filter (combines highpass + lowpass in one step)
        b, a = butter_bandpass(lowcut_hz, highcut_hz, fs, order=filter_order)
        bvp_filtered = filtfilt(b, a, bvp_raw)
    else:
        # Use only lowpass filter
        b, a = butter_lowpass(highcut_hz, fs, order=filter_order)
        bvp_filtered = filtfilt(b, a, bvp_raw)
    
    # Step 1: Wavelet denoising
    bvp_denoised = wavelet_denoise(bvp_filtered, wavelet=wavelet, level=denoise_level)
    
    # Step 2: Baseline correction (OPTIONAL)
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
    
    # Step 3: Standardization (z-score)
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
                Optional: USE_BVP_BASELINE_REDUCTION (default: False)
                Optional: BVP_DEVICE (default: 'both') - 'samsung_watch', 'empatica', or 'both'
    
    Returns:
        X_raw: (N, T) - Raw BVP windows (preprocessed)
        y_labels: (N,) - Class labels as integers
        subject_ids: (N,) - Subject IDs for each window
        label_to_id: Dictionary mapping label names to integers
    """
    print("\n" + "="*80)
    print("LOADING BVP DATA (SAMSUNG WATCH / EMPATICA)")
    print("="*80)
    
    # Get device selection from config
    bvp_device = getattr(config, 'BVP_DEVICE', 'both').lower()
    print(f"🔧 BVP Device Selection: {bvp_device.upper()}")
    
    # Search for BVP files based on device selection
    all_patterns = []
    
    if bvp_device in ['samsung_watch', 'both']:
        all_patterns.extend([
            os.path.join(data_root, "*_STIMULUS_SAMSUNG_WATCH.json"),
            os.path.join(data_root, "*", "*_STIMULUS_SAMSUNG_WATCH.json")
        ])
    
    if bvp_device in ['empatica', 'both']:
        all_patterns.extend([
            os.path.join(data_root, "*_STIMULUS_EMPATICA.json"),
            os.path.join(data_root, "*", "*_STIMULUS_EMPATICA.json")
        ])
    
    if not all_patterns:
        raise ValueError(f"Invalid BVP_DEVICE: '{bvp_device}'. Choose 'samsung_watch', 'empatica', or 'both'")
    
    files = sorted({p for pat in all_patterns for p in glob.glob(pat)})
    
    # Count files by device type
    samsung_files = [f for f in files if 'SAMSUNG_WATCH' in f]
    empatica_files = [f for f in files if 'EMPATICA' in f]
    
    print(f"Found {len(files)} BVP files total:")
    if samsung_files:
        print(f"   📱 Samsung Watch: {len(samsung_files)} files")
    if empatica_files:
        print(f"   ⌚ Empatica: {len(empatica_files)} files")
    
    if len(files) == 0:
        print("\n❌ ERROR: No BVP files found!")
        print(f"   Searched in: {data_root}")
        if os.path.exists(data_root):
            print(f"   Directory contents: {os.listdir(data_root)[:10]}")
        raise ValueError("No BVP files found. Check DATA_ROOT path.")
    
    print(f"\n📁 Sample files:")
    for f in files[:3]:
        device_type = "📱 Samsung" if 'SAMSUNG_WATCH' in f else "⌚ Empatica"
        print(f"   {device_type}: {os.path.basename(f)}")
    
    # Check preprocessing flags
    use_baseline_correction = getattr(config, 'USE_BVP_BASELINE_CORRECTION', False)
    use_baseline_reduction = getattr(config, 'USE_BVP_BASELINE_REDUCTION', False)
    
    if use_baseline_correction:
        print(f"\n🔧 BVP Baseline Correction (drift removal): ENABLED")
    else:
        print(f"\n🔧 BVP Baseline Correction (drift removal): DISABLED")
    
    if use_baseline_reduction:
        print(f"🔧 BVP Baseline Reduction (InvBase method): ENABLED")
    else:
        print(f"🔧 BVP Baseline Reduction (InvBase method): DISABLED")
    
    # Load baseline recordings if baseline reduction is enabled
    baseline_dict = {}
    if use_baseline_reduction:
        print(f"\n📂 Loading baseline recordings...")
        
        # Search for baseline files based on device selection
        # Updated patterns to match actual baseline file naming convention
        baseline_patterns = []
        if bvp_device in ['samsung_watch', 'both']:
            baseline_patterns.extend([
                os.path.join(data_root, "*_BASELINE_SAMSUNG_WATCH.json"),
                os.path.join(data_root, "*", "*_BASELINE_SAMSUNG_WATCH.json"),
                os.path.join(data_root, "*_BASELINE_STIMULUS_SAMSUNG_WATCH.json"),  # ✅ Added
                os.path.join(data_root, "*", "*_BASELINE_STIMULUS_SAMSUNG_WATCH.json")  # ✅ Added
            ])
        
        if bvp_device in ['empatica', 'both']:
            baseline_patterns.extend([
                os.path.join(data_root, "*_BASELINE_EMPATICA.json"),
                os.path.join(data_root, "*", "*_BASELINE_EMPATICA.json"),
                os.path.join(data_root, "*_BASELINE_STIMULUS_EMPATICA.json"),  # ✅ Added
                os.path.join(data_root, "*", "*_BASELINE_STIMULUS_EMPATICA.json")  # ✅ Added
            ])
        
        baseline_files = sorted({p for pat in baseline_patterns for p in glob.glob(pat)})
        print(f"   Found {len(baseline_files)} baseline files")
        
        # Debug: Show first few baseline files found
        if baseline_files:
            print(f"   📋 Sample baseline files:")
            for bf in baseline_files[:3]:
                print(f"      - {os.path.basename(bf)}")
        
        for bpath in baseline_files:
            bname = os.path.basename(bpath)
            parts = bname.split("_")
            if len(parts) < 2:
                continue
            subject = parts[0]
            
            try:
                with open(bpath, "r") as f:
                    bdata = json.load(f)
                
                bvp_baseline = bdata.get("BVP", [])
                if not bvp_baseline:
                    continue
                
                # Handle different formats
                if isinstance(bvp_baseline[0], list):
                    baseline_raw = np.array([row[1] for row in bvp_baseline], dtype=float)
                else:
                    baseline_raw = _to_num(bvp_baseline)
                
                baseline_raw = _interp_nan(baseline_raw)
                
                # Apply same preprocessing to baseline
                baseline_processed = preprocess_bvp_signal(
                    baseline_raw,
                    fs=config.BVP_FS,
                    highcut_hz=15.0,
                    lowcut_hz=0.5,
                    filter_order=6,
                    wavelet="db4",
                    denoise_level=4,
                    use_baseline_correction=use_baseline_correction,
                    use_highpass=True,
                    normalize=False  # Don't normalize baseline yet
                )
                
                baseline_dict[subject] = baseline_processed
                
            except Exception as e:
                continue
        
        print(f"   Loaded baselines for {len(baseline_dict)} subjects")
        if len(baseline_dict) > 0:
            print(f"   Subjects with baselines: {sorted(baseline_dict.keys())}")
    
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
        'parse_error': 0,
        'no_baseline': 0
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
        
        # Check if baseline is required but missing
        if use_baseline_reduction and subject not in baseline_dict:
            skipped_reasons['no_baseline'] += 1
            continue
        
        superclass = config.SUPERCLASS_MAP[emotion]
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # 🔍 DEBUG: Inspect JSON structure for first few files
            if successful_files < 3:
                print(f"\n🔍 DEBUG - Inspecting file: {fname}")
                print(f"   📋 Available JSON fields: {list(data.keys())}")
                
                # Check if BVP field exists and show its structure
                if "BVP" in data:
                    bvp_test = data["BVP"]
                    print(f"   ✅ 'BVP' field found!")
                    print(f"      Type: {type(bvp_test)}")
                    print(f"      Length: {len(bvp_test) if isinstance(bvp_test, (list, dict)) else 'N/A'}")
                    if isinstance(bvp_test, list) and len(bvp_test) > 0:
                        print(f"      First element type: {type(bvp_test[0])}")
                        print(f"      First 3 elements: {bvp_test[:3]}")
                else:
                    print(f"   ❌ 'BVP' field NOT found!")
                    print(f"   💡 Checking alternative field names...")
                    
                    # Check for alternative field names
                    possible_fields = ["PPG", "HR", "SAMSUNG_BVP", "HeartRate", "BVP_RAW", "ppg", "bvp"]
                    for field in possible_fields:
                        if field in data:
                            print(f"      ✅ Found '{field}' field (length: {len(data[field]) if isinstance(data[field], (list, dict)) else 'N/A'})")
                            if isinstance(data[field], list) and len(data[field]) > 0:
                                print(f"         Sample: {data[field][:3]}")
            
            # Extract BVP data
            bvp_data = data.get("BVP", [])
            
            if not bvp_data:
                skipped_reasons['no_data'] += 1
                continue
            
            # Handle different BVP data formats
            if isinstance(bvp_data[0], list):
                bvp_raw = np.array([row[1] for row in bvp_data], dtype=float)
            else:
                bvp_raw = _to_num(bvp_data)
            
            # Interpolate any NaN values
            bvp_raw = _interp_nan(bvp_raw)
            
            L = len(bvp_raw)
            if L < win_samples:
                skipped_reasons['insufficient_length'] += 1
                continue
            
            # Apply preprocessing pipeline
            bvp_processed = preprocess_bvp_signal(
                bvp_raw, 
                fs=config.BVP_FS,
                highcut_hz=15.0,
                lowcut_hz=0.5,
                filter_order=6,
                wavelet="db4",
                denoise_level=4,
                use_baseline_correction=use_baseline_correction,
                use_highpass=True,
                normalize=not use_baseline_reduction  # Don't normalize if using baseline reduction
            )
            
            # Apply baseline reduction if enabled
            if use_baseline_reduction:
                subject_baseline = baseline_dict[subject]
                
                # Match baseline length to signal
                if len(subject_baseline) < len(bvp_processed):
                    # Repeat baseline to match signal length
                    n_repeats = int(np.ceil(len(bvp_processed) / len(subject_baseline)))
                    subject_baseline = np.tile(subject_baseline, n_repeats)[:len(bvp_processed)]
                elif len(subject_baseline) > len(bvp_processed):
                    # Truncate baseline
                    subject_baseline = subject_baseline[:len(bvp_processed)]
                
                # Apply InvBase method
                bvp_processed = apply_baseline_reduction(bvp_processed, subject_baseline)
                
                # Normalize after reduction
                vmin, vmax = np.min(bvp_processed), np.max(bvp_processed)
                if not np.isclose(vmin, vmax):
                    bvp_processed = (bvp_processed - vmin) / (vmax - vmin)
            
            # Standardize (z-score)
            bvp_mean = bvp_processed.mean()
            bvp_std = bvp_processed.std()
            if bvp_std > 1e-8:
                bvp_processed = (bvp_processed - bvp_mean) / bvp_std
            
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
