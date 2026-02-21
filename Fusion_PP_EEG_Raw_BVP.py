"""
    Hybrid Multimodal Emotion Recognition: Preprocessed EEG + Raw BVP Fusion
    =========================================================================
    
    This script combines:
    1. EEG from PREPROCESSED dataset (KNN_ASR cleaned)
    2. BVP from RAW dataset (original signals)
    3. Multimodal fusion with cross-modal attention

    Pipeline:
    ---------
    1. Load preprocessed EEG data (with baseline reduction)
    2. Load raw BVP data (with standard preprocessing)
    3. Train EEG BiLSTM model
    4. Train BVP EMCNN model
    5. Train fusion model with cross-modal attention

    Dataset Paths:
    -------------
    - EEG: /kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR
    - BVP: /kaggle/input/datasets/ruchiabey/emognition

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
# CONFIGURATION
# ==================================================

class Config:
    """Shared configuration for all models."""
    # Paths - HYBRID DATASETS
    EEG_DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"  # Preprocessed EEG
    BVP_DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognition"  # Raw BVP
    
    # Common parameters
    NUM_CLASSES = 4  # 4-class emotion system
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Classification Mode
    USE_DUAL_BINARY = False
    CLASSIFY_WHOLE_CLIPS = False
    CLIP_AGGREGATION_METHOD = "mean"
    
    # Baseline reduction (for preprocessed EEG)
    USE_BASELINE_REDUCTION = True  # Apply InvBase method to reduce inter-subject variability
    
    # Data split mode
    SUBJECT_INDEPENDENT = True  # Subject-independent split
    CLIP_INDEPENDENT = False
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
    
    # EEG parameters (from preprocessed dataset)
    EEG_FS = 256.0
    EEG_CHANNELS = 4
    EEG_FEATURES = 26
    EEG_WINDOW_SEC = 10.0
    EEG_OVERLAP = 0.0
    EEG_BATCH_SIZE = 64
    EEG_EPOCHS = 150
    EEG_LR = 1e-3
    EEG_PATIENCE = 20
    EEG_CHECKPOINT = "best_eeg_pp_model.pt"
    
    # BVP parameters (from raw dataset)
    BVP_FS = 64
    BVP_WINDOW_SEC = 10
    BVP_WINDOW_SIZE = BVP_FS * BVP_WINDOW_SEC
    BVP_FEATURES = 5
    BVP_OVERLAP = 0.0
    BVP_BATCH_SIZE = 128
    BVP_EPOCHS = 50
    BVP_LR = 1e-3
    BVP_PATIENCE = 10
    BVP_CHECKPOINT = "best_bvp_raw_model.pt"
    
    # Fusion parameters
    FUSION_SHARED_DIM = 128
    FUSION_NUM_HEADS = 4
    FUSION_DROPOUT = 0.1
    FUSION_BATCH_SIZE = 64
    FUSION_EPOCHS = 40
    FUSION_LR = 1e-3
    FUSION_PATIENCE = 10
    FUSION_CHECKPOINT = "best_fusion_hybrid_model.pt"
    
    # File outputs
    SPLIT_FILE = "hybrid_data_split.npz"
    
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
print(f"EEG Dataset: {config.EEG_DATA_ROOT}")
print(f"BVP Dataset: {config.BVP_DATA_ROOT}")


# ==================================================
# PART 1: DATA LOADING & PREPROCESSING
# ==================================================

# --- Utility Functions ---

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
    """Apply InvBase method: divide trial FFT by baseline FFT."""
    FFT_trial = np.fft.rfft(signal, axis=0)
    FFT_baseline = np.fft.rfft(baseline, axis=0)
    FFT_reduced = FFT_trial / (np.abs(FFT_baseline) + eps)
    signal_reduced = np.fft.irfft(FFT_reduced, n=len(signal), axis=0)
    return signal_reduced.astype(np.float32)


# --- EEG Data Loading (PREPROCESSED) ---

def load_eeg_data_preprocessed(data_root):
    """Load PREPROCESSED EEG data from KNN_ASR cleaned files."""
    print("\n" + "="*80)
    print("LOADING PREPROCESSED EEG DATA")
    print("="*80)
    print(f"Source: {data_root}")
    
    patterns = [
        os.path.join(data_root, "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned", "*_STIMULUS_MUSE_cleaned.json")
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    print(f"Found {len(files)} preprocessed MUSE files")
    
    if len(files) == 0:
        raise ValueError("No preprocessed MUSE files found!")
    
    # Load baseline files
    baseline_dict = {}
    if config.USE_BASELINE_REDUCTION:
        print(f"\n🔧 Baseline Reduction: ENABLED")
        for fpath in files:
            fname = os.path.basename(fpath)
            parts = fname.split("_")
            if len(parts) < 2 or "BASELINE" in fname:
                continue
            subject = parts[0]
            if subject in baseline_dict:
                continue
            
            baseline_patterns = [
                os.path.join(data_root, f"{subject}_BASELINE_STIMULUS_MUSE.json"),
                os.path.join(data_root, subject, f"{subject}_BASELINE_STIMULUS_MUSE.json"),
                os.path.join(data_root, subject, f"{subject}_BASELINE_STIMULUS_MUSE_cleaned", f"{subject}_BASELINE_STIMULUS_MUSE_cleaned.json")
            ]
            
            for baseline_path in baseline_patterns:
                if os.path.exists(baseline_path):
                    try:
                        with open(baseline_path, "r") as f:
                            data = json.load(f)
                        tp9 = _interp_nan(_to_num(data.get("RAW_TP9", [])))
                        af7 = _interp_nan(_to_num(data.get("RAW_AF7", [])))
                        af8 = _interp_nan(_to_num(data.get("RAW_AF8", [])))
                        tp10 = _interp_nan(_to_num(data.get("RAW_TP10", [])))
                        L = min(len(tp9), len(af7), len(af8), len(tp10))
                        if L > 0:
                            baseline_signal = np.stack([tp9[:L], af7[:L], af8[:L], tp10[:L]], axis=1)
                            baseline_signal = baseline_signal - np.nanmean(baseline_signal, axis=0, keepdims=True)
                            baseline_dict[subject] = baseline_signal
                    except:
                        pass
                    break
        print(f"   ✅ Loaded {len(baseline_dict)} baseline recordings")
    
    all_windows, all_labels, all_subjects, all_file_ids, all_clip_names = [], [], [], [], []
    win_samples = int(config.EEG_WINDOW_SEC * config.EEG_FS)
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))
    
    for file_idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        if len(parts) < 2 or "BASELINE" in fname:
            continue
            
        subject = parts[0]
        emotion = parts[1].upper()
        clip_name = f"{subject}_{emotion}"
        
        if emotion not in config.SUPERCLASS_MAP:
            continue
        superclass = config.SUPERCLASS_MAP[emotion]
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            tp9 = _interp_nan(_to_num(data.get("RAW_TP9", [])))
            af7 = _interp_nan(_to_num(data.get("RAW_AF7", [])))
            af8 = _interp_nan(_to_num(data.get("RAW_AF8", [])))
            tp10 = _interp_nan(_to_num(data.get("RAW_TP10", [])))
            
            L = min(len(tp9), len(af7), len(af8), len(tp10))
            if L == 0 or L < win_samples:
                continue
            
            # Quality filtering
            hsi_tp9 = _to_num(data.get("HSI_TP9", []))[:L]
            hsi_af7 = _to_num(data.get("HSI_AF7", []))[:L]
            hsi_af8 = _to_num(data.get("HSI_AF8", []))[:L]
            hsi_tp10 = _to_num(data.get("HSI_TP10", []))[:L]
            head_on = _to_num(data.get("HeadBandOn", []))[:L]
            
            mask = np.isfinite(tp9[:L]) & np.isfinite(af7[:L]) & np.isfinite(af8[:L]) & np.isfinite(tp10[:L])
            if len(head_on) == L and len(hsi_tp9) == L:
                quality_mask = (
                    (head_on == 1) &
                    np.isfinite(hsi_tp9) & (hsi_tp9 <= 2) &
                    np.isfinite(hsi_af7) & (hsi_af7 <= 2) &
                    np.isfinite(hsi_af8) & (hsi_af8 <= 2) &
                    np.isfinite(hsi_tp10) & (hsi_tp10 <= 2)
                )
                mask = mask & quality_mask
            
            tp9, af7, af8, tp10 = tp9[:L][mask], af7[:L][mask], af8[:L][mask], tp10[:L][mask]
            L = len(tp9)
            if L < win_samples:
                continue
            
            signal = np.stack([tp9, af7, af8, tp10], axis=1)
            signal = signal - np.nanmean(signal, axis=0, keepdims=True)
            
            # Apply baseline reduction if available
            if config.USE_BASELINE_REDUCTION and subject in baseline_dict:
                baseline_signal = baseline_dict[subject]
                common_len = min(len(signal), len(baseline_signal))
                signal = apply_baseline_reduction(signal[:common_len], baseline_signal[:common_len])
                L = len(signal)
            
            # Create windows
            for start in range(0, L - win_samples + 1, step_samples):
                window = signal[start:start + win_samples]
                if len(window) == win_samples:
                    all_windows.append(window)
                    all_labels.append(superclass)
                    all_subjects.append(subject)
                    all_file_ids.append(file_idx)
                    all_clip_names.append(clip_name)
        except:
            continue
    
    X_raw = np.stack(all_windows).astype(np.float32)
    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    y_labels = np.array([label_to_id[lab] for lab in all_labels], dtype=np.int64)
    subject_ids = np.array(all_subjects)
    file_ids = np.array(all_file_ids, dtype=np.int32)
    clip_names = np.array(all_clip_names)
    
    print(f"\n✅ Preprocessed EEG loaded: {X_raw.shape}")
    print(f"   Label distribution: {Counter(all_labels)}")
    
    return X_raw, y_labels, subject_ids, file_ids, label_to_id, clip_names


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


# --- BVP Data Loading (RAW) ---

def butter_lowpass(cutoff_hz, fs, order=6):
    """Design Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def wavelet_denoise(sig, wavelet="db4", level=4, thresh_scale=1.0):
    """Wavelet denoising for BVP."""
    sig = np.asarray(sig, dtype=float)
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = thresh_scale * sigma * np.sqrt(2 * np.log(len(sig)))
    
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], uthresh, mode="soft")
    
    rec = pywt.waverec(coeffs, wavelet)
    
    if len(rec) > len(sig):
        rec = rec[:len(sig)]
    elif len(rec) < len(sig):
        rec = np.pad(rec, (0, len(sig) - len(rec)), mode="edge")
    
    return rec


def baseline_correct(sig, fs, return_normalized=True):
    """Remove baseline drift from BVP signal."""
    sig = np.asarray(sig).astype(float)
    
    min_dist_samples = max(1, int(fs * 60.0 / 200.0))
    sig_range = np.max(sig) - np.min(sig)
    if np.isclose(sig_range, 0):
        sig_range = 1.0
    
    prominence = 0.03 * sig_range
    inv = -sig
    minima_idx, _ = find_peaks(inv, distance=min_dist_samples, prominence=prominence)
    
    if len(minima_idx) < 2:
        win = int(1.5 * fs)
        if win % 2 == 0:
            win += 1
        win = max(3, min(win, len(sig)))
        baseline = medfilt(sig, kernel_size=win)
        corrected = sig - baseline
    else:
        idx = np.concatenate(([0], minima_idx, [len(sig) - 1]))
        vals = np.concatenate(([sig[0]], sig[minima_idx], [sig[-1]]))
        baseline = np.interp(np.arange(len(sig)), idx, vals)
        corrected = sig - baseline
    
    if not return_normalized:
        return corrected
    
    vmin, vmax = np.min(corrected), np.max(corrected)
    if np.isclose(vmin, vmax):
        return np.zeros_like(corrected)
    else:
        return (corrected - vmin) / (vmax - vmin)


def load_bvp_data_raw(data_root):
    """Load RAW BVP data from Empatica files."""
    print("\n" + "="*80)
    print("LOADING RAW BVP DATA")
    print("="*80)
    print(f"Source: {data_root}")
    
    patterns = [os.path.join(data_root, "*_STIMULUS_EMPATICA.json"),
                os.path.join(data_root, "*", "*_STIMULUS_EMPATICA.json")]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    print(f"Found {len(files)} raw EMPATICA files")
    
    all_signals, all_labels, all_subjects, all_file_ids, all_clip_names = [], [], [], [], []
    win_samples = int(config.BVP_WINDOW_SEC * config.BVP_FS)
    step_samples = int(win_samples * (1.0 - config.BVP_OVERLAP))
    
    for file_idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        subject = parts[0]
        emotion = parts[1].upper()
        clip_name = f"{subject}_{emotion}"
        
        if emotion not in config.SUPERCLASS_MAP:
            continue
        superclass = config.SUPERCLASS_MAP[emotion]
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            rows = data.get("BVP", [])
            if not rows:
                continue
            
            bvp = np.array([row[1] for row in rows], dtype=float)
            
            # Preprocessing
            b, a = butter_lowpass(15.0, config.BVP_FS, order=6)
            bvp_filtered = filtfilt(b, a, bvp)
            bvp_denoised = wavelet_denoise(bvp_filtered, wavelet="db4", level=4)
            bvp_normalized = baseline_correct(bvp_denoised, config.BVP_FS, return_normalized=True)
            bvp_clip = (bvp_normalized - bvp_normalized.mean()) / (bvp_normalized.std() + 1e-8)
            
            # Slice into windows
            for start in range(0, len(bvp_clip) - win_samples + 1, step_samples):
                window = bvp_clip[start:start + win_samples]
                if len(window) == win_samples:
                    all_signals.append(window)
                    all_labels.append(superclass)
                    all_subjects.append(subject)
                    all_file_ids.append(file_idx)
                    all_clip_names.append(clip_name)
        except:
            continue
    
    X_raw = np.stack(all_signals).astype(np.float32)
    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    y_labels = np.array([label_to_id[lab] for lab in all_labels], dtype=np.int64)
    subject_ids = np.array(all_subjects)
    file_ids = np.array(all_file_ids, dtype=np.int32)
    clip_names = np.array(all_clip_names)
    
    print(f"\n✅ Raw BVP loaded: {X_raw.shape}")
    print(f"   Label distribution: {Counter(all_labels)}")
    
    return X_raw, y_labels, subject_ids, file_ids, label_to_id, clip_names


def extract_bvp_features(window):
    """Extract 5 handcrafted features from BVP window."""
    mean_val = np.mean(window)
    std_val = np.std(window)
    diff_std = np.std(np.diff(window))
    
    peaks, _ = find_peaks(window, distance=config.BVP_FS * 0.4)
    hr_proxy = len(peaks) / (config.BVP_WINDOW_SEC + 1e-6)
    
    p2p = np.max(window) - np.min(window)
    
    return np.array([mean_val, std_val, diff_std, hr_proxy, p2p], dtype=np.float32)


def moving_average_backward(x, s):
    """Backward cumulative moving average."""
    if s <= 1:
        return x
    c = np.cumsum(x)
    y = np.empty_like(x)
    for i in range(s - 1):
        y[i] = c[i] / (i + 1)
    y[s-1:] = (c[s-1:] - np.concatenate(([0], c[:-s]))) / s
    return y


# --- Data Splitting ---

def create_clip_based_split(eeg_clip_names, eeg_y, eeg_subjects, 
                            bvp_clip_names, bvp_y, bvp_subjects):
    """Create clip-based splits that ensure perfect alignment between EEG and BVP."""
    print("\n" + "="*80)
    print("CREATING HYBRID CLIP-BASED SPLIT")
    print("="*80)
    
    # Find common clips
    eeg_clips_set = set(eeg_clip_names)
    bvp_clips_set = set(bvp_clip_names)
    common_clips = eeg_clips_set & bvp_clips_set
    
    print(f"\n📊 Dataset Alignment:")
    print(f"   Preprocessed EEG clips: {len(eeg_clips_set)}")
    print(f"   Raw BVP clips: {len(bvp_clips_set)}")
    print(f"   ✅ Common clips (usable for fusion): {len(common_clips)}")
    print(f"   ⚠️  EEG-only clips: {len(eeg_clips_set - bvp_clips_set)}")
    print(f"   ⚠️  BVP-only clips: {len(bvp_clips_set - eeg_clips_set)}")
    
    if len(common_clips) == 0:
        raise ValueError("No common clips found between preprocessed EEG and raw BVP!")
    
    # Get subject and label for each common clip
    clip_to_subject = {}
    clip_to_label = {}
    
    for clip in common_clips:
        subject = clip.split('_')[0]
        clip_to_subject[clip] = subject
        mask = eeg_clip_names == clip
        if mask.any():
            clip_to_label[clip] = eeg_y[mask][0]
    
    # Subject-independent stratified split
    if config.SUBJECT_INDEPENDENT:
        print("\n  Strategy: SUBJECT-INDEPENDENT split")
        
        subject_to_clips = defaultdict(list)
        for clip, subject in clip_to_subject.items():
            subject_to_clips[subject].append(clip)
        
        subjects = sorted(subject_to_clips.keys())
        subject_to_label = {}
        for subject in subjects:
            subject_clips = subject_to_clips[subject]
            subject_labels = [clip_to_label[clip] for clip in subject_clips]
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
            n_test = max(1, int(len(class_subjects) * 0.15))
            n_val = max(1, int(len(class_subjects) * 0.15))
            
            test_subjects.extend(class_subjects[:n_test])
            val_subjects.extend(class_subjects[n_test:n_test+n_val])
            train_subjects.extend(class_subjects[n_test+n_val:])
        
        train_clips = [clip for subj in train_subjects for clip in subject_to_clips[subj]]
        val_clips = [clip for subj in val_subjects for clip in subject_to_clips[subj]]
        test_clips = [clip for subj in test_subjects for clip in subject_to_clips[subj]]
    
    print(f"\n📋 Split Summary:")
    print(f"   Train clips: {len(train_clips)}")
    print(f"   Val clips: {len(val_clips)}")
    print(f"   Test clips: {len(test_clips)}")
    
    split_data = {
        'train_clips': train_clips,
        'val_clips': val_clips,
        'test_clips': test_clips,
        'all_common_clips': list(common_clips),
    }
    
    return split_data


def get_clip_indices(clip_names_array, target_clips):
    """Get indices of windows that belong to target clips."""
    target_set = set(target_clips)
    indices = np.where(np.isin(clip_names_array, list(target_set)))[0]
    return indices


# ==================================================
# PART 2: MODEL ARCHITECTURES (from fusion code)
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


class DWConv(nn.Module):
    """Depthwise Separable Convolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


def make_branch():
    """Create a CNN branch for BVP."""
    layers = []
    c = 1
    for out in [16, 32, 64]:
        layers.append(DWConv(c, out))
        c = out
    layers.append(nn.AdaptiveAvgPool1d(1))
    return nn.Sequential(*layers)


class EMCNN(nn.Module):
    """Hybrid EMCNN for BVP."""
    def __init__(self, n_classes=4, feat_dim=5):
        super().__init__()
        self.b1 = make_branch()
        self.b2 = make_branch()
        self.b3 = make_branch()
        self.feat_fc = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x1, x2, x3, feats):
        f1 = self.b1(x1).flatten(1)
        f2 = self.b2(x2).flatten(1)
        f3 = self.b3(x3).flatten(1)
        ff = self.feat_fc(feats)
        return self.fc(torch.cat([f1, f2, f3, ff], dim=1))


class BVPEncoder(nn.Module):
    """BVP encoder for fusion."""
    def __init__(self, feat_dim=5):
        super().__init__()
        self.b1 = make_branch()
        self.b2 = make_branch()
        self.b3 = make_branch()
        self.feat_fc = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, x1, x2, x3, feats):
        f1 = self.b1(x1).flatten(1)
        f2 = self.b2(x2).flatten(1)
        f3 = self.b3(x3).flatten(1)
        ff = self.feat_fc(feats)
        return torch.cat([f1, f2, f3, ff], dim=1)


class CrossModalAttention(nn.Module):
    """Cross-modal attention between EEG and BVP."""
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_eeg_to_bvp = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_bvp_to_eeg = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_eeg = nn.LayerNorm(d_model)
        self.norm_bvp = nn.LayerNorm(d_model)

    def forward(self, h_eeg, h_bvp):
        h_eeg_ = h_eeg.unsqueeze(1)
        h_bvp_ = h_bvp.unsqueeze(1)
        eeg_ctx, _ = self.attn_eeg_to_bvp(query=h_eeg_, key=h_bvp_, value=h_bvp_)
        bvp_ctx, _ = self.attn_bvp_to_eeg(query=h_bvp_, key=h_eeg_, value=h_eeg_)
        h_eeg_out = self.norm_eeg(h_eeg_ + eeg_ctx).squeeze(1)
        h_bvp_out = self.norm_bvp(h_bvp_ + bvp_ctx).squeeze(1)
        return h_eeg_out, h_bvp_out


class GatedFusion(nn.Module):
    """Gated fusion mechanism."""
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())

    def forward(self, h_eeg, h_bvp):
        g = self.gate(torch.cat([h_eeg, h_bvp], dim=1))
        return g * h_eeg + (1 - g) * h_bvp


class WindowFusionEEGBVP(nn.Module):
    """Complete fusion model."""
    def __init__(self, eeg_encoder, bvp_encoder, n_classes=4, shared_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.bvp_encoder = bvp_encoder
        self.eeg_proj = nn.Sequential(nn.Linear(512, shared_dim), nn.LayerNorm(shared_dim), nn.ReLU())
        self.bvp_proj = nn.Sequential(nn.Linear(bvp_encoder.out_dim, shared_dim), nn.LayerNorm(shared_dim), nn.ReLU())
        self.cross_attn = CrossModalAttention(d_model=shared_dim, num_heads=num_heads, dropout=dropout)
        self.fusion = GatedFusion(shared_dim)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(shared_dim, n_classes)
        )

    def forward(self, eeg_x, bvp_x):
        H_eeg = self.eeg_encoder(eeg_x)
        H_bvp = self.bvp_encoder(*bvp_x)
        H_eeg = self.eeg_proj(H_eeg)
        H_bvp = self.bvp_proj(H_bvp)
        H_eeg, H_bvp = self.cross_attn(H_eeg, H_bvp)
        H_fused = self.fusion(H_eeg, H_bvp)
        return self.classifier(H_fused)


# ==================================================
# PART 3: DATASETS
# ==================================================

class BVPDataset(Dataset):
    """PyTorch dataset for BVP."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        w = self.X[idx]
        feats = extract_bvp_features(w)
        x1 = torch.tensor(w).float().unsqueeze(0)
        x2 = torch.tensor(moving_average_backward(w, 2)).float().unsqueeze(0)
        x3 = torch.tensor(w[::2]).float().unsqueeze(0)
        return x1, x2, x3, torch.tensor(feats), torch.tensor(self.y[idx], dtype=torch.long)


class MultimodalEEGBVPDataset(Dataset):
    """Multimodal dataset for fusion."""
    def __init__(self, eeg_X, eeg_y, eeg_clip_ids, bvp_X, bvp_clip_ids):
        self.samples = []
        eeg_by_clip = {}
        for x, y, cid in zip(eeg_X, eeg_y, eeg_clip_ids):
            eeg_by_clip.setdefault(cid, []).append((x, y))
        bvp_by_clip = {}
        for x, cid in zip(bvp_X, bvp_clip_ids):
            bvp_by_clip.setdefault(cid, []).append(x)

        for cid in eeg_by_clip:
            if cid not in bvp_by_clip:
                continue
            eeg_w = eeg_by_clip[cid]
            bvp_w = bvp_by_clip[cid]
            L = min(len(eeg_w), len(bvp_w))
            for i in range(L):
                eeg_x, y = eeg_w[i]
                bvp_x = bvp_w[i]
                self.samples.append((eeg_x, bvp_x, y))

        print(f"✅ Hybrid multimodal samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eeg_x, bvp_w, y = self.samples[idx]
        eeg_x = torch.tensor(eeg_x, dtype=torch.float32)
        bvp_w = torch.tensor(bvp_w, dtype=torch.float32)
        x1 = bvp_w.unsqueeze(0)
        x2 = torch.tensor(moving_average_backward(bvp_w.numpy(), 2), dtype=torch.float32).unsqueeze(0)
        x3 = bvp_w[::2].unsqueeze(0)
        feats = torch.tensor(extract_bvp_features(bvp_w.numpy()), dtype=torch.float32)
        return eeg_x, (x1, x2, x3, feats), torch.tensor(y, dtype=torch.long)


# ==================================================
# PART 4: TRAINING FUNCTIONS (simplified versions)
# ==================================================

def train_eeg_model(X_features, y_labels, split_indices, label_mapping):
    """Train EEG BiLSTM model on preprocessed data."""
    print("\n" + "="*80)
    print("TRAINING EEG MODEL (PREPROCESSED DATA)")
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
    
    # ...existing training code...
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
    
    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    
    tr_loader = DataLoader(tr_ds, batch_size=config.EEG_BATCH_SIZE, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    
    model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3,
        n_classes=config.NUM_CLASSES, p_drop=0.4
    ).to(config.DEVICE)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_f1, best_state, wait = 0.0, None, 0
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for xb, yb in tr_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
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
    print("EEG TEST RESULTS (PREPROCESSED)")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"Test Macro-F1: {test_f1:.3f}")
    id2lab = {v: k for k, v in label_mapping.items()}
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    return model, mu, sd


def train_bvp_model(X_raw, y_labels, split_indices, label_mapping):
    """Train BVP EMCNN model on raw data."""
    print("\n" + "="*80)
    print("TRAINING BVP MODEL (RAW DATA)")
    print("="*80)
    
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    train_ds = BVPDataset(X_raw[train_idx], y_labels[train_idx])
    val_ds = BVPDataset(X_raw[val_idx], y_labels[val_idx])
    test_ds = BVPDataset(X_raw[test_idx], y_labels[test_idx])
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_labels = y_labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=config.NUM_CLASSES).astype(np.float32)
    class_sample_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_sample_weights[train_labels]
    sample_weights_tensor = torch.from_numpy(sample_weights.astype(np.float32))
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.BVP_BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=config.BVP_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BVP_BATCH_SIZE, shuffle=False)
    
    model = EMCNN(n_classes=config.NUM_CLASSES, feat_dim=5).to(config.DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.BVP_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_acc = 0.0
    wait = 0
    
    for epoch in range(1, config.BVP_EPOCHS + 1):
        model.train()
        correct = total_samples = 0
        
        for x1, x2, x3, feats, y in train_loader:
            x1, x2, x3, feats, y = x1.to(config.DEVICE), x2.to(config.DEVICE), x3.to(config.DEVICE), feats.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            out = model(x1, x2, x3, feats)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total_samples += y.size(0)
        
        train_acc = correct / total_samples
        
        model.eval()
        correct = total_samples = 0
        with torch.no_grad():
            for x1, x2, x3, feats, y in val_loader:
                x1, x2, x3, feats, y = x1.to(config.DEVICE), x2.to(config.DEVICE), x3.to(config.DEVICE), feats.to(config.DEVICE), y.to(config.DEVICE)
                out = model(x1, x2, x3, feats)
                correct += (out.argmax(1) == y).sum().item()
                total_samples += y.size(0)
        
        val_acc = correct / total_samples
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), config.BVP_CHECKPOINT)
        else:
            wait += 1
            if wait >= config.BVP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    model.load_state_dict(torch.load(config.BVP_CHECKPOINT, map_location=config.DEVICE))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x1, x2, x3, feats, y in test_loader:
            x1, x2, x3, feats = x1.to(config.DEVICE), x2.to(config.DEVICE), x3.to(config.DEVICE), feats.to(config.DEVICE)
            out = model(x1, x2, x3, feats)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    test_acc = (all_preds == all_labels).mean()
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*80)
    print("BVP TEST RESULTS (RAW)")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"Test Macro-F1: {test_f1:.3f}")
    id2lab = {v: k for k, v in label_mapping.items()}
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    return model


def train_fusion_model(eeg_X_features, eeg_y, eeg_clip_names, bvp_X_raw, bvp_y, bvp_clip_names, 
                       eeg_split_indices, bvp_split_indices, eeg_encoder, bvp_encoder):
    """Train hybrid fusion model."""
    print("\n" + "="*80)
    print("TRAINING HYBRID FUSION MODEL")
    print("="*80)
    print("Combining: Preprocessed EEG + Raw BVP")
    
    eeg_train_idx = eeg_split_indices['train']
    eeg_val_idx = eeg_split_indices['val']
    eeg_test_idx = eeg_split_indices['test']
    
    bvp_train_idx = bvp_split_indices['train']
    bvp_val_idx = bvp_split_indices['val']
    bvp_test_idx = bvp_split_indices['test']
    
    train_dataset = MultimodalEEGBVPDataset(
        eeg_X=[eeg_X_features[i] for i in eeg_train_idx],
        eeg_y=[eeg_y[i] for i in eeg_train_idx],
        eeg_clip_ids=[eeg_clip_names[i] for i in eeg_train_idx],
        bvp_X=[bvp_X_raw[i] for i in bvp_train_idx],
        bvp_clip_ids=[bvp_clip_names[i] for i in bvp_train_idx]
    )
    
    val_dataset = MultimodalEEGBVPDataset(
        eeg_X=[eeg_X_features[i] for i in eeg_val_idx],
        eeg_y=[eeg_y[i] for i in eeg_val_idx],
        eeg_clip_ids=[eeg_clip_names[i] for i in eeg_val_idx],
        bvp_X=[bvp_X_raw[i] for i in bvp_val_idx],
        bvp_clip_ids=[bvp_clip_names[i] for i in bvp_val_idx]
    )
    
    test_dataset = MultimodalEEGBVPDataset(
        eeg_X=[eeg_X_features[i] for i in eeg_test_idx],
        eeg_y=[eeg_y[i] for i in eeg_test_idx],
        eeg_clip_ids=[eeg_clip_names[i] for i in eeg_test_idx],
        bvp_X=[bvp_X_raw[i] for i in bvp_test_idx],
        bvp_clip_ids=[bvp_clip_names[i] for i in bvp_test_idx]
    )
    
    train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    train_labels = np.array(train_labels)
    
    class_counts = np.bincount(train_labels, minlength=config.NUM_CLASSES).astype(np.float32)
    class_sample_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_sample_weights[train_labels]
    sample_weights_tensor = torch.from_numpy(sample_weights.astype(np.float32))
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.FUSION_BATCH_SIZE, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    fusion_model = WindowFusionEEGBVP(
        eeg_encoder=eeg_encoder,
        bvp_encoder=bvp_encoder,
        n_classes=config.NUM_CLASSES,
        shared_dim=config.FUSION_SHARED_DIM,
        num_heads=config.FUSION_NUM_HEADS,
        dropout=config.FUSION_DROPOUT
    ).to(config.DEVICE)
    
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"Fusion model trainable parameters: {trainable_params:,}")
    
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=config.FUSION_LR)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.FUSION_EPOCHS + 1):
        fusion_model.train()
        train_loss = 0.0

        for eeg_x, bvp_x, y in train_loader:
            eeg_x = eeg_x.to(config.DEVICE)
            bvp_x = tuple(v.to(config.DEVICE) for v in bvp_x)
            y = y.to(config.DEVICE)

            optimizer.zero_grad()
            out = fusion_model(eeg_x, bvp_x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        fusion_model.eval()
        correct = total = 0
        with torch.no_grad():
            for eeg_x, bvp_x, y in val_loader:
                eeg_x = eeg_x.to(config.DEVICE)
                bvp_x = tuple(v.to(config.DEVICE) for v in bvp_x)
                y = y.to(config.DEVICE)
                out = fusion_model(eeg_x, bvp_x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} | Val acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(fusion_model.state_dict(), config.FUSION_CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= config.FUSION_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    fusion_model.load_state_dict(torch.load(config.FUSION_CHECKPOINT, map_location=config.DEVICE))
    fusion_model.eval()

    correct = total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for eeg_x, bvp_x, y in test_loader:
            eeg_x = eeg_x.to(config.DEVICE)
            bvp_x = tuple(v.to(config.DEVICE) for v in bvp_x)
            y = y.to(config.DEVICE)
            out = fusion_model(eeg_x, bvp_x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_acc = correct / total
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*80)
    print("HYBRID FUSION TEST RESULTS")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"Test Macro-F1: {test_f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=config.IDX_TO_LABEL,
                                digits=3, zero_division=0))
    
    return fusion_model


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """Complete hybrid fusion pipeline."""
    print("=" * 80)
    print("HYBRID MULTIMODAL EMOTION RECOGNITION PIPELINE")
    print("=" * 80)
    print("Dataset Configuration:")
    print(f"  EEG: PREPROCESSED (KNN_ASR cleaned)")
    print(f"  BVP: RAW (original signals)")
    print("=" * 80)
    
    # Step 1: Load data from different sources
    eeg_X_raw, eeg_y, eeg_subjects, eeg_file_ids, eeg_label_map, eeg_clip_names = load_eeg_data_preprocessed(config.EEG_DATA_ROOT)
    eeg_X_features = extract_eeg_features(eeg_X_raw)
    
    bvp_X_raw, bvp_y, bvp_subjects, bvp_file_ids, bvp_label_map, bvp_clip_names = load_bvp_data_raw(config.BVP_DATA_ROOT)
    
    # Step 2: Create synchronized splits
    splits = create_clip_based_split(
        eeg_clip_names, eeg_y, eeg_subjects,
        bvp_clip_names, bvp_y, bvp_subjects
    )
    
    eeg_split_indices = {
        'train': get_clip_indices(eeg_clip_names, splits['train_clips']),
        'val': get_clip_indices(eeg_clip_names, splits['val_clips']),
        'test': get_clip_indices(eeg_clip_names, splits['test_clips'])
    }
    
    bvp_split_indices = {
        'train': get_clip_indices(bvp_clip_names, splits['train_clips']),
        'val': get_clip_indices(bvp_clip_names, splits['val_clips']),
        'test': get_clip_indices(bvp_clip_names, splits['test_clips'])
    }
    
    # Step 3: Train EEG model (preprocessed data)
    eeg_model, eeg_mu, eeg_sd = train_eeg_model(eeg_X_features, eeg_y, eeg_split_indices, eeg_label_map)
    
    # Step 4: Train BVP model (raw data)
    bvp_model = train_bvp_model(bvp_X_raw, bvp_y, bvp_split_indices, bvp_label_map)
    
    # Step 5: Prepare fusion encoders
    print("\n" + "="*80)
    print("PREPARING HYBRID FUSION ENCODERS")
    print("="*80)
    
    # Standardize EEG features
    eeg_X_features = (eeg_X_features - eeg_mu) / eeg_sd
    
    # EEG encoder (from preprocessed model)
    eeg_encoder = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3, n_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    eeg_encoder.load_state_dict(torch.load(config.EEG_CHECKPOINT, map_location=config.DEVICE))
    eeg_encoder.classifier = nn.Identity()
    for p in eeg_encoder.parameters():
        p.requires_grad = False
    eeg_encoder.eval()
    print("✅ Preprocessed EEG encoder prepared")
    
    # BVP encoder (from raw model)
    bvp_encoder = BVPEncoder(feat_dim=5).to(config.DEVICE)
    bvp_state = torch.load(config.BVP_CHECKPOINT, map_location=config.DEVICE)
    bvp_encoder.load_state_dict(bvp_state, strict=False)
    
    # Infer BVP output dimension
    with torch.no_grad():
        dummy = torch.zeros(1, config.BVP_WINDOW_SIZE)
        x1 = dummy.unsqueeze(1).to(config.DEVICE)
        x2 = dummy.unsqueeze(1).to(config.DEVICE)
        x3 = dummy[:, ::2].unsqueeze(1).to(config.DEVICE)
        feats = torch.zeros(1, 5).to(config.DEVICE)
        out = bvp_encoder(x1, x2, x3, feats)
        bvp_encoder.out_dim = out.shape[1]
    
    for p in bvp_encoder.parameters():
        p.requires_grad = False
    bvp_encoder.eval()
    print(f"✅ Raw BVP encoder prepared (out_dim={bvp_encoder.out_dim})")
    
    # Step 6: Train hybrid fusion model
    fusion_model = train_fusion_model(
        eeg_X_features, eeg_y, eeg_clip_names,
        bvp_X_raw, bvp_y, bvp_clip_names,
        eeg_split_indices, bvp_split_indices, eeg_encoder, bvp_encoder
    )
    
    print("\n" + "=" * 80)
    print("🎉 HYBRID FUSION PIPELINE COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ EEG (Preprocessed): {config.EEG_CHECKPOINT}")
    print(f"✅ BVP (Raw): {config.BVP_CHECKPOINT}")
    print(f"✅ Hybrid Fusion: {config.FUSION_CHECKPOINT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
