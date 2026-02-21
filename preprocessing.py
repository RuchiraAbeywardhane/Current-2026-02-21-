"""
Preprocessing Module
====================
Signal preprocessing utilities for EEG and BVP data.

Usage:
    from preprocessing import extract_eeg_features, preprocess_bvp
"""

import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, find_peaks, medfilt


# ==================== UTILITY FUNCTIONS ====================

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


# ==================== EEG PREPROCESSING ====================

def apply_baseline_reduction(signal, baseline, eps=1e-12):
    """
    Apply InvBase method: divide trial FFT by baseline FFT.
    
    Args:
        signal (np.ndarray): Trial signal (T, C)
        baseline (np.ndarray): Baseline signal (T, C)
        eps (float): Small constant to prevent division by zero
    
    Returns:
        np.ndarray: Baseline-reduced signal (T, C)
    """
    FFT_trial = np.fft.rfft(signal, axis=0)
    FFT_baseline = np.fft.rfft(baseline, axis=0)
    FFT_reduced = FFT_trial / (np.abs(FFT_baseline) + eps)
    signal_reduced = np.fft.irfft(FFT_reduced, n=len(signal), axis=0)
    return signal_reduced.astype(np.float32)


def extract_eeg_features(X_raw, fs=256.0, bands=None, eps=1e-12):
    """
    Extract 26 features per channel from EEG windows.
    
    Features:
    - Differential Entropy (5 bands)
    - Log Power Spectral Density (5 bands)
    - Temporal statistics (4: mean, std, skew, kurtosis)
    - DE asymmetry (5: left-right asymmetry per band)
    - Bandpower ratios (3: theta/alpha, beta/alpha, gamma/beta)
    - Hjorth parameters (2: mobility, complexity)
    - Time-domain extras (2: log variance, zero-crossing rate)
    
    Args:
        X_raw (np.ndarray): Raw EEG windows (N, T, C)
        fs (float): Sampling frequency
        bands (list): Frequency bands (name, (low, high))
        eps (float): Small constant for numerical stability
    
    Returns:
        np.ndarray: Features (N, C, 26)
    """
    if bands is None:
        bands = [
            ("delta", (1, 3)),
            ("theta", (4, 7)),
            ("alpha", (8, 13)),
            ("beta", (14, 30)),
            ("gamma", (31, 45))
        ]
    
    N, T, C = X_raw.shape
    
    # Compute power spectral density
    P = (np.abs(np.fft.rfft(X_raw, axis=1))**2) / T
    freqs = np.fft.rfftfreq(T, d=1/fs)
    
    feature_list = []
    
    # 1) Differential Entropy (5 bands)
    de_feats = []
    for _, (lo, hi) in bands:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * (bp + eps))
        de_feats.append(de[..., None])
    de_all = np.concatenate(de_feats, axis=2)  # (N, C, 5)
    feature_list.append(de_all)
    
    # 2) Log-PSD (5 bands)
    psd_feats = []
    for _, (lo, hi) in bands:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        log_psd = np.log(bp + eps)
        psd_feats.append(log_psd[..., None])
    psd_all = np.concatenate(psd_feats, axis=2)  # (N, C, 5)
    feature_list.append(psd_all)
    
    # 3) Temporal statistics (4)
    temp_mean = X_raw.mean(axis=1)[..., None]
    temp_std = X_raw.std(axis=1)[..., None]
    temp_skew = skew(X_raw, axis=1)[..., None]
    temp_kurt = kurtosis(X_raw, axis=1)[..., None]
    temp_all = np.concatenate([temp_mean, temp_std, temp_skew, temp_kurt], axis=2)  # (N, C, 4)
    feature_list.append(temp_all)
    
    # 4) DE asymmetry (5) - left/right hemisphere difference
    de_left = (de_all[:, 0, :] + de_all[:, 1, :]) / 2  # TP9, AF7
    de_right = (de_all[:, 2, :] + de_all[:, 3, :]) / 2  # AF8, TP10
    de_asym = de_left - de_right
    de_asym_full = np.tile(de_asym[:, None, :], (1, C, 1))  # (N, C, 5)
    feature_list.append(de_asym_full)
    
    # 5) Bandpower ratios (3)
    band_bp = []
    for _, (lo, hi) in bands:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        band_bp.append(bp)
    _, theta_bp, alpha_bp, beta_bp, gamma_bp = band_bp
    
    ratio_theta_alpha = (theta_bp + eps) / (alpha_bp + eps)
    ratio_beta_alpha = (beta_bp + eps) / (alpha_bp + eps)
    ratio_gamma_beta = (gamma_bp + eps) / (beta_bp + eps)
    ratio_all = np.stack([ratio_theta_alpha, ratio_beta_alpha, ratio_gamma_beta], axis=2)  # (N, C, 3)
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
    hjorth_all = np.stack([mobility, complexity], axis=2)  # (N, C, 2)
    feature_list.append(hjorth_all)
    
    # 7) Time-domain extras (2)
    log_var = np.log(var_x + eps)
    sign_x = np.sign(Xc)
    zc = (np.diff(sign_x, axis=1) != 0).sum(axis=1) / float(T - 1 + eps)
    td_extras = np.stack([log_var, zc], axis=2)  # (N, C, 2)
    feature_list.append(td_extras)
    
    # Concatenate all features
    features = np.concatenate(feature_list, axis=2)  # (N, C, 26)
    return features.astype(np.float32)


# ==================== BVP PREPROCESSING ====================

def butter_lowpass(cutoff_hz, fs, order=6):
    """Design Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def wavelet_denoise(sig, wavelet="db4", level=4, thresh_scale=1.0):
    """
    Wavelet denoising for BVP signals.
    
    Args:
        sig (np.ndarray): Input signal
        wavelet (str): Wavelet type
        level (int): Decomposition level
        thresh_scale (float): Threshold scaling factor
    
    Returns:
        np.ndarray: Denoised signal
    """
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
    """
    Remove baseline drift from BVP signal using peak detection.
    
    Args:
        sig (np.ndarray): Input BVP signal
        fs (float): Sampling frequency
        return_normalized (bool): If True, normalize to [0, 1]
    
    Returns:
        np.ndarray: Baseline-corrected signal
    """
    sig = np.asarray(sig).astype(float)
    
    min_dist_samples = max(1, int(fs * 60.0 / 200.0))
    sig_range = np.max(sig) - np.min(sig)
    if np.isclose(sig_range, 0):
        sig_range = 1.0
    
    prominence = 0.03 * sig_range
    inv = -sig
    minima_idx, _ = find_peaks(inv, distance=min_dist_samples, prominence=prominence)
    
    if len(minima_idx) < 2:
        # Fallback to median filter
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


def preprocess_bvp(bvp_raw, fs=64):
    """
    Complete BVP preprocessing pipeline.
    
    Steps:
    1. Butterworth lowpass filter (15 Hz)
    2. Wavelet denoising
    3. Baseline correction
    4. Z-score normalization
    
    Args:
        bvp_raw (np.ndarray): Raw BVP signal
        fs (float): Sampling frequency
    
    Returns:
        np.ndarray: Preprocessed BVP signal
    """
    # 1. Lowpass filter
    b, a = butter_lowpass(15.0, fs, order=6)
    bvp_filtered = filtfilt(b, a, bvp_raw)
    
    # 2. Wavelet denoising
    bvp_denoised = wavelet_denoise(bvp_filtered, wavelet="db4", level=4)
    
    # 3. Baseline correction
    bvp_normalized = baseline_correct(bvp_denoised, fs, return_normalized=True)
    
    # 4. Z-score normalization
    bvp_final = (bvp_normalized - bvp_normalized.mean()) / (bvp_normalized.std() + 1e-8)
    
    return bvp_final


def extract_bvp_features(window, fs=64):
    """
    Extract handcrafted features from BVP window.
    
    Features:
    1. Mean amplitude
    2. Standard deviation
    3. Differential std (std of first derivative)
    4. Heart rate proxy (peak count / duration)
    5. Peak-to-peak amplitude
    
    Args:
        window (np.ndarray): BVP window
        fs (float): Sampling frequency
    
    Returns:
        np.ndarray: Feature vector (5,)
    """
    mean_val = np.mean(window)
    std_val = np.std(window)
    diff_std = np.std(np.diff(window))
    
    # Heart rate proxy
    peaks, _ = find_peaks(window, distance=fs * 0.4)
    duration = len(window) / fs
    hr_proxy = len(peaks) / (duration + 1e-6)
    
    # Peak-to-peak amplitude
    p2p = np.max(window) - np.min(window)
    
    return np.array([mean_val, std_val, diff_std, hr_proxy, p2p], dtype=np.float32)


def moving_average_backward(x, s):
    """
    Backward cumulative moving average for BVP augmentation.
    
    Args:
        x (np.ndarray): Input signal
        s (int): Window size
    
    Returns:
        np.ndarray: Smoothed signal
    """
    if s <= 1:
        return x
    c = np.cumsum(x)
    y = np.empty_like(x)
    for i in range(s - 1):
        y[i] = c[i] / (i + 1)
    y[s-1:] = (c[s-1:] - np.concatenate(([0], c[:-s]))) / s
    return y
