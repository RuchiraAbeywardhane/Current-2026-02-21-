"""
BVP Handcrafted Features Module
================================

This module extracts traditional handcrafted features from BVP (Blood Volume Pulse)
signals for emotion recognition. These features complement learned features from
deep learning models.

Features extracted (23 total):
1. Statistical features (mean, std, skewness, kurtosis, min, max, range)
2. Time-domain HRV features (SDNN, RMSSD, pNN50, SDSD)
3. Frequency-domain HRV features (LF power, HF power, LF/HF ratio)
4. Pulse amplitude features (mean, std, variability)
5. Heart rate features (mean HR, HR variability)
6. Signal quality metrics

Based on cardiovascular correlates of emotion research.

Author: Final Year Project
Date: 2026-02-27
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import signal, stats
from scipy.interpolate import interp1d


class BVPHandcraftedFeatures(nn.Module):
    """
    PyTorch-compatible handcrafted feature extraction for BVP signals.
    
    This module extracts comprehensive time-domain, frequency-domain, and 
    heart rate variability (HRV) features from BVP signals, based on research
    on cardiovascular correlates of emotion.
    
    Features (23 total):
    - Statistical (7): mean, std, skewness, kurtosis, min, max, range
    - Time-domain HRV (4): SDNN, RMSSD, pNN50, SDSD
    - Frequency-domain HRV (3): LF power, HF power, LF/HF ratio
    - Pulse features (5): mean amplitude, std amplitude, amplitude CV, rise time, fall time
    - Heart rate (3): mean HR, std HR, HR range
    - Signal quality (1): number of valid peaks
    
    Args:
        sampling_rate (float): BVP sampling rate in Hz (default: 64.0)
        min_peak_height (float): Minimum peak height for detection (default: None, auto-detect)
        min_peak_distance (int): Minimum distance between peaks in samples (default: 20)
        
    Input Shape:
        - x: [batch_size, time_steps, 1] - BVP signal tensor
        
    Output Shape:
        - features: [batch_size, 23] - handcrafted features tensor
    """
    
    def __init__(self, sampling_rate=64.0, min_peak_height=None, min_peak_distance=20):
        super(BVPHandcraftedFeatures, self).__init__()
        
        self.sampling_rate = sampling_rate
        self.min_peak_height = min_peak_height
        self.min_peak_distance = min_peak_distance
        self.feature_dim = 23  # Updated: 23 features total
        
        # Frequency bands for HRV analysis (in Hz)
        self.lf_band = (0.04, 0.15)  # Low frequency (sympathetic + parasympathetic)
        self.hf_band = (0.15, 0.4)   # High frequency (parasympathetic)
        
    def forward(self, x):
        """
        Extract handcrafted features from BVP signals.
        
        Args:
            x (torch.Tensor): Input BVP signal [batch_size, time_steps, 1]
            
        Returns:
            torch.Tensor: Extracted features [batch_size, 23]
        """
        # Convert to numpy for processing
        x_np = x.detach().cpu().numpy()
        batch_size = x_np.shape[0]
        
        # Initialize feature matrix
        features = np.zeros((batch_size, self.feature_dim), dtype=np.float32)
        
        # Process each sample in the batch independently
        for i in range(batch_size):
            bvp_signal = x_np[i, :, 0]
            sample_features = self._extract_single_sample_features(bvp_signal)
            features[i, :] = sample_features
        
        # Convert back to PyTorch tensor
        features_tensor = torch.from_numpy(features).to(x.device)
        
        return features_tensor
    
    def _extract_single_sample_features(self, bvp_signal):
        """
        Extract comprehensive features from a single BVP signal.
        
        Args:
            bvp_signal (np.ndarray): 1D BVP signal [time_steps]
            
        Returns:
            np.ndarray: Feature vector [23]
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # ============================================================
        # 1. STATISTICAL FEATURES (indices 0-6)
        # ============================================================
        features[0] = np.mean(bvp_signal)
        features[1] = np.std(bvp_signal)
        features[2] = stats.skew(bvp_signal)
        features[3] = stats.kurtosis(bvp_signal)
        features[4] = np.min(bvp_signal)
        features[5] = np.max(bvp_signal)
        features[6] = features[5] - features[4]  # Range
        
        # ============================================================
        # 2. PEAK DETECTION
        # ============================================================
        peak_indices, peak_properties = signal.find_peaks(
            bvp_signal,
            height=self.min_peak_height,
            distance=self.min_peak_distance
        )
        
        num_peaks = len(peak_indices)
        features[22] = num_peaks  # Signal quality indicator (index 22)
        
        # ============================================================
        # 3. TIME-DOMAIN HRV FEATURES (indices 7-10)
        # ============================================================
        if num_peaks >= 2:
            # RR intervals in seconds
            rr_intervals = np.diff(peak_indices) / self.sampling_rate
            
            # SDNN: Standard deviation of NN intervals
            features[7] = np.std(rr_intervals)
            
            # RMSSD: Root Mean Square of Successive Differences
            if len(rr_intervals) >= 2:
                successive_diffs = np.diff(rr_intervals)
                features[8] = np.sqrt(np.mean(successive_diffs ** 2))
                
                # pNN50: Percentage of successive differences > 50ms
                nn50 = np.sum(np.abs(successive_diffs) > 0.05)
                features[9] = (nn50 / len(successive_diffs)) * 100.0
                
                # SDSD: Standard deviation of successive differences
                features[10] = np.std(successive_diffs)
            else:
                features[8] = 0.0
                features[9] = 0.0
                features[10] = 0.0
        else:
            features[7:11] = 0.0
        
        # ============================================================
        # 4. FREQUENCY-DOMAIN HRV FEATURES (indices 11-13)
        # ============================================================
        if num_peaks >= 3:
            try:
                rr_intervals = np.diff(peak_indices) / self.sampling_rate
                
                # Create interpolated RR time series at 4 Hz
                rr_times = np.cumsum(rr_intervals)
                rr_times = np.insert(rr_times, 0, 0)
                rr_intervals_full = np.append(rr_intervals[0], rr_intervals)
                
                # Interpolate to uniform time grid
                fs_interp = 4.0  # 4 Hz interpolation
                t_interp = np.arange(0, rr_times[-1], 1/fs_interp)
                
                if len(t_interp) > 10 and len(rr_times) > 2:
                    f_interp = interp1d(rr_times, rr_intervals_full, kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')
                    rr_interp = f_interp(t_interp)
                    
                    # Compute power spectral density using Welch's method
                    freqs, psd = signal.welch(rr_interp, fs=fs_interp, 
                                             nperseg=min(256, len(rr_interp)))
                    
                    # LF power (0.04-0.15 Hz)
                    lf_mask = (freqs >= self.lf_band[0]) & (freqs < self.lf_band[1])
                    features[11] = np.trapz(psd[lf_mask], freqs[lf_mask])
                    
                    # HF power (0.15-0.4 Hz)
                    hf_mask = (freqs >= self.hf_band[0]) & (freqs < self.hf_band[1])
                    features[12] = np.trapz(psd[hf_mask], freqs[hf_mask])
                    
                    # LF/HF ratio
                    if features[12] > 0:
                        features[13] = features[11] / features[12]
                    else:
                        features[13] = 0.0
            except:
                features[11:14] = 0.0
        else:
            features[11:14] = 0.0
        
        # ============================================================
        # 5. PULSE AMPLITUDE FEATURES (indices 14-18)
        # ============================================================
        if num_peaks >= 1:
            peak_amplitudes = bvp_signal[peak_indices]
            
            # Mean peak amplitude
            features[14] = np.mean(peak_amplitudes)
            
            # Std peak amplitude
            features[15] = np.std(peak_amplitudes)
            
            # Coefficient of variation (CV) of peak amplitude
            if features[14] != 0:
                features[16] = (features[15] / features[14]) * 100.0
            else:
                features[16] = 0.0
            
            # Pulse wave characteristics: rise and fall times
            if num_peaks >= 2:
                rise_times = []
                fall_times = []
                
                for j in range(min(num_peaks-1, 5)):  # Analyze first 5 peaks
                    peak_idx = peak_indices[j]
                    
                    # Find trough before peak (rise)
                    search_start = max(0, peak_idx - self.min_peak_distance)
                    trough_before = np.argmin(bvp_signal[search_start:peak_idx]) + search_start
                    rise_time = (peak_idx - trough_before) / self.sampling_rate
                    rise_times.append(rise_time)
                    
                    # Find trough after peak (fall)
                    search_end = min(len(bvp_signal), peak_idx + self.min_peak_distance)
                    trough_after = np.argmin(bvp_signal[peak_idx:search_end]) + peak_idx
                    fall_time = (trough_after - peak_idx) / self.sampling_rate
                    fall_times.append(fall_time)
                
                features[17] = np.mean(rise_times) if rise_times else 0.0
                features[18] = np.mean(fall_times) if fall_times else 0.0
            else:
                features[17] = 0.0
                features[18] = 0.0
        else:
            features[14:19] = 0.0
        
        # ============================================================
        # 6. HEART RATE FEATURES (indices 19-21)
        # ============================================================
        if num_peaks >= 2:
            rr_intervals = np.diff(peak_indices) / self.sampling_rate
            
            # Mean heart rate (BPM)
            mean_rr = np.mean(rr_intervals)
            if mean_rr > 0:
                features[19] = 60.0 / mean_rr
            else:
                features[19] = 0.0
            
            # Std of heart rate
            hr_values = 60.0 / rr_intervals
            features[20] = np.std(hr_values)
            
            # Heart rate range
            features[21] = np.max(hr_values) - np.min(hr_values)
        else:
            features[19:22] = 0.0
        
        # Handle NaN or Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def get_feature_names(self):
        """
        Get names of all extracted features.
        
        Returns:
            list: List of feature names [23]
        """
        return [
            # Statistical (0-6)
            'mean', 'std', 'skewness', 'kurtosis', 'min', 'max', 'range',
            # Time-domain HRV (7-10)
            'sdnn', 'rmssd', 'pnn50', 'sdsd',
            # Frequency-domain HRV (11-13)
            'lf_power', 'hf_power', 'lf_hf_ratio',
            # Pulse features (14-18)
            'mean_peak_amp', 'std_peak_amp', 'peak_amp_cv', 'rise_time', 'fall_time',
            # Heart rate (19-21)
            'mean_hr', 'std_hr', 'hr_range',
            # Signal quality (22)
            'num_peaks'
        ]
    
    def get_output_dim(self):
        """
        Get the output feature dimension.
        
        Returns:
            int: Number of features (23)
        """
        return self.feature_dim


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("BVP HANDCRAFTED FEATURES TEST")
    print("=" * 80)
    
    # Create feature extractor
    feature_extractor = BVPHandcraftedFeatures(
        sampling_rate=64.0,
        min_peak_distance=20
    )
    
    print(f"\nFeature extractor initialized:")
    print(f"  Sampling rate: {feature_extractor.sampling_rate} Hz")
    print(f"  Feature dimension: {feature_extractor.get_output_dim()}")
    print(f"\nFeature names:")
    for i, name in enumerate(feature_extractor.get_feature_names()):
        print(f"  {i:2d}. {name}")
    
    # Test with synthetic BVP data
    print("\n" + "-" * 80)
    print("Test 1: Synthetic BVP signal with clear peaks")
    print("-" * 80)
    
    batch_size = 4
    time_steps = 256
    
    # Create synthetic BVP signal with peaks
    t = np.linspace(0, 4, time_steps)  # 4 seconds
    bvp_signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz signal (~60 BPM)
    bvp_signal = np.tile(bvp_signal, (batch_size, 1))[:, :, np.newaxis]
    bvp_tensor = torch.from_numpy(bvp_signal.astype(np.float32))
    
    print(f"Input shape: {bvp_tensor.shape}")
    
    # Extract features
    features = feature_extractor(bvp_tensor)
    print(f"Output shape: {features.shape}")
    print(f"Expected: [{batch_size}, {feature_extractor.get_output_dim()}]")
    
    # Display features for first sample
    print("\nFeatures for first sample:")
    feature_names = feature_extractor.get_feature_names()
    for i, (name, value) in enumerate(zip(feature_names, features[0].numpy())):
        print(f"  {name:15s}: {value:10.4f}")
    
    # Test with random noise
    print("\n" + "-" * 80)
    print("Test 2: Random noise (edge case)")
    print("-" * 80)
    
    noise_signal = torch.randn(batch_size, time_steps, 1) * 0.1
    features_noise = feature_extractor(noise_signal)
    
    print(f"Input shape: {noise_signal.shape}")
    print(f"Output shape: {features_noise.shape}")
    print("\nFeatures for first sample:")
    for i, (name, value) in enumerate(zip(feature_names, features_noise[0].numpy())):
        print(f"  {name:15s}: {value:10.4f}")
    
    # Test with real-world-like BVP signal
    print("\n" + "-" * 80)
    print("Test 3: Realistic BVP signal")
    print("-" * 80)
    
    # Create more realistic BVP with multiple frequency components
    t = np.linspace(0, 4, time_steps)
    heart_rate = 1.2  # 72 BPM
    bvp_realistic = (
        np.sin(2 * np.pi * heart_rate * t) +  # Main pulse
        0.3 * np.sin(2 * np.pi * heart_rate * 2 * t) +  # Harmonic
        0.1 * np.random.randn(time_steps)  # Noise
    )
    bvp_realistic = np.tile(bvp_realistic, (batch_size, 1))[:, :, np.newaxis]
    bvp_realistic_tensor = torch.from_numpy(bvp_realistic.astype(np.float32))
    
    features_realistic = feature_extractor(bvp_realistic_tensor)
    
    print(f"Input shape: {bvp_realistic_tensor.shape}")
    print(f"Output shape: {features_realistic.shape}")
    print("\nFeatures for first sample:")
    for i, (name, value) in enumerate(zip(feature_names, features_realistic[0].numpy())):
        print(f"  {name:15s}: {value:10.4f}")
    
    # Test gradient flow (to ensure PyTorch compatibility)
    print("\n" + "-" * 80)
    print("Test 4: PyTorch compatibility check")
    print("-" * 80)
    
    # Note: Handcrafted features are not differentiable by design
    # They are computed in numpy, but can be used in PyTorch pipelines
    print("✅ Features are extracted in numpy (not differentiable)")
    print("✅ Output is converted to PyTorch tensor")
    print("✅ Can be used in hybrid models with learned features")
    
    # Count parameters (should be 0 - no learnable parameters)
    total_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"\nLearnable parameters: {total_params}")
    print("(Expected: 0 - this is a handcrafted feature extractor)")
    
    print("\n" + "=" * 80)
    print("✅ BVP HANDCRAFTED FEATURES TEST COMPLETE!")
    print("=" * 80)
