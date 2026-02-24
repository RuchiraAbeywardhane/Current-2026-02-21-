"""
BVP Handcrafted Features Module
================================

This module extracts traditional handcrafted features from BVP (Blood Volume Pulse)
signals for emotion recognition. These features complement learned features from
deep learning models.

Features extracted:
1. Statistical features (mean, std, skewness, kurtosis)
2. Heart rate related features (RR intervals, RMSSD)
3. Pulse amplitude features (peak amplitudes)

Author: Final Year Project
Date: 2026-02-24
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import signal, stats


class BVPHandcraftedFeatures(nn.Module):
    """
    PyTorch-compatible handcrafted feature extraction for BVP signals.
    
    This module extracts traditional time-domain and heart rate variability (HRV)
    features from BVP signals. Features are computed in numpy for numerical
    stability and returned as PyTorch tensors.
    
    Features (11 total):
    - Statistical: mean, std, skewness, kurtosis (4 features)
    - RR intervals: mean_rr, std_rr, rmssd (3 features)
    - Pulse amplitude: mean_peak_amp, std_peak_amp (2 features)
    - Additional: num_peaks, heart_rate (2 features)
    
    Args:
        sampling_rate (float): BVP sampling rate in Hz (default: 64.0)
        min_peak_height (float): Minimum peak height for detection (default: None, auto-detect)
        min_peak_distance (int): Minimum distance between peaks in samples (default: 20)
        
    Input Shape:
        - x: [batch_size, time_steps, 1] - BVP signal tensor
        
    Output Shape:
        - features: [batch_size, 11] - handcrafted features tensor
    """
    
    def __init__(self, sampling_rate=64.0, min_peak_height=None, min_peak_distance=20):
        super(BVPHandcraftedFeatures, self).__init__()
        
        self.sampling_rate = sampling_rate
        self.min_peak_height = min_peak_height
        self.min_peak_distance = min_peak_distance
        self.feature_dim = 11  # Total number of features
        
    def forward(self, x):
        """
        Extract handcrafted features from BVP signals.
        
        Args:
            x (torch.Tensor): Input BVP signal [batch_size, time_steps, 1]
            
        Returns:
            torch.Tensor: Extracted features [batch_size, 11]
        """
        # Convert to numpy for processing
        # Detach from computation graph to avoid gradient issues
        x_np = x.detach().cpu().numpy()  # [batch_size, time_steps, 1]
        batch_size = x_np.shape[0]
        
        # Initialize feature matrix
        features = np.zeros((batch_size, self.feature_dim), dtype=np.float32)
        
        # Process each sample in the batch independently
        for i in range(batch_size):
            # Extract 1D signal from [time_steps, 1]
            bvp_signal = x_np[i, :, 0]  # [time_steps]
            
            # Extract features for this sample
            sample_features = self._extract_single_sample_features(bvp_signal)
            features[i, :] = sample_features
        
        # Convert back to PyTorch tensor and move to same device as input
        features_tensor = torch.from_numpy(features).to(x.device)
        
        return features_tensor
    
    def _extract_single_sample_features(self, bvp_signal):
        """
        Extract features from a single BVP signal.
        
        Args:
            bvp_signal (np.ndarray): 1D BVP signal [time_steps]
            
        Returns:
            np.ndarray: Feature vector [11]
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # ============================================================
        # 1. STATISTICAL FEATURES (indices 0-3)
        # ============================================================
        
        # Mean of the signal
        features[0] = np.mean(bvp_signal)
        
        # Standard deviation of the signal
        features[1] = np.std(bvp_signal)
        
        # Skewness (measure of asymmetry)
        # Higher values indicate more positive skew
        features[2] = stats.skew(bvp_signal)
        
        # Kurtosis (measure of tail heaviness)
        # Higher values indicate heavier tails
        features[3] = stats.kurtosis(bvp_signal)
        
        # ============================================================
        # 2. PEAK DETECTION
        # ============================================================
        
        # Detect peaks in the BVP signal
        # Peaks correspond to systolic points (heart beats)
        peak_indices, peak_properties = signal.find_peaks(
            bvp_signal,
            height=self.min_peak_height,  # Minimum peak height
            distance=self.min_peak_distance  # Minimum distance between peaks
        )
        
        num_peaks = len(peak_indices)
        features[9] = num_peaks  # Store number of peaks (index 9)
        
        # ============================================================
        # 3. HEART RATE RELATED FEATURES (indices 4-6, 10)
        # ============================================================
        
        # Only compute if we have at least 2 peaks
        if num_peaks >= 2:
            # RR intervals: time between consecutive peaks (in seconds)
            rr_intervals = np.diff(peak_indices) / self.sampling_rate
            
            # Mean RR interval
            features[4] = np.mean(rr_intervals)
            
            # Standard deviation of RR intervals (SDNN)
            # Measure of overall HRV
            features[5] = np.std(rr_intervals)
            
            # RMSSD: Root Mean Square of Successive Differences
            # Measure of short-term HRV
            if len(rr_intervals) >= 2:
                successive_diffs = np.diff(rr_intervals)
                features[6] = np.sqrt(np.mean(successive_diffs ** 2))
            else:
                features[6] = 0.0
            
            # Heart rate in BPM (beats per minute)
            # HR = 60 / mean_RR
            if features[4] > 0:
                features[10] = 60.0 / features[4]
            else:
                features[10] = 0.0
                
        else:
            # Not enough peaks: set HR features to 0
            features[4] = 0.0  # mean_rr
            features[5] = 0.0  # std_rr
            features[6] = 0.0  # rmssd
            features[10] = 0.0  # heart_rate
        
        # ============================================================
        # 4. PULSE AMPLITUDE FEATURES (indices 7-8)
        # ============================================================
        
        # Only compute if we have at least 1 peak
        if num_peaks >= 1:
            # Get peak amplitudes from the signal
            peak_amplitudes = bvp_signal[peak_indices]
            
            # Mean peak amplitude
            features[7] = np.mean(peak_amplitudes)
            
            # Standard deviation of peak amplitudes
            features[8] = np.std(peak_amplitudes)
        else:
            # No peaks detected: set amplitude features to 0
            features[7] = 0.0  # mean_peak_amp
            features[8] = 0.0  # std_peak_amp
        
        # Handle NaN or Inf values (replace with 0)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def get_feature_names(self):
        """
        Get names of all extracted features.
        
        Returns:
            list: List of feature names
        """
        return [
            'mean',           # 0: Mean of signal
            'std',            # 1: Standard deviation
            'skewness',       # 2: Skewness
            'kurtosis',       # 3: Kurtosis
            'mean_rr',        # 4: Mean RR interval
            'std_rr',         # 5: Std RR interval (SDNN)
            'rmssd',          # 6: RMSSD
            'mean_peak_amp',  # 7: Mean peak amplitude
            'std_peak_amp',   # 8: Std peak amplitude
            'num_peaks',      # 9: Number of peaks detected
            'heart_rate'      # 10: Heart rate (BPM)
        ]
    
    def get_output_dim(self):
        """
        Get the output feature dimension.
        
        Returns:
            int: Number of features (11)
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
