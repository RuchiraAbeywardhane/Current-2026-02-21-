"""
Emotion Detection from BVP Signal
This script identifies emotional states (arousal vs neutral) using BVP signals
and marks those time segments for EEG analysis.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class BVPEmotionDetector:
    """
    Detects emotional states from BVP (Blood Volume Pulse) signals.
    Based on heart rate variability and arousal detection.
    """
    
    def __init__(self, sampling_rate: int = 64):
        """
        Initialize the BVP emotion detector.
        
        Args:
            sampling_rate: Sampling rate of BVP signal in Hz
        """
        self.fs = sampling_rate
        self.emotional_segments = []
        self.neutral_segments = []
        
    def preprocess_bvp(self, bvp_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess BVP signal with bandpass filtering.
        
        Args:
            bvp_signal: Raw BVP signal
            
        Returns:
            Filtered BVP signal
        """
        # Bandpass filter for BVP (typical range: 0.5-8 Hz for heart rate)
        nyquist = self.fs / 2
        low = 0.5 / nyquist
        high = 8.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_bvp = signal.filtfilt(b, a, bvp_signal)
        
        return filtered_bvp
    
    def detect_peaks(self, bvp_signal: np.ndarray) -> np.ndarray:
        """
        Detect peaks (heartbeats) in BVP signal.
        
        Args:
            bvp_signal: Preprocessed BVP signal
            
        Returns:
            Array of peak indices
        """
        # Find peaks with minimum distance between peaks (minimum heart rate consideration)
        min_distance = int(0.4 * self.fs)  # Minimum 0.4s between beats (150 BPM max)
        
        peaks, _ = signal.find_peaks(bvp_signal, 
                                     distance=min_distance,
                                     prominence=np.std(bvp_signal) * 0.3)
        
        return peaks
    
    def calculate_hrv_features(self, ibi: np.ndarray) -> dict:
        """
        Calculate Heart Rate Variability features from Inter-Beat Intervals.
        
        Args:
            ibi: Inter-beat intervals in seconds
            
        Returns:
            Dictionary of HRV features
        """
        if len(ibi) < 2:
            return None
        
        # Time domain features
        mean_ibi = np.mean(ibi)
        std_ibi = np.std(ibi)
        rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))  # Root mean square of successive differences
        
        # Heart rate
        mean_hr = 60.0 / mean_ibi if mean_ibi > 0 else 0
        
        # pNN50: percentage of successive differences > 50ms
        diff_ibi = np.abs(np.diff(ibi) * 1000)  # Convert to ms
        pnn50 = np.sum(diff_ibi > 50) / len(diff_ibi) * 100 if len(diff_ibi) > 0 else 0
        
        return {
            'mean_hr': mean_hr,
            'std_ibi': std_ibi,
            'rmssd': rmssd,
            'pnn50': pnn50
        }
    
    def detect_emotional_states(self, 
                                bvp_signal: np.ndarray, 
                                window_size: int = 10,
                                threshold: float = 1.5) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Detect emotional vs neutral states using sliding window analysis.
        
        Args:
            bvp_signal: Raw BVP signal
            window_size: Window size in seconds for analysis
            threshold: Z-score threshold for emotional state detection
            
        Returns:
            emotion_labels: Array of labels (1=emotional, 0=neutral) for each sample
            features_df: DataFrame with extracted features per window
        """
        # Preprocess BVP
        filtered_bvp = self.preprocess_bvp(bvp_signal)
        
        # Detect peaks
        peaks = self.detect_peaks(filtered_bvp)
        
        if len(peaks) < 3:
            print("Warning: Not enough peaks detected in BVP signal")
            return np.zeros(len(bvp_signal)), pd.DataFrame()
        
        # Calculate IBI
        ibi = np.diff(peaks) / self.fs  # Inter-beat intervals in seconds
        
        # Sliding window analysis
        window_samples = window_size * self.fs
        hop_samples = window_samples // 2  # 50% overlap
        
        features_list = []
        emotion_labels = np.zeros(len(bvp_signal))
        
        for start in range(0, len(bvp_signal) - window_samples, hop_samples):
            end = start + window_samples
            
            # Find peaks in this window
            window_peaks = peaks[(peaks >= start) & (peaks < end)]
            
            if len(window_peaks) < 2:
                continue
            
            # Calculate IBI for this window
            window_ibi = np.diff(window_peaks) / self.fs
            
            # Extract HRV features
            hrv_features = self.calculate_hrv_features(window_ibi)
            
            if hrv_features is None:
                continue
            
            hrv_features['start_time'] = start / self.fs
            hrv_features['end_time'] = end / self.fs
            features_list.append(hrv_features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        if len(features_df) == 0:
            return emotion_labels, features_df
        
        # Detect emotional states based on arousal indicators
        # High arousal = higher HR, lower HRV (lower RMSSD, lower pNN50)
        hr_zscore = zscore(features_df['mean_hr'])
        rmssd_zscore = zscore(features_df['rmssd'])
        
        # Emotional state: high HR OR low RMSSD (indicating arousal)
        arousal_score = hr_zscore - rmssd_zscore
        features_df['arousal_score'] = arousal_score
        features_df['is_emotional'] = (arousal_score > threshold).astype(int)
        
        # Label the original signal
        for idx, row in features_df.iterrows():
            if row['is_emotional']:
                start_idx = int(row['start_time'] * self.fs)
                end_idx = int(row['end_time'] * self.fs)
                emotion_labels[start_idx:end_idx] = 1
        
        # Store segments
        self.emotional_segments = features_df[features_df['is_emotional'] == 1][['start_time', 'end_time']].values.tolist()
        self.neutral_segments = features_df[features_df['is_emotional'] == 0][['start_time', 'end_time']].values.tolist()
        
        return emotion_labels, features_df
    
    def get_emotional_segments(self) -> List[Tuple[float, float]]:
        """Return list of emotional segments as (start_time, end_time) tuples."""
        return self.emotional_segments
    
    def get_neutral_segments(self) -> List[Tuple[float, float]]:
        """Return list of neutral segments as (start_time, end_time) tuples."""
        return self.neutral_segments
    
    def visualize_results(self, 
                         bvp_signal: np.ndarray, 
                         emotion_labels: np.ndarray,
                         features_df: pd.DataFrame,
                         save_path: str = None):
        """
        Visualize BVP signal with emotional segments highlighted.
        
        Args:
            bvp_signal: Raw BVP signal
            emotion_labels: Emotion labels for each sample
            features_df: DataFrame with features
            save_path: Path to save the figure (optional)
        """
        time = np.arange(len(bvp_signal)) / self.fs
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: BVP Signal with emotional regions
        axes[0].plot(time, bvp_signal, 'b-', alpha=0.7, linewidth=0.5)
        axes[0].fill_between(time, bvp_signal.min(), bvp_signal.max(), 
                            where=emotion_labels > 0, alpha=0.3, color='red', 
                            label='Emotional State')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('BVP Amplitude')
        axes[0].set_title('BVP Signal with Detected Emotional States')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Heart Rate over time
        if len(features_df) > 0:
            window_times = (features_df['start_time'] + features_df['end_time']) / 2
            colors = ['red' if x == 1 else 'green' for x in features_df['is_emotional']]
            axes[1].scatter(window_times, features_df['mean_hr'], c=colors, alpha=0.6, s=50)
            axes[1].plot(window_times, features_df['mean_hr'], 'k--', alpha=0.3)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Heart Rate (BPM)')
            axes[1].set_title('Heart Rate Over Time (Red=Emotional, Green=Neutral)')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Arousal Score
        if len(features_df) > 0 and 'arousal_score' in features_df.columns:
            window_times = (features_df['start_time'] + features_df['end_time']) / 2
            axes[2].plot(window_times, features_df['arousal_score'], 'b-', marker='o')
            axes[2].axhline(y=1.5, color='r', linestyle='--', label='Threshold')
            axes[2].fill_between(window_times, axes[2].get_ylim()[0], axes[2].get_ylim()[1],
                                where=features_df['is_emotional'] > 0, alpha=0.3, color='red')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Arousal Score (Z-score)')
            axes[2].set_title('Arousal Score Over Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def export_segments(self, output_path: str):
        """
        Export emotional and neutral segments to CSV file.
        
        Args:
            output_path: Path to save the CSV file
        """
        all_segments = []
        
        for start, end in self.emotional_segments:
            all_segments.append({
                'start_time': start,
                'end_time': end,
                'duration': end - start,
                'state': 'emotional'
            })
        
        for start, end in self.neutral_segments:
            all_segments.append({
                'start_time': start,
                'end_time': end,
                'duration': end - start,
                'state': 'neutral'
            })
        
        df = pd.DataFrame(all_segments)
        df = df.sort_values('start_time').reset_index(drop=True)
        df.to_csv(output_path, index=False)
        print(f"Segments exported to: {output_path}")


def main():
    """
    Example usage of BVP Emotion Detector
    """
    # Example: Load your BVP signal
    # Replace this with your actual data loading
    
    # Simulated BVP signal (replace with your data)
    print("Loading BVP signal...")
    # bvp_signal = np.load('path_to_your_bvp_data.npy')
    
    # For demonstration, create a synthetic signal
    fs = 64  # sampling rate
    duration = 120  # seconds
    t = np.arange(0, duration, 1/fs)
    
    # Simulate BVP with varying heart rate (emotional changes)
    base_hr = 75  # BPM
    bvp_signal = np.sin(2 * np.pi * 1.2 * t)  # Base frequency
    
    # Add emotional arousal segments (increased HR)
    emotional_periods = [(20, 35), (60, 80)]
    for start, end in emotional_periods:
        mask = (t >= start) & (t <= end)
        bvp_signal[mask] += 0.5 * np.sin(2 * np.pi * 1.5 * t[mask])
    
    # Add noise
    bvp_signal += 0.1 * np.random.randn(len(bvp_signal))
    
    # Initialize detector
    print("Initializing BVP Emotion Detector...")
    detector = BVPEmotionDetector(sampling_rate=fs)
    
    # Detect emotional states
    print("Detecting emotional states...")
    emotion_labels, features_df = detector.detect_emotional_states(
        bvp_signal, 
        window_size=10,  # 10-second windows
        threshold=1.5    # Z-score threshold
    )
    
    # Print results
    print("\n" + "="*60)
    print("EMOTION DETECTION RESULTS")
    print("="*60)
    print(f"Total duration: {len(bvp_signal)/fs:.2f} seconds")
    print(f"Emotional segments detected: {len(detector.get_emotional_segments())}")
    print(f"Neutral segments detected: {len(detector.get_neutral_segments())}")
    
    print("\nEmotional Segments:")
    for i, (start, end) in enumerate(detector.get_emotional_segments(), 1):
        print(f"  {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    print("\nFeatures Summary:")
    if len(features_df) > 0:
        print(features_df.describe())
    
    # Visualize
    print("\nGenerating visualization...")
    detector.visualize_results(
        bvp_signal, 
        emotion_labels, 
        features_df,
        save_path='e:\\FInal Year Project\\MyCodeSpace\\Current(2026-02-21)\\emotion_detection_results.png'
    )
    
    # Export segments
    detector.export_segments('e:\\FInal Year Project\\MyCodeSpace\\Current(2026-02-21)\\emotional_segments.csv')
    
    print("\nDone! You can now use these emotional segments to analyze corresponding EEG data.")


if __name__ == "__main__":
    main()
