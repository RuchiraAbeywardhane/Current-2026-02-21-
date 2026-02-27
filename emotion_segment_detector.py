"""
Emotion Segment Detection using BVP Signal Analysis
This script identifies segments where a subject is experiencing emotions vs. neutral states
using Blood Volume Pulse (BVP) signals.

Based on the principle that emotional arousal causes changes in:
- Heart Rate (HR)
- Heart Rate Variability (HRV)
- Blood volume pulse amplitude
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class EmotionSegmentDetector:
    """
    Detects emotional segments in physiological recordings using BVP signal analysis.
    """
    
    def __init__(self, sampling_rate: int = 64):
        """
        Initialize the emotion segment detector.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of the BVP signal in Hz
        """
        self.fs = sampling_rate
        self.emotional_segments = []
        self.neutral_segments = []
        
    def preprocess_bvp(self, bvp_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess BVP signal: bandpass filter and normalize.
        
        Parameters:
        -----------
        bvp_signal : np.ndarray
            Raw BVP signal
            
        Returns:
        --------
        np.ndarray
            Preprocessed BVP signal
        """
        # Bandpass filter for BVP (typical range: 0.5-8 Hz for heart rate)
        nyquist = self.fs / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = 8.0 / nyquist
        
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_bvp = signal.filtfilt(b, a, bvp_signal)
        
        return filtered_bvp
    
    def detect_peaks(self, bvp_signal: np.ndarray) -> np.ndarray:
        """
        Detect peaks in BVP signal (heartbeats).
        
        Parameters:
        -----------
        bvp_signal : np.ndarray
            Preprocessed BVP signal
            
        Returns:
        --------
        np.ndarray
            Indices of detected peaks
        """
        # Find peaks with minimum distance based on max heart rate (200 bpm = 0.3s)
        min_distance = int(0.3 * self.fs)
        
        peaks, _ = signal.find_peaks(bvp_signal, 
                                     distance=min_distance,
                                     prominence=np.std(bvp_signal) * 0.5)
        return peaks
    
    def calculate_heart_rate(self, peaks: np.ndarray, window_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate instantaneous heart rate from peak positions.
        
        Parameters:
        -----------
        peaks : np.ndarray
            Indices of detected peaks
        window_size : int
            Window size for moving average (in samples)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Time points and heart rate values (in BPM)
        """
        if window_size is None:
            window_size = self.fs * 5  # 5 second window
        
        # Calculate inter-beat intervals (IBI)
        ibi = np.diff(peaks) / self.fs  # in seconds
        
        # Convert to heart rate (BPM)
        hr = 60.0 / ibi
        
        # Time points (use midpoint between peaks)
        time_points = (peaks[:-1] + peaks[1:]) / 2 / self.fs
        
        return time_points, hr
    
    def calculate_hrv_features(self, peaks: np.ndarray, window_size: int = 30) -> pd.DataFrame:
        """
        Calculate Heart Rate Variability (HRV) features in sliding windows.
        
        Parameters:
        -----------
        peaks : np.ndarray
            Indices of detected peaks
        window_size : int
            Window size in seconds
            
        Returns:
        --------
        pd.DataFrame
            HRV features for each window
        """
        window_samples = window_size * self.fs
        signal_length = peaks[-1]
        
        features = []
        
        # Slide window through the signal
        for start in range(0, int(signal_length), window_samples // 2):  # 50% overlap
            end = start + window_samples
            
            # Get peaks in this window
            window_peaks = peaks[(peaks >= start) & (peaks < end)]
            
            if len(window_peaks) < 5:  # Need at least 5 beats
                continue
            
            # Calculate IBI
            ibi = np.diff(window_peaks) / self.fs * 1000  # in milliseconds
            
            # Time-domain HRV features
            mean_hr = 60000 / np.mean(ibi) if len(ibi) > 0 else 0
            sdnn = np.std(ibi)  # Standard deviation of IBI
            rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))  # Root mean square of successive differences
            
            # Additional features
            nn50 = np.sum(np.abs(np.diff(ibi)) > 50)  # Number of successive IBIs differing by > 50ms
            pnn50 = (nn50 / len(ibi)) * 100 if len(ibi) > 0 else 0
            
            features.append({
                'window_start': start / self.fs,
                'window_end': end / self.fs,
                'mean_hr': mean_hr,
                'sdnn': sdnn,
                'rmssd': rmssd,
                'pnn50': pnn50,
                'mean_ibi': np.mean(ibi)
            })
        
        return pd.DataFrame(features)
    
    def detect_emotional_states(self, hrv_features: pd.DataFrame, 
                                threshold_method: str = 'zscore',
                                threshold_value: float = 0.5) -> pd.DataFrame:
        """
        Detect emotional vs. neutral states based on HRV features.
        
        Emotional arousal typically shows:
        - Increased heart rate
        - Decreased HRV (lower SDNN, RMSSD)
        - Higher variability in these metrics
        
        Parameters:
        -----------
        hrv_features : pd.DataFrame
            HRV features from calculate_hrv_features
        threshold_method : str
            'zscore' or 'percentile'
        threshold_value : float
            Threshold for classification (z-score or percentile)
            
        Returns:
        --------
        pd.DataFrame
            Features with emotion labels
        """
        df = hrv_features.copy()
        
        # Normalize features
        df['mean_hr_z'] = zscore(df['mean_hr'])
        df['sdnn_z'] = zscore(df['sdnn'])
        df['rmssd_z'] = zscore(df['rmssd'])
        
        # Emotion score: high HR + low HRV indicates emotional arousal
        df['emotion_score'] = df['mean_hr_z'] - (df['sdnn_z'] + df['rmssd_z']) / 2
        
        if threshold_method == 'zscore':
            # Classify as emotional if emotion_score > threshold
            df['is_emotional'] = df['emotion_score'] > threshold_value
        elif threshold_method == 'percentile':
            # Use percentile threshold
            threshold = np.percentile(df['emotion_score'], threshold_value * 100)
            df['is_emotional'] = df['emotion_score'] > threshold
        
        # Label states
        df['state'] = df['is_emotional'].map({True: 'Emotional', False: 'Neutral'})
        
        return df
    
    def merge_adjacent_segments(self, segments: List[Tuple[float, float]], 
                               max_gap: float = 5.0) -> List[Tuple[float, float]]:
        """
        Merge adjacent segments with small gaps between them.
        
        Parameters:
        -----------
        segments : List[Tuple[float, float]]
            List of (start_time, end_time) tuples
        max_gap : float
            Maximum gap to merge (in seconds)
            
        Returns:
        --------
        List[Tuple[float, float]]
            Merged segments
        """
        if not segments:
            return []
        
        # Sort by start time
        segments = sorted(segments)
        merged = [segments[0]]
        
        for current in segments[1:]:
            previous = merged[-1]
            
            # If current segment is close to previous, merge them
            if current[0] - previous[1] <= max_gap:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def identify_segments(self, bvp_signal: np.ndarray, 
                         threshold_method: str = 'zscore',
                         threshold_value: float = 0.5,
                         min_segment_duration: float = 5.0,
                         merge_gap: float = 5.0) -> Dict:
        """
        Complete pipeline to identify emotional and neutral segments.
        
        Parameters:
        -----------
        bvp_signal : np.ndarray
            Raw BVP signal
        threshold_method : str
            'zscore' or 'percentile'
        threshold_value : float
            Threshold for classification
        min_segment_duration : float
            Minimum duration for a segment (in seconds)
        merge_gap : float
            Maximum gap to merge adjacent segments (in seconds)
            
        Returns:
        --------
        Dict
            Dictionary containing emotional and neutral segments with their features
        """
        # Preprocess BVP
        print("Preprocessing BVP signal...")
        filtered_bvp = self.preprocess_bvp(bvp_signal)
        
        # Detect peaks
        print("Detecting heartbeats...")
        peaks = self.detect_peaks(filtered_bvp)
        print(f"Detected {len(peaks)} heartbeats")
        
        # Calculate HRV features
        print("Calculating HRV features...")
        hrv_features = self.calculate_hrv_features(peaks)
        
        if len(hrv_features) == 0:
            print("Warning: No HRV features could be calculated!")
            return {'emotional_segments': [], 'neutral_segments': [], 'features': pd.DataFrame()}
        
        # Detect emotional states
        print("Detecting emotional states...")
        classified_features = self.detect_emotional_states(hrv_features, 
                                                          threshold_method, 
                                                          threshold_value)
        
        # Extract segments
        emotional_segs = []
        neutral_segs = []
        
        for _, row in classified_features.iterrows():
            segment = (row['window_start'], row['window_end'])
            duration = segment[1] - segment[0]
            
            if duration >= min_segment_duration:
                if row['is_emotional']:
                    emotional_segs.append(segment)
                else:
                    neutral_segs.append(segment)
        
        # Merge adjacent segments
        emotional_segs = self.merge_adjacent_segments(emotional_segs, merge_gap)
        neutral_segs = self.merge_adjacent_segments(neutral_segs, merge_gap)
        
        self.emotional_segments = emotional_segs
        self.neutral_segments = neutral_segs
        
        print(f"\nFound {len(emotional_segs)} emotional segments")
        print(f"Found {len(neutral_segs)} neutral segments")
        
        return {
            'emotional_segments': emotional_segs,
            'neutral_segments': neutral_segs,
            'features': classified_features,
            'peaks': peaks,
            'filtered_bvp': filtered_bvp
        }
    
    def visualize_results(self, bvp_signal: np.ndarray, results: Dict, 
                         save_path: str = None):
        """
        Visualize the emotion detection results.
        
        Parameters:
        -----------
        bvp_signal : np.ndarray
            Original BVP signal
        results : Dict
            Results from identify_segments
        save_path : str
            Path to save the figure (optional)
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        time_axis = np.arange(len(bvp_signal)) / self.fs
        
        # Plot 1: BVP signal with detected peaks
        axes[0].plot(time_axis, bvp_signal, 'b-', alpha=0.5, label='Raw BVP')
        axes[0].plot(time_axis, results['filtered_bvp'], 'k-', label='Filtered BVP')
        
        peak_times = results['peaks'] / self.fs
        peak_values = results['filtered_bvp'][results['peaks']]
        axes[0].plot(peak_times, peak_values, 'ro', markersize=4, label='Detected peaks')
        
        axes[0].set_ylabel('BVP Signal')
        axes[0].set_title('BVP Signal with Detected Heartbeats')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Heart Rate
        hr_time, hr_values = self.calculate_heart_rate(results['peaks'])
        axes[1].plot(hr_time, hr_values, 'b-', linewidth=1.5)
        axes[1].set_ylabel('Heart Rate (BPM)')
        axes[1].set_title('Instantaneous Heart Rate')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: HRV features
        features = results['features']
        window_centers = (features['window_start'] + features['window_end']) / 2
        
        ax3_twin = axes[2].twinx()
        axes[2].plot(window_centers, features['mean_hr'], 'b-', label='Mean HR', linewidth=2)
        ax3_twin.plot(window_centers, features['sdnn'], 'r-', label='SDNN', linewidth=2)
        
        axes[2].set_ylabel('Mean HR (BPM)', color='b')
        ax3_twin.set_ylabel('SDNN (ms)', color='r')
        axes[2].set_title('HRV Features Over Time')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # Plot 4: Emotion score and segments
        axes[3].plot(window_centers, features['emotion_score'], 'k-', linewidth=2, label='Emotion Score')
        axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight emotional segments
        for start, end in results['emotional_segments']:
            axes[3].axvspan(start, end, alpha=0.3, color='red', label='Emotional' if start == results['emotional_segments'][0][0] else '')
        
        # Highlight neutral segments
        for start, end in results['neutral_segments']:
            axes[3].axvspan(start, end, alpha=0.3, color='green', label='Neutral' if start == results['neutral_segments'][0][0] else '')
        
        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_ylabel('Emotion Score')
        axes[3].set_title('Emotional State Classification')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def export_segments(self, results: Dict, output_path: str):
        """
        Export detected segments to CSV file.
        
        Parameters:
        -----------
        results : Dict
            Results from identify_segments
        output_path : str
            Path to save the CSV file
        """
        segments_data = []
        
        for start, end in results['emotional_segments']:
            segments_data.append({
                'start_time': start,
                'end_time': end,
                'duration': end - start,
                'state': 'Emotional'
            })
        
        for start, end in results['neutral_segments']:
            segments_data.append({
                'start_time': start,
                'end_time': end,
                'duration': end - start,
                'state': 'Neutral'
            })
        
        df = pd.DataFrame(segments_data)
        df = df.sort_values('start_time').reset_index(drop=True)
        df.to_csv(output_path, index=False)
        print(f"Segments exported to {output_path}")
        
        return df


def main():
    """
    Example usage of the EmotionSegmentDetector
    """
    # Example: Load your BVP data
    # Replace this with your actual data loading
    print("=== Emotion Segment Detector ===\n")
    
    # Generate synthetic BVP data for demonstration
    # In practice, load your actual BVP signal here
    sampling_rate = 64  # Hz
    duration = 300  # seconds (5 minutes)
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Simulate BVP with varying heart rate
    base_hr = 70  # BPM
    bvp_signal = np.zeros_like(t)
    
    for i, time in enumerate(t):
        # Create emotional arousal periods
        if 50 < time < 100 or 200 < time < 250:
            hr = base_hr + 20 + 10 * np.sin(2 * np.pi * 0.1 * time)
        else:
            hr = base_hr + 5 * np.sin(2 * np.pi * 0.05 * time)
        
        bvp_signal[i] = np.sin(2 * np.pi * (hr / 60) * time)
    
    # Add noise
    bvp_signal += 0.1 * np.random.randn(len(bvp_signal))
    
    print("Synthetic BVP signal generated for demonstration.")
    print("Replace this with your actual BVP data loading.\n")
    
    # Initialize detector
    detector = EmotionSegmentDetector(sampling_rate=sampling_rate)
    
    # Identify emotional segments
    results = detector.identify_segments(
        bvp_signal,
        threshold_method='zscore',
        threshold_value=0.5,
        min_segment_duration=10.0,
        merge_gap=5.0
    )
    
    # Print results
    print("\n=== Emotional Segments ===")
    for i, (start, end) in enumerate(results['emotional_segments'], 1):
        print(f"Segment {i}: {start:.2f}s - {end:.2f}s (Duration: {end-start:.2f}s)")
    
    print("\n=== Neutral Segments ===")
    for i, (start, end) in enumerate(results['neutral_segments'], 1):
        print(f"Segment {i}: {start:.2f}s - {end:.2f}s (Duration: {end-start:.2f}s)")
    
    # Visualize results
    detector.visualize_results(bvp_signal, results, 
                              save_path='emotion_detection_results.png')
    
    # Export segments
    segments_df = detector.export_segments(results, 'emotion_segments.csv')
    print(f"\nTotal emotional time: {segments_df[segments_df['state']=='Emotional']['duration'].sum():.2f}s")
    print(f"Total neutral time: {segments_df[segments_df['state']=='Neutral']['duration'].sum():.2f}s")


if __name__ == "__main__":
    main()
