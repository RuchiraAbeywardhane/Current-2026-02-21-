"""
Dataset Comparison Analysis: Raw vs Preprocessed EEG Data
==========================================================

This script compares the raw dataset with the preprocessed dataset to analyze:
1. Data availability (files, clips, windows)
2. Signal quality metrics (SNR, artifacts, variance)
3. Feature distributions (DE, PSD, temporal stats)
4. Class distribution balance
5. Subject-level statistics
6. Preprocessing impact on model performance

Datasets:
---------
- Raw: /kaggle/input/datasets/ruchiabey/emognition (*_STIMULUS_MUSE.json)
- Preprocessed: /kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR (*_STIMULUS_MUSE_cleaned.json)

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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


# ==================================================
# CONFIGURATION
# ==================================================

class ComparisonConfig:
    """Configuration for dataset comparison."""
    # Dataset paths
    RAW_DATA_ROOT = "/kaggle/input/datasets/ruchiabey/asr-outputv2-0/ASR_output"
    PREPROCESSED_DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"
    
    # EEG parameters
    EEG_FS = 256.0
    EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
    EEG_WINDOW_SEC = 10.0
    
    # Emotion mapping
    EMOTION_MAP = {
        "ENTHUSIASM": "Q1",
        "FEAR": "Q2",
        "SADNESS": "Q3",
        "NEUTRAL": "Q4"
    }
    
    # Analysis parameters
    SEED = 42
    MAX_FILES_PER_DATASET = None  # None = all files, or set a number for quick testing
    
    # Output paths
    OUTPUT_DIR = "dataset_comparison_results"
    REPORT_FILE = "comparison_report.txt"
    FIGURES_DIR = "figures"


config = ComparisonConfig()
random.seed(config.SEED)
np.random.seed(config.SEED)

# Create output directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(config.OUTPUT_DIR, config.FIGURES_DIR), exist_ok=True)


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

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


def calculate_snr(signal):
    """Calculate Signal-to-Noise Ratio."""
    signal_power = np.mean(signal ** 2)
    noise = np.diff(signal)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)


def calculate_signal_quality_metrics(signal):
    """Calculate comprehensive signal quality metrics."""
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = np.mean(signal)
    metrics['std'] = np.std(signal)
    metrics['variance'] = np.var(signal)
    metrics['skewness'] = skew(signal)
    metrics['kurtosis'] = kurtosis(signal)
    
    # Signal quality
    metrics['snr_db'] = calculate_snr(signal)
    metrics['zero_crossings'] = np.sum(np.diff(np.sign(signal)) != 0)
    metrics['range'] = np.max(signal) - np.min(signal)
    
    # Outliers (beyond 3 std)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    outliers = np.abs(signal - mean_val) > 3 * std_val
    metrics['outlier_ratio'] = np.sum(outliers) / len(signal)
    
    # Nan/Inf count
    metrics['nan_count'] = np.sum(~np.isfinite(signal))
    
    return metrics


def calculate_frequency_metrics(signal, fs=256.0):
    """Calculate frequency-domain metrics."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    
    metrics = {}
    
    # Band powers
    bands = [
        ('delta', 1, 3),
        ('theta', 4, 7),
        ('alpha', 8, 13),
        ('beta', 14, 30),
        ('gamma', 31, 45)
    ]
    
    for band_name, low, high in bands:
        idx = (freqs >= low) & (freqs <= high)
        metrics[f'{band_name}_power'] = np.sum(psd[idx])
    
    # Dominant frequency
    dominant_idx = np.argmax(psd)
    metrics['dominant_freq'] = freqs[dominant_idx]
    
    return metrics


# ==================================================
# DATA LOADING
# ==================================================

def load_dataset(data_root, dataset_name, file_pattern="*_STIMULUS_MUSE.json"):
    """Load dataset and extract comprehensive statistics."""
    print("\n" + "="*80)
    print(f"LOADING {dataset_name.upper()} DATASET")
    print("="*80)
    print(f"Path: {data_root}")
    print(f"Pattern: {file_pattern}")
    
    # Find files
    patterns = [
        os.path.join(data_root, file_pattern),
        os.path.join(data_root, "*", file_pattern),
        os.path.join(data_root, "*", "*_cleaned", file_pattern)  # For preprocessed nested structure
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    
    if config.MAX_FILES_PER_DATASET:
        files = files[:config.MAX_FILES_PER_DATASET]
    
    print(f"Found {len(files)} files")
    
    if len(files) == 0:
        print(f"⚠️  WARNING: No files found!")
        return None
    
    # Initialize storage
    dataset_info = {
        'name': dataset_name,
        'files': [],
        'subjects': set(),
        'emotions': Counter(),
        'clips': [],
        'signal_stats': defaultdict(list),
        'channel_stats': {ch: defaultdict(list) for ch in config.EEG_CHANNELS},
        'quality_issues': Counter(),
        'total_samples': 0,
        'valid_samples': 0,
        'raw_signals': [],  # Store some raw signals for visualization
    }
    
    # Process each file
    for file_idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        
        if len(parts) < 2:
            dataset_info['quality_issues']['invalid_filename'] += 1
            continue
        
        # Skip baseline files
        if "BASELINE" in fname:
            dataset_info['quality_issues']['baseline_file'] += 1
            continue
        
        subject = parts[0]
        emotion = parts[1].upper()
        
        if emotion not in config.EMOTION_MAP:
            dataset_info['quality_issues']['unknown_emotion'] += 1
            continue
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Extract channels
            tp9 = _interp_nan(_to_num(data.get("RAW_TP9", [])))
            af7 = _interp_nan(_to_num(data.get("RAW_AF7", [])))
            af8 = _interp_nan(_to_num(data.get("RAW_AF8", [])))
            tp10 = _interp_nan(_to_num(data.get("RAW_TP10", [])))
            
            L = min(len(tp9), len(af7), len(af8), len(tp10))
            
            if L == 0:
                dataset_info['quality_issues']['empty_signal'] += 1
                continue
            
            dataset_info['total_samples'] += L
            
            # Quality filtering (if available)
            if 'HSI_TP9' in data and 'HeadBandOn' in data:
                hsi_tp9 = _to_num(data.get("HSI_TP9", []))[:L]
                hsi_af7 = _to_num(data.get("HSI_AF7", []))[:L]
                hsi_af8 = _to_num(data.get("HSI_AF8", []))[:L]
                hsi_tp10 = _to_num(data.get("HSI_TP10", []))[:L]
                head_on = _to_num(data.get("HeadBandOn", []))[:L]
                
                quality_mask = (
                    (head_on == 1) &
                    np.isfinite(hsi_tp9) & (hsi_tp9 <= 2) &
                    np.isfinite(hsi_af7) & (hsi_af7 <= 2) &
                    np.isfinite(hsi_af8) & (hsi_af8 <= 2) &
                    np.isfinite(hsi_tp10) & (hsi_tp10 <= 2)
                )
                
                quality_ratio = np.sum(quality_mask) / len(quality_mask)
                dataset_info['signal_stats']['quality_ratio'].append(quality_ratio)
            else:
                quality_mask = np.ones(L, dtype=bool)
                dataset_info['signal_stats']['quality_ratio'].append(1.0)
            
            tp9_clean = tp9[:L][quality_mask]
            af7_clean = af7[:L][quality_mask]
            af8_clean = af8[:L][quality_mask]
            tp10_clean = tp10[:L][quality_mask]
            
            L_clean = len(tp9_clean)
            
            if L_clean < int(config.EEG_WINDOW_SEC * config.EEG_FS):
                dataset_info['quality_issues']['insufficient_length'] += 1
                continue
            
            dataset_info['valid_samples'] += L_clean
            
            # Store file info
            dataset_info['files'].append(fname)
            dataset_info['subjects'].add(subject)
            dataset_info['emotions'][emotion] += 1
            dataset_info['clips'].append(f"{subject}_{emotion}")
            
            # Calculate statistics per channel
            for ch_name, ch_data in zip(config.EEG_CHANNELS, [tp9_clean, af7_clean, af8_clean, tp10_clean]):
                # Signal quality metrics
                quality = calculate_signal_quality_metrics(ch_data)
                for key, val in quality.items():
                    dataset_info['channel_stats'][ch_name][key].append(val)
                
                # Frequency metrics
                freq_metrics = calculate_frequency_metrics(ch_data, config.EEG_FS)
                for key, val in freq_metrics.items():
                    dataset_info['channel_stats'][ch_name][key].append(val)
            
            # Store overall signal length
            dataset_info['signal_stats']['signal_length'].append(L_clean)
            dataset_info['signal_stats']['duration_sec'].append(L_clean / config.EEG_FS)
            
            # Store first few signals for visualization
            if len(dataset_info['raw_signals']) < 10:
                dataset_info['raw_signals'].append({
                    'subject': subject,
                    'emotion': emotion,
                    'tp9': tp9_clean[:int(5 * config.EEG_FS)],  # First 5 seconds
                    'af7': af7_clean[:int(5 * config.EEG_FS)],
                    'af8': af8_clean[:int(5 * config.EEG_FS)],
                    'tp10': tp10_clean[:int(5 * config.EEG_FS)]
                })
        
        except Exception as e:
            dataset_info['quality_issues']['parse_error'] += 1
            continue
        
        # Progress
        if (file_idx + 1) % 50 == 0:
            print(f"   Processed {file_idx + 1}/{len(files)} files...")
    
    # Summary
    print(f"\n📊 {dataset_name.upper()} SUMMARY:")
    print(f"   Valid files: {len(dataset_info['files'])}")
    print(f"   Subjects: {len(dataset_info['subjects'])}")
    print(f"   Clips: {len(dataset_info['clips'])}")
    print(f"   Total samples: {dataset_info['total_samples']:,}")
    print(f"   Valid samples: {dataset_info['valid_samples']:,}")
    print(f"   Sample retention: {100*dataset_info['valid_samples']/max(1, dataset_info['total_samples']):.1f}%")
    print(f"\n   Emotion distribution:")
    for emotion, count in dataset_info['emotions'].most_common():
        print(f"      {emotion}: {count}")
    
    if dataset_info['quality_issues']:
        print(f"\n   Quality issues:")
        for issue, count in dataset_info['quality_issues'].items():
            print(f"      {issue}: {count}")
    
    return dataset_info


# ==================================================
# COMPARISON ANALYSIS
# ==================================================

def compare_signal_quality(raw_data, preprocessed_data, output_dir):
    """Compare signal quality metrics between datasets."""
    print("\n" + "="*80)
    print("SIGNAL QUALITY COMPARISON")
    print("="*80)
    
    if raw_data is None or preprocessed_data is None:
        print("⚠️  One or both datasets are missing!")
        return
    
    comparison_results = {}
    
    for ch in config.EEG_CHANNELS:
        print(f"\n📡 Channel: {ch}")
        print("-" * 40)
        
        comparison_results[ch] = {}
        
        # Compare each metric
        metrics_to_compare = ['snr_db', 'std', 'outlier_ratio', 'nan_count']
        
        for metric in metrics_to_compare:
            raw_values = raw_data['channel_stats'][ch].get(metric, [])
            prep_values = preprocessed_data['channel_stats'][ch].get(metric, [])
            
            if not raw_values or not prep_values:
                continue
            
            raw_mean = np.mean(raw_values)
            prep_mean = np.mean(prep_values)
            improvement = ((prep_mean - raw_mean) / abs(raw_mean)) * 100 if raw_mean != 0 else 0
            
            print(f"   {metric}:")
            print(f"      Raw: {raw_mean:.3f}")
            print(f"      Preprocessed: {prep_mean:.3f}")
            print(f"      Change: {improvement:+.1f}%")
            
            comparison_results[ch][metric] = {
                'raw_mean': raw_mean,
                'prep_mean': prep_mean,
                'improvement_pct': improvement
            }
            
            # Statistical test
            if len(raw_values) > 1 and len(prep_values) > 1:
                t_stat, p_value = stats.ttest_ind(raw_values, prep_values)
                print(f"      p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")
                comparison_results[ch][metric]['p_value'] = p_value
    
    return comparison_results


def compare_frequency_content(raw_data, preprocessed_data, output_dir):
    """Compare frequency content between datasets."""
    print("\n" + "="*80)
    print("FREQUENCY CONTENT COMPARISON")
    print("="*80)
    
    if raw_data is None or preprocessed_data is None:
        print("⚠️  One or both datasets are missing!")
        return
    
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, ch in enumerate(config.EEG_CHANNELS):
        ax = axes[idx]
        
        # Get band powers
        raw_powers = []
        prep_powers = []
        
        for band in bands:
            raw_band = raw_data['channel_stats'][ch].get(f'{band}_power', [])
            prep_band = preprocessed_data['channel_stats'][ch].get(f'{band}_power', [])
            
            raw_powers.append(np.mean(raw_band) if raw_band else 0)
            prep_powers.append(np.mean(prep_band) if prep_band else 0)
        
        # Plot comparison
        x = np.arange(len(bands))
        width = 0.35
        
        ax.bar(x - width/2, raw_powers, width, label='Raw', alpha=0.8)
        ax.bar(x + width/2, prep_powers, width, label='Preprocessed', alpha=0.8)
        
        ax.set_title(f'{ch} - Band Power Comparison')
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Power (μV²)')
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Remove extra subplot
    axes[-2].axis('off')
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, config.FIGURES_DIR, 'frequency_comparison.png'), dpi=150)
    print(f"   ✅ Saved frequency comparison plot")
    plt.close()


def compare_sample_signals(raw_data, preprocessed_data, output_dir):
    """Compare sample raw signals."""
    print("\n" + "="*80)
    print("SAMPLE SIGNAL COMPARISON")
    print("="*80)
    
    if raw_data is None or preprocessed_data is None:
        print("⚠️  One or both datasets are missing!")
        return
    
    # Plot first 3 matching subjects
    n_samples = min(3, len(raw_data['raw_signals']), len(preprocessed_data['raw_signals']))
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        raw_sig = raw_data['raw_signals'][i]
        prep_sig = preprocessed_data['raw_signals'][i]
        
        # Time vector (5 seconds)
        t = np.arange(len(raw_sig['tp9'])) / config.EEG_FS
        
        # Raw signal
        ax_raw = axes[i, 0]
        for ch_name in config.EEG_CHANNELS:
            if ch_name.lower() in raw_sig:
                ax_raw.plot(t, raw_sig[ch_name.lower()], label=ch_name, alpha=0.7)
        ax_raw.set_title(f'Raw - {raw_sig["subject"]} ({raw_sig["emotion"]})')
        ax_raw.set_xlabel('Time (s)')
        ax_raw.set_ylabel('Amplitude (μV)')
        ax_raw.legend(loc='upper right')
        ax_raw.grid(alpha=0.3)
        
        # Preprocessed signal
        ax_prep = axes[i, 1]
        for ch_name in config.EEG_CHANNELS:
            if ch_name.lower() in prep_sig:
                ax_prep.plot(t, prep_sig[ch_name.lower()], label=ch_name, alpha=0.7)
        ax_prep.set_title(f'Preprocessed - {prep_sig["subject"]} ({prep_sig["emotion"]})')
        ax_prep.set_xlabel('Time (s)')
        ax_prep.set_ylabel('Amplitude (μV)')
        ax_prep.legend(loc='upper right')
        ax_prep.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, config.FIGURES_DIR, 'sample_signals.png'), dpi=150)
    print(f"   ✅ Saved sample signals plot")
    plt.close()


def generate_comparison_report(raw_data, preprocessed_data, comparison_results, output_dir):
    """Generate comprehensive comparison report."""
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    report_path = os.path.join(output_dir, config.REPORT_FILE)
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET COMPARISON REPORT: RAW vs PREPROCESSED\n")
        f.write("="*80 + "\n\n")
        
        # Dataset overview
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n\n")
        
        for dataset, name in [(raw_data, "RAW"), (preprocessed_data, "PREPROCESSED")]:
            if dataset is None:
                f.write(f"{name} Dataset: NOT AVAILABLE\n\n")
                continue
            
            f.write(f"{name} Dataset:\n")
            f.write(f"   Files: {len(dataset['files'])}\n")
            f.write(f"   Subjects: {len(dataset['subjects'])}\n")
            f.write(f"   Clips: {len(dataset['clips'])}\n")
            f.write(f"   Total samples: {dataset['total_samples']:,}\n")
            f.write(f"   Valid samples: {dataset['valid_samples']:,}\n")
            f.write(f"   Retention rate: {100*dataset['valid_samples']/max(1, dataset['total_samples']):.1f}%\n")
            f.write(f"   Avg duration: {np.mean(dataset['signal_stats']['duration_sec']):.1f}s\n")
            f.write(f"\n   Emotion distribution:\n")
            for emotion, count in dataset['emotions'].most_common():
                f.write(f"      {emotion}: {count}\n")
            f.write("\n")
        
        # Signal quality comparison
        if comparison_results:
            f.write("\n2. SIGNAL QUALITY IMPROVEMENTS\n")
            f.write("-" * 80 + "\n\n")
            
            for ch in config.EEG_CHANNELS:
                f.write(f"{ch}:\n")
                if ch in comparison_results:
                    for metric, values in comparison_results[ch].items():
                        if isinstance(values, dict):
                            f.write(f"   {metric}:\n")
                            f.write(f"      Raw: {values['raw_mean']:.3f}\n")
                            f.write(f"      Preprocessed: {values['prep_mean']:.3f}\n")
                            f.write(f"      Improvement: {values['improvement_pct']:+.1f}%\n")
                            if 'p_value' in values:
                                sig = " (significant)" if values['p_value'] < 0.05 else ""
                                f.write(f"      p-value: {values['p_value']:.4f}{sig}\n")
                f.write("\n")
        
        # Recommendations
        f.write("\n3. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        if raw_data and preprocessed_data:
            snr_improvement = np.mean([
                comparison_results.get(ch, {}).get('snr_db', {}).get('improvement_pct', 0)
                for ch in config.EEG_CHANNELS
            ])
            
            if snr_improvement > 10:
                f.write("✅ Preprocessed dataset shows significant SNR improvement (>10%).\n")
                f.write("   RECOMMENDATION: Use preprocessed dataset for model training.\n\n")
            elif snr_improvement > 0:
                f.write("⚠️  Preprocessed dataset shows moderate improvement.\n")
                f.write("   RECOMMENDATION: Compare model performance on both datasets.\n\n")
            else:
                f.write("⚠️  Preprocessing shows minimal or negative impact on SNR.\n")
                f.write("   RECOMMENDATION: Review preprocessing pipeline or use raw data.\n\n")
            
            retention_raw = raw_data['valid_samples'] / max(1, raw_data['total_samples'])
            retention_prep = preprocessed_data['valid_samples'] / max(1, preprocessed_data['total_samples'])
            
            if retention_prep > retention_raw:
                f.write(f"✅ Preprocessed dataset has better data retention ({retention_prep:.1%} vs {retention_raw:.1%}).\n\n")
            else:
                f.write(f"⚠️  Raw dataset has better data retention ({retention_raw:.1%} vs {retention_prep:.1%}).\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"   ✅ Report saved to: {report_path}")


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """Run complete dataset comparison analysis."""
    print("="*80)
    print("DATASET COMPARISON ANALYSIS")
    print("="*80)
    print(f"Raw dataset: {config.RAW_DATA_ROOT}")
    print(f"Preprocessed dataset: {config.PREPROCESSED_DATA_ROOT}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print("="*80)
    
    # Load datasets
    raw_data = load_dataset(
        config.RAW_DATA_ROOT,
        "raw",
        file_pattern="*_STIMULUS_MUSE.json"
    )
    
    preprocessed_data = load_dataset(
        config.PREPROCESSED_DATA_ROOT,
        "preprocessed",
        file_pattern="*_STIMULUS_MUSE_cleaned.json"
    )
    
    # Compare signal quality
    comparison_results = compare_signal_quality(raw_data, preprocessed_data, config.OUTPUT_DIR)
    
    # Compare frequency content
    compare_frequency_content(raw_data, preprocessed_data, config.OUTPUT_DIR)
    
    # Compare sample signals
    compare_sample_signals(raw_data, preprocessed_data, config.OUTPUT_DIR)
    
    # Generate report
    generate_comparison_report(raw_data, preprocessed_data, comparison_results, config.OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("🎉 COMPARISON ANALYSIS COMPLETE! 🎉")
    print("="*80)
    print(f"Results saved to: {config.OUTPUT_DIR}/")
    print(f"   - Report: {config.REPORT_FILE}")
    print(f"   - Figures: {config.FIGURES_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()
