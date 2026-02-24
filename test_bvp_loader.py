"""
BVP Data Loader Test Script
============================

This script tests the BVP data loader module to verify:
- Load ALL BVP data from dataset
- Load baseline recordings per subject
- Apply baseline reduction for ALL subjects
- Save sample visualizations from different subjects

Usage:
    python test_bvp_loader.py

Author: Final Year Project
Date: 2026-02-24
"""

import os
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import BVP configuration
from bvp_config import BVPConfig

# Import BVP data loader
from bvp_data_loader import (
    preprocess_bvp_signal,
    apply_baseline_reduction,
    _to_num,
    _interp_nan
)


# ==================================================
# CONFIGURATION
# ==================================================

# Use the existing BVP configuration
config = BVPConfig()

# Override specific settings for testing/visualization
config.OUTPUT_DIR = "bvp_baseline_reduction_results"
config.NUM_SUBJECTS_TO_VISUALIZE = 5  # Number of subjects to save visualizations for
config.VIS_DURATION_SEC = 30  # Show first 30 seconds in plots

# Set random seed
random.seed(config.SEED)
np.random.seed(config.SEED)


# ==================================================
# LOAD ALL BVP DATA WITH BASELINE REDUCTION
# ==================================================

def load_all_bvp_data_with_baseline_reduction(config):
    """
    Load ALL BVP data from dataset and apply baseline reduction per subject.
    
    Args:
        config: BVPConfig object with all settings
    
    Returns:
        baseline_dict: {subject_id: baseline_signal}
        stimulus_dict: {subject_id: [(emotion, signal_no_br, signal_with_br, filename), ...]}
        stats: Dictionary with statistics
    """
    print("\n" + "="*80)
    print("LOADING ALL BVP DATA WITH BASELINE REDUCTION")
    print("="*80)
    print(f"Data Root: {config.DATA_ROOT}")
    print(f"Device: EMPATICA")
    print(f"Sampling Rate: {config.BVP_FS} Hz")
    print(f"Baseline Correction: {config.USE_BVP_BASELINE_CORRECTION}")
    print(f"Baseline Reduction: {config.USE_BVP_BASELINE_REDUCTION}")
    print("="*80)
    
    # ============================================================
    # STEP 1: Load all baseline files
    # ============================================================
    print("\n📂 Step 1: Loading Empatica baseline files...")
    
    baseline_patterns = [
        os.path.join(config.DATA_ROOT, "*_BASELINE_EMPATICA.json"),
        os.path.join(config.DATA_ROOT, "*", "*_BASELINE_EMPATICA.json"),
        os.path.join(config.DATA_ROOT, "*_BASELINE_STIMULUS_EMPATICA.json"),
        os.path.join(config.DATA_ROOT, "*", "*_BASELINE_STIMULUS_EMPATICA.json")
    ]
    
    baseline_files = sorted({p for pat in baseline_patterns for p in glob.glob(pat)})
    print(f"   Found {len(baseline_files)} baseline files")
    
    baseline_dict = {}
    baseline_load_errors = 0
    
    for bpath in baseline_files:
        bname = os.path.basename(bpath)
        parts = bname.split("_")
        if len(parts) < 2:
            continue
        subject = parts[0]
        
        try:
            with open(bpath, "r") as f:
                bdata = json.load(f)
            
            # Try different field names
            bvp_baseline = None
            for field in ["BVPRaw", "BVP", "BVPProcessed"]:
                if field in bdata and bdata[field]:
                    bvp_baseline = bdata[field]
                    break
            
            if not bvp_baseline:
                continue
            
            # Handle different formats
            if isinstance(bvp_baseline[0], list):
                baseline_raw = np.array([row[1] for row in bvp_baseline], dtype=float)
            else:
                baseline_raw = _to_num(bvp_baseline)
            
            baseline_raw = _interp_nan(baseline_raw)
            
            # Preprocess baseline using config parameters
            baseline_processed = preprocess_bvp_signal(
                baseline_raw,
                fs=config.BVP_FS,
                highcut_hz=config.BVP_LOWPASS_CUTOFF,
                lowcut_hz=config.BVP_HIGHPASS_CUTOFF,
                filter_order=config.BVP_FILTER_ORDER,
                wavelet=config.BVP_WAVELET,
                denoise_level=config.BVP_DENOISE_LEVEL,
                use_baseline_correction=config.USE_BVP_BASELINE_CORRECTION,
                use_highpass=config.USE_BVP_HIGHPASS,
                normalize=False
            )
            
            baseline_dict[subject] = baseline_processed
            print(f"   ✅ {subject}: {len(baseline_processed)} samples ({len(baseline_processed)/config.BVP_FS:.1f}s)")
            
        except Exception as e:
            baseline_load_errors += 1
            continue
    
    print(f"\n   Total baselines loaded: {len(baseline_dict)}")
    print(f"   Subjects with baselines: {sorted(baseline_dict.keys())}")
    if baseline_load_errors > 0:
        print(f"   Failed to load: {baseline_load_errors} files")
    
    # ============================================================
    # STEP 2: Load all stimulus files
    # ============================================================
    print("\n📂 Step 2: Loading Empatica stimulus files...")
    
    stimulus_patterns = [
        os.path.join(config.DATA_ROOT, "*_STIMULUS_EMPATICA.json"),
        os.path.join(config.DATA_ROOT, "*", "*_STIMULUS_EMPATICA.json")
    ]
    
    stimulus_files = sorted({p for pat in stimulus_patterns for p in glob.glob(pat)})
    stimulus_files = [f for f in stimulus_files if "BASELINE" not in f]
    
    print(f"   Found {len(stimulus_files)} stimulus files")
    
    # ============================================================
    # STEP 3: Process each stimulus with baseline reduction
    # ============================================================
    print("\n🔧 Step 3: Processing stimulus files with baseline reduction...")
    
    stimulus_dict = defaultdict(list)
    
    processed_count = 0
    skipped_no_baseline = 0
    skipped_errors = 0
    skipped_unknown_emotion = 0
    
    for spath in stimulus_files:
        sname = os.path.basename(spath)
        parts = sname.split("_")
        
        if len(parts) < 2:
            skipped_errors += 1
            continue
        
        subject = parts[0]
        emotion = parts[1].upper()
        
        # Check if emotion is in config mapping
        if emotion not in config.SUPERCLASS_MAP:
            skipped_unknown_emotion += 1
            continue
        
        # Check if baseline exists for this subject
        if subject not in baseline_dict:
            skipped_no_baseline += 1
            continue
        
        try:
            # Load stimulus file
            with open(spath, "r") as f:
                sdata = json.load(f)
            
            # Try different field names
            bvp_stimulus = None
            for field in ["BVPRaw", "BVP", "BVPProcessed"]:
                if field in sdata and sdata[field]:
                    bvp_stimulus = sdata[field]
                    break
            
            if not bvp_stimulus:
                skipped_errors += 1
                continue
            
            # Handle different formats
            if isinstance(bvp_stimulus[0], list):
                stimulus_raw = np.array([row[1] for row in bvp_stimulus], dtype=float)
            else:
                stimulus_raw = _to_num(bvp_stimulus)
            
            stimulus_raw = _interp_nan(stimulus_raw)
            
            # Preprocess WITHOUT baseline reduction (for comparison)
            stimulus_no_br = preprocess_bvp_signal(
                stimulus_raw,
                fs=config.BVP_FS,
                highcut_hz=config.BVP_LOWPASS_CUTOFF,
                lowcut_hz=config.BVP_HIGHPASS_CUTOFF,
                filter_order=config.BVP_FILTER_ORDER,
                wavelet=config.BVP_WAVELET,
                denoise_level=config.BVP_DENOISE_LEVEL,
                use_baseline_correction=config.USE_BVP_BASELINE_CORRECTION,
                use_highpass=config.USE_BVP_HIGHPASS,
                normalize=True
            )
            
            # Preprocess WITH baseline reduction
            stimulus_processed = preprocess_bvp_signal(
                stimulus_raw,
                fs=config.BVP_FS,
                highcut_hz=config.BVP_LOWPASS_CUTOFF,
                lowcut_hz=config.BVP_HIGHPASS_CUTOFF,
                filter_order=config.BVP_FILTER_ORDER,
                wavelet=config.BVP_WAVELET,
                denoise_level=config.BVP_DENOISE_LEVEL,
                use_baseline_correction=config.USE_BVP_BASELINE_CORRECTION,
                use_highpass=config.USE_BVP_HIGHPASS,
                normalize=False
            )
            
            # Get baseline for this subject
            subject_baseline = baseline_dict[subject].copy()
            
            # Match baseline length to stimulus
            if len(subject_baseline) < len(stimulus_processed):
                n_repeats = int(np.ceil(len(stimulus_processed) / len(subject_baseline)))
                subject_baseline = np.tile(subject_baseline, n_repeats)[:len(stimulus_processed)]
            elif len(subject_baseline) > len(stimulus_processed):
                subject_baseline = subject_baseline[:len(stimulus_processed)]
            
            # Apply baseline reduction
            stimulus_with_br = apply_baseline_reduction(stimulus_processed, subject_baseline)
            
            # Normalize after reduction
            vmin, vmax = np.min(stimulus_with_br), np.max(stimulus_with_br)
            if not np.isclose(vmin, vmax):
                stimulus_with_br = (stimulus_with_br - vmin) / (vmax - vmin)
            
            # Get superclass label
            superclass = config.SUPERCLASS_MAP[emotion]
            
            # Store results
            stimulus_dict[subject].append((superclass, stimulus_no_br, stimulus_with_br, sname))
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"   Processed {processed_count} files...")
            
        except Exception as e:
            skipped_errors += 1
            continue
    
    print(f"\n   ✅ Successfully processed: {processed_count} files")
    print(f"   ⏭️  Skipped (no baseline): {skipped_no_baseline} files")
    print(f"   ⏭️  Skipped (unknown emotion): {skipped_unknown_emotion} files")
    print(f"   ❌ Skipped (errors): {skipped_errors} files")
    
    # ============================================================
    # STEP 4: Compile statistics
    # ============================================================
    stats = {
        'total_baselines': len(baseline_dict),
        'total_subjects_with_data': len(stimulus_dict),
        'total_stimulus_processed': processed_count,
        'subjects': list(stimulus_dict.keys()),
        'emotions_per_subject': {subj: [e for e, _, _, _ in trials] for subj, trials in stimulus_dict.items()}
    }
    
    print("\n📊 Summary Statistics:")
    print(f"   Total subjects with baselines: {stats['total_baselines']}")
    print(f"   Total subjects with stimulus data: {stats['total_subjects_with_data']}")
    print(f"   Total stimulus files processed: {stats['total_stimulus_processed']}")
    if len(stimulus_dict) > 0:
        print(f"   Average trials per subject: {processed_count / len(stimulus_dict):.1f}")
    
    return baseline_dict, stimulus_dict, stats


# ==================================================
# VISUALIZATION
# ==================================================

def save_subject_visualizations(baseline_dict, stimulus_dict, config):
    """
    Save visualizations for multiple subjects showing baseline reduction effects.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Select subjects to visualize
    subjects = list(stimulus_dict.keys())
    if len(subjects) > config.NUM_SUBJECTS_TO_VISUALIZE:
        # Select subjects with most trials
        subjects = sorted(subjects, key=lambda s: len(stimulus_dict[s]), reverse=True)[:config.NUM_SUBJECTS_TO_VISUALIZE]
    
    print(f"\n📊 Creating visualizations for {len(subjects)} subjects...")
    
    vis_length = config.BVP_FS * config.VIS_DURATION_SEC
    zoom_length = config.BVP_FS * 5  # 5 seconds zoom window
    
    for idx, subject in enumerate(subjects, 1):
        print(f"\n   Subject {idx}/{len(subjects)}: {subject}")
        
        baseline = baseline_dict[subject]
        trials = stimulus_dict[subject]
        
        print(f"      Baseline: {len(baseline)} samples ({len(baseline)/config.BVP_FS:.1f}s)")
        print(f"      Trials: {len(trials)}")
        
        # Create figure for this subject with 4 columns
        fig, axes = plt.subplots(len(trials) + 1, 4, figsize=(24, 4 * (len(trials) + 1)))
        
        if len(trials) == 0:
            axes = axes.reshape(1, 4)
        elif len(trials) == 1:
            axes = axes.reshape(2, 4)
        
        time_full = np.arange(min(vis_length, len(baseline))) / config.BVP_FS
        time_zoom = np.arange(min(zoom_length, len(baseline))) / config.BVP_FS
        
        # Row 0: Baseline
        baseline_vis = baseline[:min(vis_length, len(baseline))]
        baseline_zoom = baseline[:min(zoom_length, len(baseline))]
        
        # Column 0: Full baseline
        axes[0, 0].plot(time_full, baseline_vis, 'g-', linewidth=0.8)
        axes[0, 0].set_title(f'Subject {subject} - Baseline (Resting)', fontweight='bold')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Column 1: Zoomed baseline
        axes[0, 1].plot(time_zoom, baseline_zoom, 'g-', linewidth=1.2)
        axes[0, 1].set_title(f'Baseline - ZOOMED (First 5s)', fontweight='bold', color='darkgreen')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Column 2: Info
        axes[0, 2].text(0.5, 0.5, f'Subject: {subject}\nBaseline Recording\n(Resting State)', 
                       ha='center', va='center', fontsize=12, transform=axes[0, 2].transAxes)
        axes[0, 2].axis('off')
        
        # Column 3: Stats
        axes[0, 3].text(0.5, 0.5, f'Total Trials: {len(trials)}\nEmotions: {", ".join(set([e for e, _, _, _ in trials]))}', 
                       ha='center', va='center', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].axis('off')
        
        # Subsequent rows: Each trial
        for trial_idx, (emotion, signal_no_br, signal_with_br, filename) in enumerate(trials, 1):
            time_trial_full = np.arange(min(vis_length, len(signal_no_br))) / config.BVP_FS
            time_trial_zoom = np.arange(min(zoom_length, len(signal_no_br))) / config.BVP_FS
            
            signal_no_br_vis = signal_no_br[:min(vis_length, len(signal_no_br))]
            signal_with_br_vis = signal_with_br[:min(vis_length, len(signal_with_br))]
            
            signal_no_br_zoom = signal_no_br[:min(zoom_length, len(signal_no_br))]
            signal_with_br_zoom = signal_with_br[:min(zoom_length, len(signal_with_br))]
            
            # Column 0: Without baseline reduction (full)
            axes[trial_idx, 0].plot(time_trial_full, signal_no_br_vis, 'orange', linewidth=0.8)
            axes[trial_idx, 0].set_title(f'{emotion} - WITHOUT Baseline Reduction', fontweight='bold')
            axes[trial_idx, 0].set_ylabel('Normalized')
            axes[trial_idx, 0].grid(True, alpha=0.3)
            
            # Column 1: With baseline reduction (full)
            axes[trial_idx, 1].plot(time_trial_full, signal_with_br_vis, 'purple', linewidth=0.8)
            axes[trial_idx, 1].set_title(f'{emotion} - WITH Baseline Reduction', fontweight='bold')
            axes[trial_idx, 1].set_ylabel('Normalized')
            axes[trial_idx, 1].grid(True, alpha=0.3)
            
            # Column 2: Comparison (full)
            axes[trial_idx, 2].plot(time_trial_full, signal_no_br_vis, 'orange', linewidth=0.8, 
                                   label='No BR', alpha=0.7)
            axes[trial_idx, 2].plot(time_trial_full, signal_with_br_vis, 'purple', linewidth=0.8, 
                                   label='With BR', alpha=0.7)
            axes[trial_idx, 2].set_title(f'{emotion} - Comparison', fontweight='bold')
            axes[trial_idx, 2].set_ylabel('Normalized')
            axes[trial_idx, 2].legend(loc='upper right', fontsize=8)
            axes[trial_idx, 2].grid(True, alpha=0.3)
            
            # Column 3: ZOOMED Comparison (first 5 seconds)
            axes[trial_idx, 3].plot(time_trial_zoom, signal_no_br_zoom, 'orange', linewidth=1.5, 
                                   label='No BR', alpha=0.8, marker='o', markersize=2, markevery=10)
            axes[trial_idx, 3].plot(time_trial_zoom, signal_with_br_zoom, 'purple', linewidth=1.5, 
                                   label='With BR', alpha=0.8, marker='s', markersize=2, markevery=10)
            axes[trial_idx, 3].set_title(f'{emotion} - ZOOMED Comparison (First 5s)', 
                                        fontweight='bold', color='darkred')
            axes[trial_idx, 3].set_ylabel('Normalized')
            axes[trial_idx, 3].legend(loc='upper right', fontsize=8)
            axes[trial_idx, 3].grid(True, alpha=0.3)
            
            # Add shaded region to show zoom area in full comparison
            axes[trial_idx, 2].axvspan(0, 5, alpha=0.1, color='red', label='Zoom Region')
        
        # Set x-label for bottom row
        for col in range(4):
            if col == 1 or col == 3:
                axes[-1, col].set_xlabel('Time (seconds) - ZOOMED')
            else:
                axes[-1, col].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        output_path = os.path.join(config.OUTPUT_DIR, f'subject_{subject}_baseline_reduction.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"      ✅ Saved: {output_path}")
        plt.close()
    
    print(f"\n✅ All visualizations saved to: {config.OUTPUT_DIR}/")


def create_summary_report(baseline_dict, stimulus_dict, stats, config):
    """Create a summary report of the baseline reduction process."""
    print("\n" + "="*80)
    print("CREATING SUMMARY REPORT")
    print("="*80)
    
    report_path = os.path.join(config.OUTPUT_DIR, 'baseline_reduction_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BVP BASELINE REDUCTION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Data Root: {config.DATA_ROOT}\n")
        f.write(f"Sampling Rate: {config.BVP_FS} Hz\n")
        f.write(f"Baseline Correction: {config.USE_BVP_BASELINE_CORRECTION}\n")
        f.write(f"Baseline Reduction: {config.USE_BVP_BASELINE_REDUCTION}\n")
        f.write(f"Highpass Filter: {config.USE_BVP_HIGHPASS}\n")
        f.write(f"Lowpass Cutoff: {config.BVP_LOWPASS_CUTOFF} Hz\n")
        f.write(f"Highpass Cutoff: {config.BVP_HIGHPASS_CUTOFF} Hz\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total subjects with baselines: {stats['total_baselines']}\n")
        f.write(f"Total subjects with stimulus data: {stats['total_subjects_with_data']}\n")
        f.write(f"Total stimulus files processed: {stats['total_stimulus_processed']}\n")
        if stats['total_subjects_with_data'] > 0:
            f.write(f"Average trials per subject: {stats['total_stimulus_processed'] / stats['total_subjects_with_data']:.1f}\n\n")
        
        f.write("SUBJECTS WITH BASELINES\n")
        f.write("-"*80 + "\n")
        for subject in sorted(baseline_dict.keys()):
            baseline_length = len(baseline_dict[subject])
            baseline_duration = baseline_length / config.BVP_FS
            num_trials = len(stimulus_dict.get(subject, []))
            
            f.write(f"{subject}: Baseline={baseline_duration:.1f}s, Trials={num_trials}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED BREAKDOWN BY SUBJECT\n")
        f.write("="*80 + "\n\n")
        
        for subject in sorted(stimulus_dict.keys()):
            trials = stimulus_dict[subject]
            emotions = [e for e, _, _, _ in trials]
            emotion_counts = {}
            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            
            f.write(f"Subject: {subject}\n")
            f.write(f"  Total trials: {len(trials)}\n")
            f.write(f"  Emotions: {dict(emotion_counts)}\n")
            f.write(f"  Files:\n")
            for emotion, _, _, filename in trials:
                f.write(f"    - {filename} ({emotion})\n")
            f.write("\n")
    
    print(f"✅ Summary report saved: {report_path}")


# ==================================================
# MAIN TEST EXECUTION
# ==================================================

def main():
    """Main test execution."""
    print("="*80)
    print("BVP BASELINE REDUCTION - COMPLETE DATASET TEST")
    print("="*80)
    print(f"Using BVPConfig from bvp_config.py")
    print(f"Data path: {config.DATA_ROOT}")
    print(f"BVP sampling rate: {config.BVP_FS} Hz")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Subjects to visualize: {config.NUM_SUBJECTS_TO_VISUALIZE}")
    print("="*80)
    
    # Load all data with baseline reduction
    baseline_dict, stimulus_dict, stats = load_all_bvp_data_with_baseline_reduction(config)
    
    # Save visualizations for selected subjects
    save_subject_visualizations(baseline_dict, stimulus_dict, config)
    
    # Create summary report
    create_summary_report(baseline_dict, stimulus_dict, stats, config)
    
    print("\n" + "="*80)
    print("🎉 BASELINE REDUCTION TEST COMPLETE!")
    print("="*80)
    print(f"\n📁 Output Directory: {config.OUTPUT_DIR}/")
    print(f"   - Subject visualizations: subject_*.png")
    print(f"   - Summary report: baseline_reduction_report.txt")
    print("="*80)


if __name__ == "__main__":
    main()
