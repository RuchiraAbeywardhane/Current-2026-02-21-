"""
EEG Data Loader – Emoky Dataset (EKM-ED)
=========================================

Loads the EmoKey Moments EEG Dataset (EKM-ED) whose structure is:

    <DATA_ROOT>/
    └── 0.0078125S/          ← timestep folder (1/128 s  →  128 Hz)
        ├── 1/               ← subject ID folder
        │   ├── ANGER.csv
        │   ├── FEAR.csv
        │   ├── HAPPINESS.csv
        │   ├── NEUTRAL_ANGER.csv
        │   ├── NEUTRAL_FEAR.csv
        │   ├── NEUTRAL_HAPPINESS.csv
        │   ├── NEUTRAL_SADNESS.csv
        │   └── SADNESS.csv
        └── 103/

Key differences vs the Emognition loader:
- CSV files, not JSON
- Emotion label  = filename stem (ANGER, FEAR, …)
- Subject ID     = parent folder name (numeric, e.g. "1", "103")
- Sampling rate  = 128 Hz  (timestep 0.0078125 s)
- Baseline       = NEUTRAL_<EMOTION>.csv for each trial emotion
                   (per-subject, per-emotion neutral recording)
- No HeadBandOn / HSI quality cols when all-ones – handled gracefully

Emotion → quadrant mapping (same 4-class scheme as the rest of the pipeline):
    HAPPINESS  → Q1  (Positive + High Arousal)
    FEAR       → Q2  (Negative + High Arousal)
    ANGER      → Q2  (Negative + High Arousal)
    SADNESS    → Q3  (Negative + Low Arousal)

Returns the same 5-tuple as eeg_data_loader.py:
    X_raw, y_labels, subject_ids, trial_ids, label_to_id

Author: Final Year Project
Date: 2026
"""

import os
import glob
from collections import Counter

import numpy as np
import pandas as pd

from eeg_feature_extractor import extract_eeg_features   # shared module


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Sampling rate derived from the folder name: 0.0078125 s = 1/128 s
EMOKY_FS = 128.0

# Timestep sub-folder name (used to locate the data)
EMOKY_TIMESTEP_FOLDER = "0.0078125S"

# Raw EEG channel column names (same MUSE layout as Emognition)
RAW_COLS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

# Quality-filter column names (may be absent or all-1 in some subjects)
HSI_COLS  = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]
HEADBAND_COL = "HeadBandOn"

# Emotion filenames that are actual stimuli (not neutral baselines)
STIMULUS_EMOTIONS = {"ANGER", "FEAR", "HAPPINESS", "SADNESS"}

# Neutral baseline filename stems map to their emotion counterpart
NEUTRAL_MAP = {
    "NEUTRAL_ANGER":     "ANGER",
    "NEUTRAL_FEAR":      "FEAR",
    "NEUTRAL_HAPPINESS": "HAPPINESS",
    "NEUTRAL_SADNESS":   "SADNESS",
}

# Emotion → 4-quadrant superclass
#   Q1 = Positive + High Arousal
#   Q2 = Negative + High Arousal
#   Q3 = Negative + Low Arousal
#   Q4 = Positive + Low Arousal  (not present in this dataset)
EMOKY_SUPERCLASS_MAP = {
    "HAPPINESS": "Q1",
    "FEAR":      "Q2",
    "ANGER":     "Q2",
    "SADNESS":   "Q3",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN / Inf values in a 1-D array."""
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _read_raw_channels(csv_path: str):
    """
    Read a MUSE CSV file and return the 4 raw EEG channels as a (T, 4) float32
    array, after NaN interpolation and mean-centering.

    Also applies quality filtering (HeadBandOn == 1 and HSI_* <= 2) when the
    relevant columns are present and meaningful (i.e. not all-ones).

    Returns
    -------
    signal : np.ndarray, shape (T, 4) or shape (0, 4) if unusable
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"      ⚠️  Cannot read {csv_path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)

    # ── Raw channels ──────────────────────────────────────────────────────────
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        print(f"      ⚠️  Missing columns {missing} in {os.path.basename(csv_path)}")
        return np.zeros((0, 4), dtype=np.float32)

    ch = {}
    for col in RAW_COLS:
        ch[col] = _interp_nan(pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64))

    L = min(len(v) for v in ch.values())
    if L == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # ── Quality mask ──────────────────────────────────────────────────────────
    mask = (
        np.isfinite(ch["RAW_TP9"][:L])  &
        np.isfinite(ch["RAW_AF7"][:L])  &
        np.isfinite(ch["RAW_AF8"][:L])  &
        np.isfinite(ch["RAW_TP10"][:L])
    )

    # HeadBandOn filter (only if column exists and is not trivially all-1)
    if HEADBAND_COL in df.columns:
        head_on = pd.to_numeric(df[HEADBAND_COL], errors="coerce").to_numpy(np.float64)[:L]
        if not np.all(head_on == 1.0):           # skip filter if col is all-1
            mask &= (head_on == 1.0)

    # HSI filter (only if all 4 cols exist and are not trivially all-1)
    hsi_present = all(c in df.columns for c in HSI_COLS)
    if hsi_present:
        hsi_arrays = [
            pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)[:L]
            for c in HSI_COLS
        ]
        # Only apply if values actually vary (not all 1.0)
        if any(not np.all(h == 1.0) for h in hsi_arrays):
            for h in hsi_arrays:
                mask &= np.isfinite(h) & (h <= 2.0)

    tp9  = ch["RAW_TP9"][:L][mask]
    af7  = ch["RAW_AF7"][:L][mask]
    af8  = ch["RAW_AF8"][:L][mask]
    tp10 = ch["RAW_TP10"][:L][mask]

    if len(tp9) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    signal = np.stack([tp9, af7, af8, tp10], axis=1).astype(np.float32)
    signal -= signal.mean(axis=0, keepdims=True)          # mean-centre
    return signal


def _apply_baseline_reduction(signal: np.ndarray,
                               baseline: np.ndarray,
                               eps: float = 1e-12) -> np.ndarray:
    """
    InvBase: divide trial FFT by baseline FFT then iFFT back to time domain.
    Identical algorithm to the one used in eeg_data_loader.py.
    """
    FFT_trial    = np.fft.rfft(signal,   axis=0)
    FFT_baseline = np.fft.rfft(baseline, axis=0)
    FFT_reduced  = FFT_trial / (np.abs(FFT_baseline) + eps)
    return np.fft.irfft(FFT_reduced, n=len(signal), axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# BASELINE CACHE
# ──────────────────────────────────────────────────────────────────────────────

def _load_baselines(subject_dir: str) -> dict:
    """
    Load all NEUTRAL_*.csv files for one subject.

    Returns
    -------
    baselines : dict  {emotion_str -> np.ndarray (T, 4)}
        e.g. {"ANGER": array, "FEAR": array, …}
    """
    baselines = {}
    for neutral_stem, emotion in NEUTRAL_MAP.items():
        csv_path = os.path.join(subject_dir, f"{neutral_stem}.csv")
        if not os.path.exists(csv_path):
            continue
        sig = _read_raw_channels(csv_path)
        if sig.shape[0] > 0:
            baselines[emotion] = sig
    return baselines


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOADER  (public API – same 5-tuple signature as eeg_data_loader.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_eeg_data(data_root: str, config):
    """
    Load EEG data from the Emoky (EKM-ED) dataset.

    Parameters
    ----------
    data_root : str
        Path to the dataset root that **contains** the ``0.0078125S`` folder,
        i.e. the ``muse_wearable_data/preprocessed/clean-signals`` directory.
    config : Config
        Pipeline configuration object.  The following attributes are used:
            EEG_WINDOW_SEC, EEG_OVERLAP, USE_BASELINE_REDUCTION,
            CLIP_INDEPENDENT

    Returns
    -------
    X_raw       : np.ndarray  (N, T, 4)
    y_labels    : np.ndarray  (N,)  int64 class indices
    subject_ids : np.ndarray  (N,)  str
    trial_ids   : np.ndarray  (N,)  str  "<subject>_<emotion>"
    label_to_id : dict        {quadrant_str -> int}
    """
    print("\n" + "=" * 80)
    print("LOADING EEG DATA  –  Emoky / EKM-ED Dataset")
    print("=" * 80)

    # ── Locate timestep folder ─────────────────────────────────────────────
    ts_dir = os.path.join(data_root, EMOKY_TIMESTEP_FOLDER)
    if not os.path.isdir(ts_dir):
        # Accept data_root pointing directly at the timestep folder
        if os.path.basename(data_root) == EMOKY_TIMESTEP_FOLDER:
            ts_dir = data_root
        else:
            raise ValueError(
                f"Cannot find '{EMOKY_TIMESTEP_FOLDER}' under:\n  {data_root}\n"
                f"Set DATA_ROOT to the 'clean-signals' parent directory, "
                f"or directly to the '{EMOKY_TIMESTEP_FOLDER}' folder."
            )

    # Sampling rate for this dataset (overrides config.EEG_FS for windowing)
    fs = EMOKY_FS
    win_samples  = int(config.EEG_WINDOW_SEC * fs)
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))

    print(f"   Timestep folder : {ts_dir}")
    print(f"   Sampling rate   : {fs} Hz")
    print(f"   Window          : {config.EEG_WINDOW_SEC}s  ({win_samples} samples)")
    print(f"   Overlap         : {config.EEG_OVERLAP*100:.0f}%  (step {step_samples} samples)")
    print(f"   Baseline reduce : {config.USE_BASELINE_REDUCTION}")

    # ── Discover subject folders ───────────────────────────────────────────
    subject_dirs = sorted([
        d for d in glob.glob(os.path.join(ts_dir, "*"))
        if os.path.isdir(d)
    ])

    if not subject_dirs:
        raise ValueError(f"No subject sub-folders found under: {ts_dir}")

    print(f"\n   Found {len(subject_dirs)} subject folder(s): "
          f"{[os.path.basename(d) for d in subject_dirs[:6]]}"
          f"{'…' if len(subject_dirs) > 6 else ''}")

    # ── Accumulators ──────────────────────────────────────────────────────
    all_windows, all_labels, all_subjects, all_trials = [], [], [], []

    skipped = Counter()
    reduced_count = 0
    not_reduced_count = 0

    # ── Process each subject ──────────────────────────────────────────────
    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)

        # Load per-emotion baselines for this subject (once)
        baselines = {}
        if config.USE_BASELINE_REDUCTION:
            baselines = _load_baselines(subj_dir)
            if not baselines:
                print(f"   ⚠️  No NEUTRAL_*.csv found for subject {subject_id}")

        # Process each stimulus emotion CSV
        for emotion in STIMULUS_EMOTIONS:
            csv_path = os.path.join(subj_dir, f"{emotion}.csv")
            if not os.path.exists(csv_path):
                skipped["missing_csv"] += 1
                continue

            superclass = EMOKY_SUPERCLASS_MAP.get(emotion)
            if superclass is None:
                skipped["unknown_emotion"] += 1
                continue

            # ── Read raw channels ─────────────────────────────────────────
            signal = _read_raw_channels(csv_path)
            if signal.shape[0] == 0:
                skipped["no_data"] += 1
                continue

            # ── Baseline reduction ────────────────────────────────────────
            if config.USE_BASELINE_REDUCTION and emotion in baselines:
                baseline_sig = baselines[emotion]
                common_len = min(len(signal), len(baseline_sig))

                if common_len < win_samples:
                    skipped["too_short_after_baseline_match"] += 1
                    not_reduced_count += 1
                    continue

                signal   = _apply_baseline_reduction(
                    signal[:common_len], baseline_sig[:common_len]
                )
                reduced_count += 1
            else:
                not_reduced_count += 1

            L = len(signal)
            if L < win_samples:
                skipped["insufficient_length"] += 1
                continue

            # ── Windowing ─────────────────────────────────────────────────
            trial_id = f"{subject_id}_{emotion}"
            n_windows_added = 0
            for start in range(0, L - win_samples + 1, step_samples):
                window = signal[start:start + win_samples]
                if len(window) == win_samples:
                    all_windows.append(window)
                    all_labels.append(superclass)
                    all_subjects.append(subject_id)
                    all_trials.append(trial_id)
                    n_windows_added += 1

            if n_windows_added == 0:
                skipped["no_windows_extracted"] += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n📊 File Processing Summary:")
    print(f"   Subjects processed    : {len(subject_dirs)}")
    print(f"   Windows extracted     : {len(all_windows)}")
    if skipped:
        print(f"   Skipped (reasons)     :")
        for reason, count in skipped.items():
            print(f"      {reason}: {count}")

    if len(all_windows) == 0:
        raise ValueError(
            "No valid EEG windows extracted from the Emoky dataset.\n"
            f"  data_root   = {data_root}\n"
            f"  ts_dir      = {ts_dir}\n"
            f"  win_samples = {win_samples}  (EEG_WINDOW_SEC={config.EEG_WINDOW_SEC}, fs={fs})"
        )

    # ── Convert to arrays ─────────────────────────────────────────────────
    X_raw = np.stack(all_windows).astype(np.float32)

    unique_labels = sorted(set(all_labels))
    label_to_id   = {lab: i for i, lab in enumerate(unique_labels)}
    y_labels      = np.array([label_to_id[lb] for lb in all_labels], dtype=np.int64)
    subject_ids   = np.array(all_subjects)
    trial_ids     = np.array(all_trials)

    print(f"\n✅ Emoky EEG data loaded : {X_raw.shape}")
    print(f"   Label distribution   : {Counter(all_labels)}")
    print(f"   Unique subjects      : {len(np.unique(subject_ids))}")
    print(f"   Unique trials        : {len(np.unique(trial_ids))}")
    print(f"   Label map            : {label_to_id}")

    if config.USE_BASELINE_REDUCTION:
        total = reduced_count + not_reduced_count
        print(f"\n📊 Baseline Reduction:")
        print(f"   ✅ Reduced          : {reduced_count}")
        print(f"   ⚠️  Not reduced      : {not_reduced_count}")
        if total > 0:
            print(f"   📈 Reduction rate   : {100*reduced_count/total:.1f}%")

    return X_raw, y_labels, subject_ids, trial_ids, label_to_id


# ──────────────────────────────────────────────────────────────────────────────
# RE-EXPORT feature extractor and data splitter
# (so callers can do a single import from this module, matching the pattern
#  used for eeg_data_loader.py in EEGPipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

from eeg_data_loader import create_data_splits          # identical logic
from eeg_feature_extractor import extract_eeg_features  # identical logic
