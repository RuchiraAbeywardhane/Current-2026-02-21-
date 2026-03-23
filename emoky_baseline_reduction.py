"""
Emoky Dataset – Baseline Reduction Script
==========================================

Reads the raw Emoky (EKM-ED) CSVs, applies per-subject per-emotion
baseline (neutral) reduction, and writes the processed signals out as
new CSVs in a mirror directory tree.

What "baseline reduction" means here
--------------------------------------
For each stimulus recording (ANGER.csv, FEAR.csv, …) there is a
matching neutral recording (NEUTRAL_ANGER.csv, NEUTRAL_FEAR.csv, …).

The InvBase algorithm (Zheng & Lu, 2015) is applied:

    signal_reduced = iFFT( FFT(trial) / |FFT(neutral)| )

This suppresses subject-specific resting-state spectral profiles so
that only the emotion-evoked deviation remains.

Dataset structure expected
--------------------------
    <DATA_ROOT>/
    └── 0.0078125S/
        ├── 1/
        │   ├── ANGER.csv
        │   ├── NEUTRAL_ANGER.csv
        │   ├── FEAR.csv
        │   ├── NEUTRAL_FEAR.csv
        │   ├── HAPPINESS.csv
        │   ├── NEUTRAL_HAPPINESS.csv
        │   ├── SADNESS.csv
        │   └── NEUTRAL_SADNESS.csv
        └── 103/
            └── …

Output structure (mirrored under OUT_ROOT)
------------------------------------------
    <OUT_ROOT>/
    └── 0.0078125S/
        ├── 1/
        │   ├── ANGER_baseline_reduced.csv
        │   ├── FEAR_baseline_reduced.csv
        │   ├── HAPPINESS_baseline_reduced.csv
        │   └── SADNESS_baseline_reduced.csv
        └── 103/
            └── …

Usage
-----
    python emoky_baseline_reduction.py

Set DATA_ROOT and OUT_ROOT at the top of this file before running.
On Kaggle set OUT_ROOT to "/kaggle/working/baseline_reduced".

Author: Final Year Project
Date  : 2026
"""

import os
import glob
from collections import Counter

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# ① USER CONFIGURATION  ← edit here
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = (
    "/kaggle/input/datasets/ruchiabey/emoky-dataset/"
    "EmoKey Moments EEG Dataset (EKM-ED)/"
    "muse_wearable_data/preprocessed/clean-signals"
)

# Where to write the reduced CSVs.
# On Kaggle use "/kaggle/working/baseline_reduced"
OUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "baseline_reduced"
)

# Set to False to keep ALL original columns in the output CSV.
# Set to True to write only the 4 RAW channels + TimeStamp (smaller files).
WRITE_RAW_ONLY = False

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

EMOKY_TIMESTEP_FOLDER = "0.0078125S"

RAW_COLS     = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
HSI_COLS     = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]
HEADBAND_COL = "HeadBandOn"

STIMULUS_EMOTIONS = ["ANGER", "FEAR", "HAPPINESS", "SADNESS"]

NEUTRAL_MAP = {
    "ANGER":     "NEUTRAL_ANGER",
    "FEAR":      "NEUTRAL_FEAR",
    "HAPPINESS": "NEUTRAL_HAPPINESS",
    "SADNESS":   "NEUTRAL_SADNESS",
}

# ──────────────────────────────────────────────────────────────────────────────
# PATH DISCOVERY  (same robust logic as eeg_data_loader_emoky.py)
# ──────────────────────────────────────────────────────────────────────────────

def _find_subject_root(data_root: str) -> str:
    """
    Locate the directory that directly contains per-subject sub-folders.
    Tries three strategies before raising a clear error.
    """
    # Strategy 1 – canonical layout: data_root/0.0078125S/
    ts_dir = os.path.join(data_root, EMOKY_TIMESTEP_FOLDER)
    if os.path.isdir(ts_dir):
        return ts_dir

    # Strategy 2 – data_root IS the timestep folder
    if os.path.basename(data_root) == EMOKY_TIMESTEP_FOLDER and os.path.isdir(data_root):
        return data_root

    # Strategy 3 – recursive walk
    stimulus_set = {f"{e}.csv" for e in STIMULUS_EMOTIONS}
    if os.path.isdir(data_root):
        print(f"  ⚠️  '{EMOKY_TIMESTEP_FOLDER}' not found directly — scanning recursively …")
        candidate_parents: set[str] = set()
        for root, dirs, files in os.walk(data_root):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if stimulus_set & {f.upper() for f in files}:
                candidate_parents.add(os.path.dirname(root))
        if candidate_parents:
            found = min(candidate_parents, key=lambda p: p.count(os.sep))
            print(f"  ✅ Auto-discovered subject root: {found}")
            return found

    raise ValueError(
        f"Cannot find Emoky subject folders under:\n  {data_root}\n"
        f"Expected sub-folders containing ANGER.csv / FEAR.csv etc."
    )

# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL I/O HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN/Inf in a 1-D float array."""
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _read_raw_channels(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a MUSE CSV and return:
        signal  : (T, 4) float32  – quality-filtered, mean-centred raw EEG
        mask    : (L,)   bool     – which original rows passed the quality filter

    Returns (zeros array, empty mask) on any failure.
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        print(f"      ⚠️  Cannot read {os.path.basename(csv_path)}: {exc}")
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=bool)

    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        print(f"      ⚠️  Missing columns {missing} in {os.path.basename(csv_path)}")
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=bool)

    ch = {c: _interp_nan(pd.to_numeric(df[c], errors="coerce").to_numpy()) for c in RAW_COLS}
    L = min(len(v) for v in ch.values())
    if L == 0:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=bool)

    mask = np.ones(L, dtype=bool)
    for c in RAW_COLS:
        mask &= np.isfinite(ch[c][:L])

    if HEADBAND_COL in df.columns:
        ho = pd.to_numeric(df[HEADBAND_COL], errors="coerce").to_numpy()[:L]
        if not np.all(ho == 1.0):
            mask &= (ho == 1.0)

    if all(c in df.columns for c in HSI_COLS):
        hsi = [pd.to_numeric(df[c], errors="coerce").to_numpy()[:L] for c in HSI_COLS]
        if any(not np.all(h == 1.0) for h in hsi):
            for h in hsi:
                mask &= np.isfinite(h) & (h <= 2.0)

    signal = np.stack([ch[c][:L][mask] for c in RAW_COLS], axis=1).astype(np.float32)
    if signal.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=bool)

    signal -= signal.mean(axis=0, keepdims=True)   # mean-centre
    return signal, mask


def _load_full_df(csv_path: str) -> pd.DataFrame | None:
    """Return the full DataFrame (all columns) or None on failure."""
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        print(f"      ⚠️  Cannot read {os.path.basename(csv_path)}: {exc}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# BASELINE REDUCTION CORE
# ──────────────────────────────────────────────────────────────────────────────

def apply_baseline_reduction(
    trial: np.ndarray,
    baseline: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    InvBase spectral baseline reduction.

    Parameters
    ----------
    trial    : (T, C) float32  – stimulus EEG (mean-centred)
    baseline : (T, C) float32  – neutral EEG  (same length, mean-centred)
    eps      : float           – numerical floor to avoid division by zero

    Returns
    -------
    reduced  : (T, C) float32  – baseline-suppressed EEG signal
    """
    FFT_trial    = np.fft.rfft(trial,    axis=0)   # (F, C)
    FFT_baseline = np.fft.rfft(baseline, axis=0)   # (F, C)
    FFT_reduced  = FFT_trial / (np.abs(FFT_baseline) + eps)
    reduced      = np.fft.irfft(FFT_reduced, n=len(trial), axis=0)
    return reduced.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# PER-SUBJECT PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def _process_subject(subj_dir: str, out_subj_dir: str, stats: Counter) -> None:
    """
    Run baseline reduction for all 4 emotions of one subject and write
    the results to out_subj_dir.
    """
    sid = os.path.basename(subj_dir)
    os.makedirs(out_subj_dir, exist_ok=True)

    for emotion in STIMULUS_EMOTIONS:
        trial_path   = os.path.join(subj_dir, f"{emotion}.csv")
        neutral_stem = NEUTRAL_MAP[emotion]
        neutral_path = os.path.join(subj_dir, f"{neutral_stem}.csv")

        # ── Check inputs exist ────────────────────────────────────────────
        if not os.path.exists(trial_path):
            print(f"    ⚠️  [{sid}] Missing {emotion}.csv — skipped")
            stats["missing_trial"] += 1
            continue

        if not os.path.exists(neutral_path):
            print(f"    ⚠️  [{sid}] Missing {neutral_stem}.csv — {emotion} not reduced")
            stats["missing_neutral"] += 1
            # Still write the unreduced signal so every emotion has an output
            _write_unreduced(trial_path, out_subj_dir, emotion, stats)
            continue

        # ── Read signals ──────────────────────────────────────────────────
        trial_sig,   trial_mask   = _read_raw_channels(trial_path)
        neutral_sig, _            = _read_raw_channels(neutral_path)

        if trial_sig.shape[0] == 0:
            print(f"    ⚠️  [{sid}] No usable data in {emotion}.csv — skipped")
            stats["no_data_trial"] += 1
            continue

        if neutral_sig.shape[0] == 0:
            print(f"    ⚠️  [{sid}] No usable data in {neutral_stem}.csv — {emotion} not reduced")
            stats["no_data_neutral"] += 1
            _write_unreduced(trial_path, out_subj_dir, emotion, stats)
            continue

        # ── Align lengths (truncate to shorter) ───────────────────────────
        common_len = min(len(trial_sig), len(neutral_sig))
        trial_sig   = trial_sig[:common_len]
        neutral_sig = neutral_sig[:common_len]

        # ── Apply InvBase ─────────────────────────────────────────────────
        reduced_sig = apply_baseline_reduction(trial_sig, neutral_sig)

        # ── Build output DataFrame ────────────────────────────────────────
        out_path = os.path.join(out_subj_dir, f"{emotion}_baseline_reduced.csv")

        if WRITE_RAW_ONLY:
            # Compact format: only the 4 reduced raw channels
            out_df = pd.DataFrame(
                reduced_sig, columns=RAW_COLS
            )
        else:
            # Full format: carry all original columns, overwrite RAW_* only
            full_df = _load_full_df(trial_path)
            if full_df is None:
                # Fallback to raw-only if full load fails
                out_df = pd.DataFrame(reduced_sig, columns=RAW_COLS)
            else:
                # Keep only the rows that passed the quality filter
                full_df = full_df.iloc[:len(trial_mask)][trial_mask[:len(full_df)]]
                full_df = full_df.iloc[:common_len].copy().reset_index(drop=True)
                for i, col in enumerate(RAW_COLS):
                    full_df[col] = reduced_sig[:, i]
                out_df = full_df

        out_df.to_csv(out_path, index=False)
        print(f"    ✅  [{sid}] {emotion:<12s}  "
              f"{common_len:6d} samples  →  {os.path.basename(out_path)}")
        stats["reduced"] += 1

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: write unreduced signal (neutral not available)
# ──────────────────────────────────────────────────────────────────────────────

def _write_unreduced(trial_path: str, out_subj_dir: str,
                     emotion: str, stats: Counter) -> None:
    """
    Copy the trial signal as-is (unreduced) to the output folder so every
    emotion always has a corresponding output file.
    """
    trial_sig, _ = _read_raw_channels(trial_path)
    if trial_sig.shape[0] == 0:
        stats["no_data_trial"] += 1
        return

    out_path = os.path.join(out_subj_dir, f"{emotion}_baseline_reduced.csv")

    if WRITE_RAW_ONLY:
        out_df = pd.DataFrame(trial_sig, columns=RAW_COLS)
    else:
        full_df = _load_full_df(trial_path)
        out_df = full_df if full_df is not None else pd.DataFrame(trial_sig, columns=RAW_COLS)

    out_df.to_csv(out_path, index=False)
    print(f"    ➡️   Wrote unreduced copy: {os.path.basename(out_path)}")
    stats["unreduced_copy"] += 1

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  Emoky EKM-ED  –  Baseline Reduction")
    print("=" * 70)
    print(f"  DATA_ROOT    : {DATA_ROOT}")
    print(f"  OUT_ROOT     : {OUT_ROOT}")
    print(f"  WRITE_RAW_ONLY : {WRITE_RAW_ONLY}")
    print("=" * 70 + "\n")

    # ── Locate subject root ───────────────────────────────────────────────
    ts_dir = _find_subject_root(DATA_ROOT)
    print(f"  Subject root : {ts_dir}\n")

    # ── Discover subject folders ──────────────────────────────────────────
    subject_dirs = sorted(
        [d for d in glob.glob(os.path.join(ts_dir, "*")) if os.path.isdir(d)],
        key=lambda p: (len(os.path.basename(p)), os.path.basename(p)),
    )

    if not subject_dirs:
        raise ValueError(f"No subject sub-folders found under: {ts_dir}")

    print(f"  Found {len(subject_dirs)} subject(s): "
          f"{[os.path.basename(d) for d in subject_dirs]}\n")

    # ── Mirror the timestep folder name in the output path ───────────────
    ts_folder_name = os.path.basename(ts_dir)   # "0.0078125S"
    out_ts_dir     = os.path.join(OUT_ROOT, ts_folder_name)

    # ── Process all subjects ──────────────────────────────────────────────
    stats: Counter = Counter()

    for subj_dir in subject_dirs:
        sid = os.path.basename(subj_dir)
        out_subj_dir = os.path.join(out_ts_dir, sid)
        print(f"  ── Subject {sid} ──")
        _process_subject(subj_dir, out_subj_dir, stats)
        print()

    # ── Final summary ─────────────────────────────────────────────────────
    total = stats["reduced"] + stats["unreduced_copy"]
    print("=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"  ✅ Reduced             : {stats['reduced']}")
    print(f"  ➡️  Unreduced copies    : {stats['unreduced_copy']}")
    print(f"  ⚠️  Missing trial CSV   : {stats['missing_trial']}")
    print(f"  ⚠️  Missing neutral CSV : {stats['missing_neutral']}")
    print(f"  ⚠️  No-data trial       : {stats['no_data_trial']}")
    print(f"  ⚠️  No-data neutral     : {stats['no_data_neutral']}")
    if total > 0:
        print(f"  📈 Reduction rate      : {100 * stats['reduced'] / total:.1f}%")
    print(f"\n  Output written to: {OUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
