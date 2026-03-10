"""
Emoky Dataset – t-SNE Visualizer
==================================

Produces three families of t-SNE plots from the EKM-ED dataset:

  1. EMOTION-WISE  (all subjects pooled)
     → One scatter plot, each point coloured by emotion label
        (ANGER / FEAR / HAPPINESS / SADNESS)

  2. SUBJECT-WISE  (all emotions pooled)
     → One scatter plot, each point coloured by subject ID

  3. PER-SUBJECT EMOTION BREAKDOWN
     → One subplot grid (one panel per subject),
       each panel coloured by emotion label for that subject only

Features are the same 26-per-channel handcrafted features used by the
training pipeline (extracted via eeg_feature_extractor.py), so the
t-SNE embedding directly reflects the input space the model sees.

Usage
-----
    python emoky_tsne_visualizer.py

Configure DATA_ROOT at the top of the script before running.

Output
------
All figures are saved as high-resolution PNGs in an ``tsne_outputs/``
sub-folder next to this script, AND shown interactively.

Author: Final Year Project
Date: 2026
"""

import os
import glob
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 11})

# ──────────────────────────────────────────────────────────────────────────────
# ① USER CONFIGURATION  ← edit here
# ──────────────────────────────────────────────────────────────────────────────

# Path to the 'clean-signals' folder (parent of '0.0078125S')
# On Kaggle: "/kaggle/input/.../clean-signals"
DATA_ROOT = "/kaggle/input/datasets/emoky/EmoKey Moments EEG Dataset (EKM-ED)/muse_wearable_data/preprocessed/clean-signals"
# t-SNE hyper-parameters
TSNE_PERPLEXITY  = 30      # typical range 5–50
TSNE_N_ITER      = 1000
TSNE_RANDOM_SEED = 42

# EEG windowing (must match EmokyConfig)
EEG_FS          = 128.0
EEG_WINDOW_SEC  = 10.0
EEG_OVERLAP     = 0.5      # 50 % overlap

# Output folder
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsne_outputs")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS (mirror eeg_data_loader_emoky.py)
# ──────────────────────────────────────────────────────────────────────────────

EMOKY_TIMESTEP_FOLDER = "0.0078125S"
RAW_COLS     = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
HSI_COLS     = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]
HEADBAND_COL = "HeadBandOn"
NEUTRAL_MAP  = {
    "NEUTRAL_ANGER":     "ANGER",
    "NEUTRAL_FEAR":      "FEAR",
    "NEUTRAL_HAPPINESS": "HAPPINESS",
    "NEUTRAL_SADNESS":   "SADNESS",
}
STIMULUS_EMOTIONS = ["ANGER", "FEAR", "HAPPINESS", "SADNESS"]

# Consistent colour palette – one colour per raw emotion label
EMOTION_COLORS = {
    "ANGER":     "#E63946",   # vivid red
    "FEAR":      "#F4A261",   # orange
    "HAPPINESS": "#2A9D8F",   # teal-green
    "SADNESS":   "#457B9D",   # steel blue
}

# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL HELPERS  (identical to eeg_data_loader_emoky.py)
# ──────────────────────────────────────────────────────────────────────────────

def _interp_nan(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():   return a
    if not m.any(): return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _read_raw_channels(csv_path: str) -> np.ndarray:
    """Return (T, 4) float32 array or (0,4) on failure."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"  ⚠️  Cannot read {csv_path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)

    if any(c not in df.columns for c in RAW_COLS):
        return np.zeros((0, 4), dtype=np.float32)

    ch = {c: _interp_nan(pd.to_numeric(df[c], errors="coerce").to_numpy()) for c in RAW_COLS}
    L  = min(len(v) for v in ch.values())
    if L == 0:
        return np.zeros((0, 4), dtype=np.float32)

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

    cols = np.stack([ch[c][:L][mask] for c in RAW_COLS], axis=1).astype(np.float32)
    if cols.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cols -= cols.mean(axis=0, keepdims=True)
    return cols


def _apply_baseline_reduction(signal, baseline, eps=1e-12):
    FFT_s = np.fft.rfft(signal,   axis=0)
    FFT_b = np.fft.rfft(baseline, axis=0)
    return np.fft.irfft(FFT_s / (np.abs(FFT_b) + eps), n=len(signal), axis=0).astype(np.float32)


def _load_baselines(subject_dir: str) -> dict:
    baselines = {}
    for neutral_stem, emotion in NEUTRAL_MAP.items():
        path = os.path.join(subject_dir, f"{neutral_stem}.csv")
        if not os.path.exists(path):
            continue
        sig = _read_raw_channels(path)
        if sig.shape[0] > 0:
            baselines[emotion] = sig
    return baselines


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (26 per channel, same as eeg_feature_extractor.py)
# ──────────────────────────────────────────────────────────────────────────────

BANDS = [("delta", (1, 3)), ("theta", (4, 7)), ("alpha", (8, 13)),
         ("beta", (14, 30)), ("gamma", (31, 45))]


def _extract_window_features(windows: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Extract 26 features per channel from a batch of windows.

    Parameters
    ----------
    windows : (N, T, C)
    fs      : sampling frequency

    Returns
    -------
    features : (N, C*26)  — flattened for t-SNE
    """
    from scipy.stats import skew, kurtosis as kurt

    N, T, C = windows.shape
    eps = 1e-12

    P     = (np.abs(np.fft.rfft(windows, axis=1)) ** 2) / T
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    feats = []

    # 1) Differential Entropy (5)
    de_all = []
    for _, (lo, hi) in BANDS:
        m  = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        de_all.append(0.5 * np.log(2 * np.pi * np.e * (bp + eps))[..., None])
    de_all = np.concatenate(de_all, axis=2)   # (N, C, 5)
    feats.append(de_all)

    # 2) Log-PSD (5)
    psd_all = []
    for _, (lo, hi) in BANDS:
        m  = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        psd_all.append(np.log(bp + eps)[..., None])
    feats.append(np.concatenate(psd_all, axis=2))

    # 3) Temporal stats (4)
    feats.append(windows.mean(axis=1)[..., None])
    feats.append(windows.std(axis=1)[..., None])
    feats.append(skew(windows, axis=1)[..., None])
    feats.append(kurt(windows, axis=1)[..., None])

    # 4) DE asymmetry (5)  left=(0,1), right=(2,3)
    de_left  = (de_all[:, 0, :] + de_all[:, 1, :]) / 2
    de_right = (de_all[:, 2, :] + de_all[:, 3, :]) / 2
    de_asym  = de_left - de_right
    feats.append(np.tile(de_asym[:, None, :], (1, C, 1)))

    # 5) Bandpower ratios (3)
    band_bp = []
    for _, (lo, hi) in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        band_bp.append(P[:, m, :].mean(axis=1))
    _, theta, alpha, beta, gamma = band_bp
    feats.append(np.stack([
        (theta + eps) / (alpha + eps),
        (beta  + eps) / (alpha + eps),
        (gamma + eps) / (beta  + eps),
    ], axis=2))

    # 6) Hjorth parameters (2)
    Xc    = windows - windows.mean(axis=1, keepdims=True)
    dx    = np.diff(Xc, axis=1)
    ddx   = np.diff(dx,  axis=1)
    var_x   = (Xc  ** 2).mean(axis=1) + eps
    var_dx  = (dx  ** 2).mean(axis=1) + eps
    var_ddx = (ddx ** 2).mean(axis=1) + eps
    mob   = np.sqrt(var_dx  / var_x)
    comp  = np.sqrt(var_ddx / var_dx) / (mob + eps)
    feats.append(np.stack([mob, comp], axis=2))

    # 7) Time-domain extras (2)
    log_var = np.log(var_x + eps)
    zc      = (np.diff(np.sign(Xc), axis=1) != 0).sum(axis=1) / float(T - 1 + eps)
    feats.append(np.stack([log_var, zc], axis=2))

    # Concatenate → (N, C, 26), then flatten → (N, C*26)
    feat_array = np.concatenate(feats, axis=2)   # (N, C, 26)
    return feat_array.reshape(N, -1).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (raw emotion labels, not quadrants)
# ──────────────────────────────────────────────────────────────────────────────

def load_emoky_raw(data_root: str):
    """
    Load windowed features from the Emoky dataset keeping the raw emotion label
    (ANGER / FEAR / HAPPINESS / SADNESS) instead of the quadrant mapping.

    Returns
    -------
    X          : (N, 104)  standardised features
    emotions   : (N,)      str  raw emotion name
    subjects   : (N,)      str  subject folder name
    """
    ts_dir = os.path.join(data_root, EMOKY_TIMESTEP_FOLDER)
    if not os.path.isdir(ts_dir):
        if os.path.basename(data_root) == EMOKY_TIMESTEP_FOLDER:
            ts_dir = data_root
        else:
            raise ValueError(
                f"Cannot find '{EMOKY_TIMESTEP_FOLDER}' under:\n  {data_root}"
            )

    win_samples  = int(EEG_WINDOW_SEC * EEG_FS)
    step_samples = int(win_samples * (1.0 - EEG_OVERLAP))

    subject_dirs = sorted(
        [d for d in glob.glob(os.path.join(ts_dir, "*")) if os.path.isdir(d)],
        key=lambda p: (len(os.path.basename(p)), os.path.basename(p))
    )

    if not subject_dirs:
        raise ValueError(f"No subject folders found under: {ts_dir}")

    print(f"Found {len(subject_dirs)} subject(s): "
          f"{[os.path.basename(d) for d in subject_dirs]}")

    all_windows, all_emotions, all_subjects = [], [], []

    for subj_dir in subject_dirs:
        sid       = os.path.basename(subj_dir)
        baselines = _load_baselines(subj_dir)

        for emotion in STIMULUS_EMOTIONS:
            csv_path = os.path.join(subj_dir, f"{emotion}.csv")
            if not os.path.exists(csv_path):
                print(f"  ⚠️  Missing: {csv_path}")
                continue

            signal = _read_raw_channels(csv_path)
            if signal.shape[0] == 0:
                print(f"  ⚠️  No usable data: {csv_path}")
                continue

            # Baseline reduction when available
            if emotion in baselines:
                bl  = baselines[emotion]
                cln = min(len(signal), len(bl))
                if cln >= win_samples:
                    signal = _apply_baseline_reduction(signal[:cln], bl[:cln])

            L = len(signal)
            if L < win_samples:
                print(f"  ⚠️  Too short after processing: {sid}/{emotion}")
                continue

            n_added = 0
            for start in range(0, L - win_samples + 1, step_samples):
                w = signal[start:start + win_samples]
                if len(w) == win_samples:
                    all_windows.append(w)
                    all_emotions.append(emotion)
                    all_subjects.append(sid)
                    n_added += 1

            print(f"  ✅  Subject {sid:>4s}  {emotion:<12s}  → {n_added:3d} windows")

    if not all_windows:
        raise ValueError("No windows extracted. Check DATA_ROOT.")

    print(f"\nTotal windows : {len(all_windows)}")
    print(f"Emotion dist  : {Counter(all_emotions)}")
    print(f"Subject dist  : {Counter(all_subjects)}\n")

    # Extract features
    print("Extracting features …")
    X_raw  = np.stack(all_windows).astype(np.float32)
    X_feat = _extract_window_features(X_raw, fs=EEG_FS)

    # Standardise
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_feat)

    return X, np.array(all_emotions), np.array(all_subjects)


# ──────────────────────────────────────────────────────────────────────────────
# T-SNE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_tsne(X: np.ndarray, tag: str = "") -> np.ndarray:
    n = len(X)
    perp = min(TSNE_PERPLEXITY, max(5, n // 4))
    print(f"  Running t-SNE on {n} samples (perplexity={perp}) …", end=" ", flush=True)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        n_iter=TSNE_N_ITER,
        random_state=TSNE_RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(X)
    print("done.")
    return emb


# ──────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _legend_patches(label_color_map: dict) -> list:
    return [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in label_color_map.items()
    ]


def _save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  💾  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 – EMOTION-WISE  (all subjects pooled)
# ──────────────────────────────────────────────────────────────────────────────

def plot_emotion_wise(X: np.ndarray, emotions: np.ndarray):
    print("\n── Plot 1 : Emotion-wise (all subjects) ──")
    emb = run_tsne(X, tag="emotion_all")

    fig, ax = plt.subplots(figsize=(9, 7))
    for emotion in STIMULUS_EMOTIONS:
        idx = emotions == emotion
        if not idx.any():
            continue
        ax.scatter(
            emb[idx, 0], emb[idx, 1],
            c=EMOTION_COLORS[emotion],
            label=emotion,
            s=18, alpha=0.65, linewidths=0,
        )

    ax.set_title("t-SNE – All Subjects, Coloured by Emotion", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1");  ax.set_ylabel("t-SNE dim 2")
    ax.legend(handles=_legend_patches(EMOTION_COLORS),
              title="Emotion", loc="best", framealpha=0.85)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    _save(fig, "01_tsne_emotion_wise_all_subjects")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 – SUBJECT-WISE  (all emotions pooled)
# ──────────────────────────────────────────────────────────────────────────────

def plot_subject_wise(X: np.ndarray, subjects: np.ndarray):
    print("\n── Plot 2 : Subject-wise (all emotions) ──")
    emb = run_tsne(X, tag="subject_all")

    unique_subjects = sorted(set(subjects),
                             key=lambda s: (len(s), s))
    n_subj  = len(unique_subjects)
    cmap    = plt.cm.get_cmap("tab20", n_subj)
    subj_colors = {s: cmap(i) for i, s in enumerate(unique_subjects)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for sid in unique_subjects:
        idx = subjects == sid
        ax.scatter(
            emb[idx, 0], emb[idx, 1],
            color=subj_colors[sid],
            label=f"S{sid}",
            s=18, alpha=0.65, linewidths=0,
        )

    ax.set_title("t-SNE – All Emotions, Coloured by Subject", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1");  ax.set_ylabel("t-SNE dim 2")
    ncol = max(1, n_subj // 15 + 1)
    ax.legend(title="Subject", loc="best", framealpha=0.85,
              fontsize=8, ncol=ncol, markerscale=1.5)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    _save(fig, "02_tsne_subject_wise_all_emotions")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3 – PER-SUBJECT EMOTION BREAKDOWN
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_subject_emotions(X: np.ndarray, emotions: np.ndarray, subjects: np.ndarray):
    """
    One subplot per subject.  Each panel runs its OWN t-SNE on that subject's
    windows only, then colours points by emotion.  This shows intra-subject
    separability cleanly without cross-subject variance dominating.
    """
    print("\n── Plot 3 : Per-subject emotion breakdown ──")

    unique_subjects = sorted(set(subjects), key=lambda s: (len(s), s))
    n_subj = len(unique_subjects)

    # Grid layout: try to keep it roughly square
    n_cols = min(4, n_subj)
    n_rows = int(np.ceil(n_subj / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle("t-SNE per Subject – Coloured by Emotion",
                 fontsize=16, fontweight="bold", y=1.01)

    for ax_idx, sid in enumerate(unique_subjects):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]

        mask = subjects == sid
        X_s  = X[mask]
        E_s  = emotions[mask]

        if len(X_s) < 6:
            ax.text(0.5, 0.5, f"Subject {sid}\n(too few samples)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"Subject {sid}", fontsize=10)
            ax.axis("off")
            continue

        emb = run_tsne(X_s, tag=f"subj_{sid}")

        for emotion in STIMULUS_EMOTIONS:
            eidx = E_s == emotion
            if not eidx.any():
                continue
            ax.scatter(
                emb[eidx, 0], emb[eidx, 1],
                c=EMOTION_COLORS[emotion],
                label=emotion,
                s=22, alpha=0.70, linewidths=0,
            )

        present = sorted({e for e in E_s if e in EMOTION_COLORS})
        ax.legend(
            handles=[mpatches.Patch(color=EMOTION_COLORS[e], label=e) for e in present],
            fontsize=7, loc="best", framealpha=0.75, markerscale=1.2,
        )
        win_counts = Counter(E_s)
        subtitle   = "  ".join(f"{e[0]}:{win_counts[e]}" for e in STIMULUS_EMOTIONS if e in win_counts)
        ax.set_title(f"Subject {sid}\n({subtitle})", fontsize=9)
        ax.set_xlabel("dim 1", fontsize=8);  ax.set_ylabel("dim 2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for ax_idx in range(n_subj, n_rows * n_cols):
        row, col = divmod(ax_idx, n_cols)
        axes[row][col].axis("off")

    # Shared legend at figure level
    legend_handles = _legend_patches(EMOTION_COLORS)
    fig.legend(
        handles=legend_handles,
        title="Emotion",
        loc="lower center",
        ncol=len(EMOTION_COLORS),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    _save(fig, "03_tsne_per_subject_emotion_breakdown")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# BONUS PLOT 4 – COMBINED: emotion colour + subject shape
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined(X: np.ndarray, emotions: np.ndarray, subjects: np.ndarray):
    """
    Single global t-SNE.  Colour = emotion, marker shape = subject.
    Useful when there are ≤ 6 subjects (marker variety stays readable).
    """
    print("\n── Plot 4 : Combined (emotion colour + subject marker) ──")

    unique_subjects = sorted(set(subjects), key=lambda s: (len(s), s))
    n_subj = len(unique_subjects)

    MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h",
               "*", "8", "p", "H", "+", "x", "d", "|", "_", "1"]
    subj_marker = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(unique_subjects)}

    emb = run_tsne(X, tag="combined")

    fig, ax = plt.subplots(figsize=(11, 8))

    for sid in unique_subjects:
        for emotion in STIMULUS_EMOTIONS:
            idx = (subjects == sid) & (emotions == emotion)
            if not idx.any():
                continue
            ax.scatter(
                emb[idx, 0], emb[idx, 1],
                c=EMOTION_COLORS[emotion],
                marker=subj_marker[sid],
                s=30, alpha=0.65, linewidths=0.2, edgecolors="none",
            )

    ax.set_title("t-SNE – Emotion (colour) × Subject (marker)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1");  ax.set_ylabel("t-SNE dim 2")

    # Two separate legends
    emotion_leg = ax.legend(
        handles=_legend_patches(EMOTION_COLORS),
        title="Emotion", loc="upper left", framealpha=0.85,
    )
    ax.add_artist(emotion_leg)

    subj_handles = [
        plt.scatter([], [], marker=subj_marker[s], color="grey",
                    s=40, label=f"S{s}")
        for s in unique_subjects
    ]
    ax.legend(handles=subj_handles, title="Subject",
              loc="upper right", fontsize=8, framealpha=0.85,
              ncol=max(1, n_subj // 8 + 1))

    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    _save(fig, "04_tsne_combined_emotion_subject")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Emoky EKM-ED  –  t-SNE Visualiser")
    print("=" * 70)
    print(f"  DATA_ROOT : {DATA_ROOT}")
    print(f"  OUT_DIR   : {OUT_DIR}")
    print("=" * 70)

    # ── Load & featurise ──────────────────────────────────────────────────
    X, emotions, subjects = load_emoky_raw(DATA_ROOT)
    print(f"\nFeature matrix : {X.shape}  (samples × features)")

    # ── Run all plots ─────────────────────────────────────────────────────
    plot_emotion_wise(X, emotions)
    plot_subject_wise(X, subjects)
    plot_per_subject_emotions(X, emotions, subjects)

    n_subj = len(set(subjects))
    if n_subj <= 12:           # combined plot stays readable up to ~12 subjects
        plot_combined(X, emotions, subjects)
    else:
        print(f"\nSkipping combined plot (>{12} subjects → too cluttered).")

    print("\n" + "=" * 70)
    print(f"  ✅  All plots saved to:  {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
