"""
EEG Pipeline Configuration
==========================

This module contains all configuration parameters for the EEG emotion recognition pipeline.

Author: Final Year Project
Date: 2026
"""

import torch

class Config:
    """EEG-specific configuration."""
    # Paths
    DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"
    # DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognition"
    # DATA_ROOT = "/kaggle/input/datasets/ruchiabey/asr-outputv2-0/ASR_output"
    
    # Common parameters
    NUM_CLASSES = 4
    SEED = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Baseline reduction (InvBase method)
    USE_BASELINE_REDUCTION = True  # For EEG
    
    # Data split mode
    SUBJECT_INDEPENDENT = False
    CLIP_INDEPENDENT = True
    
    # Stratified split parameters
    USE_STRATIFIED_GROUP_SPLIT = True
    MIN_SAMPLES_PER_CLASS = 10
    
    # Label mappings (4-class emotion quadrants)
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",  # Positive + High Arousal
        "FEAR": "Q2",         # Negative + High Arousal
        "SADNESS": "Q3",      # Negative + Low Arousal
        "NEUTRAL": "Q4",      # Positive + Low Arousal
    }
    
    SUPERCLASS_ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    IDX_TO_LABEL = ["Q1_Positive_Active", "Q2_Negative_Active", 
                    "Q3_Negative_Calm", "Q4_Positive_Calm"]
    
    # EEG parameters
    EEG_FS = 256.0
    EEG_CHANNELS = 4  # TP9, AF7, AF8, TP10
    EEG_FEATURES = 26
    EEG_WINDOW_SEC = 10.0
    EEG_OVERLAP = 0.5 if CLIP_INDEPENDENT else 0.0
    EEG_BATCH_SIZE = 32 if CLIP_INDEPENDENT else 64
    EEG_EPOCHS = 200 if CLIP_INDEPENDENT else 150
    EEG_LR = 5e-4 if CLIP_INDEPENDENT else 1e-3
    EEG_PATIENCE = 30 if CLIP_INDEPENDENT else 20
    EEG_CHECKPOINT = "best_eeg_model.pt"
    
    # Augmentation settings
    USE_MIXUP = True  # Set to True to enable Mixup data augmentation
    MIXUP_ALPHA = 0.2  # Mixup interpolation strength (only used if USE_MIXUP=True)
    LABEL_SMOOTHING = 0.1 if CLIP_INDEPENDENT else 0.0
    
    # Frequency bands for feature extraction
    BANDS = [("delta", (1, 3)), ("theta", (4, 7)), ("alpha", (8, 13)), 
             ("beta", (14, 30)), ("gamma", (31, 45))]

class EmokyConfig:
    """
    EEG configuration for the Emoky / EKM-ED dataset.

    Dataset path must point to the 'clean-signals' directory that contains
    the '0.0078125S' timestep folder:

        <DATA_ROOT>/
        └── 0.0078125S/
            ├── 1/
            │   ├── ANGER.csv
            │   ├── FEAR.csv
            │   ├── HAPPINESS.csv
            │   ├── NEUTRAL_ANGER.csv
            │   ├── NEUTRAL_FEAR.csv
            │   ├── NEUTRAL_HAPPINESS.csv
            │   ├── NEUTRAL_SADNESS.csv
            │   └── SADNESS.csv
            └── 103/

    Emotion → quadrant mapping:
        HAPPINESS → Q1  (Positive + High Arousal)
        FEAR      → Q2  (Negative + High Arousal)
        ANGER     → Q2  (Negative + High Arousal)
        SADNESS   → Q3  (Negative + Low Arousal)

    Only 3 quadrants are present in this dataset (Q1, Q2, Q3).
    """

    # ── Path ──────────────────────────────────────────────────────────────────
    # Set to the 'clean-signals' folder (parent of '0.0078125S')
    DATA_ROOT = "/kaggle/input/datasets/emoky/EmoKey Moments EEG Dataset (EKM-ED)/muse_wearable_data/preprocessed/clean-signals"

    # ── Common parameters ─────────────────────────────────────────────────────
    NUM_CLASSES = 3   # Q1, Q2, Q3  (no Q4 in this dataset)
    SEED        = 0
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Baseline reduction ────────────────────────────────────────────────────
    # NEUTRAL_<EMOTION>.csv acts as the per-emotion, per-subject baseline
    USE_BASELINE_REDUCTION = True

    # ── Data split mode ───────────────────────────────────────────────────────
    SUBJECT_INDEPENDENT   = False
    CLIP_INDEPENDENT      = True

    # ── Stratified split parameters ───────────────────────────────────────────
    USE_STRATIFIED_GROUP_SPLIT = True
    MIN_SAMPLES_PER_CLASS      = 10

    # ── Label mappings ────────────────────────────────────────────────────────
    SUPERCLASS_MAP = {
        "HAPPINESS": "Q1",   # Positive + High Arousal
        "FEAR":      "Q2",   # Negative + High Arousal
        "ANGER":     "Q2",   # Negative + High Arousal
        "SADNESS":   "Q3",   # Negative + Low Arousal
    }

    SUPERCLASS_ID  = {"Q1": 0, "Q2": 1, "Q3": 2}
    IDX_TO_LABEL   = ["Q1_Positive_Active", "Q2_Negative_Active", "Q3_Negative_Calm"]

    # ── EEG parameters ────────────────────────────────────────────────────────
    # NOTE: The Emoky loader uses EMOKY_FS = 128 Hz internally for windowing.
    #       EEG_FS here is kept at 128 so feature extraction (PSD, DE, Hjorth)
    #       uses the correct sampling rate.
    EEG_FS       = 128.0
    EEG_CHANNELS = 4          # TP9, AF7, AF8, TP10
    EEG_FEATURES = 26
    EEG_WINDOW_SEC = 10.0
    EEG_OVERLAP    = 0.5 if CLIP_INDEPENDENT else 0.0

    EEG_BATCH_SIZE = 32  if CLIP_INDEPENDENT else 64
    EEG_EPOCHS     = 200 if CLIP_INDEPENDENT else 150
    EEG_LR         = 5e-4 if CLIP_INDEPENDENT else 1e-3
    EEG_PATIENCE   = 30  if CLIP_INDEPENDENT else 20
    EEG_CHECKPOINT = "best_eeg_model_emoky.pt"

    # ── Augmentation settings ─────────────────────────────────────────────────
    USE_MIXUP       = True
    MIXUP_ALPHA     = 0.2
    LABEL_SMOOTHING = 0.1 if CLIP_INDEPENDENT else 0.0

    # ── Frequency bands for feature extraction ────────────────────────────────
    # Gamma upper-limit lowered to 45 Hz (Nyquist for 128 Hz is 64 Hz, safe)
    BANDS = [("delta", (1, 3)), ("theta", (4, 7)), ("alpha", (8, 13)),
             ("beta", (14, 30)), ("gamma", (31, 45))]
