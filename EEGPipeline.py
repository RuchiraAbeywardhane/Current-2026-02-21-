"""
    EEG-Only Emotion Recognition Pipeline
    ======================================

    Supports two datasets – switch by changing DATASET below:

        DATASET = "emognition"   →  Emognition (JSON, 256 Hz, 4-class)
        DATASET = "emoky"        →  Emoky EKM-ED (CSV, 128 Hz, 3-class)

    Author: Final Year Project
    Date: 2026
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="EEG BiLSTM Emotion Recognition Pipeline")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["emognition", "emoky"],
    default="emoky",
    help='Dataset to use: "emognition" (JSON, 256Hz, 4-class) or "emoky" (CSV, 128Hz, 3-class)'
)
parser.add_argument(
    "--data_root",
    type=str,
    default=None,
    help="Path to the dataset root directory. Overrides the default path in config."
)
args = parser.parse_args()
DATASET = args.dataset
# ──────────────────────────────────────────────────────────────────────────────

if DATASET == "emoky":
    from eeg_config import EmokyConfig as ConfigClass
    from eeg_data_loader_emoky import (
        load_eeg_data,
        extract_eeg_features,
        create_data_splits,
    )
else:  # "emognition"  (default / fallback)
    from eeg_config import Config as ConfigClass
    from eeg_data_loader import (
        load_eeg_data,
        extract_eeg_features,
        create_data_splits,
    )

from eeg_bilstm_model import SimpleBiLSTMClassifier
from eeg_trainer import train_eeg_model

# Global config instance
config = ConfigClass()

# Override DATA_ROOT if provided via command line
if args.data_root:
    config.DATA_ROOT = args.data_root

# Set random seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

print(f"Dataset : {DATASET.upper()}")
print(f"Device  : {config.DEVICE}")


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """EEG-only emotion recognition pipeline."""
    print("=" * 80)
    print("EEG-ONLY EMOTION RECOGNITION PIPELINE")
    print("=" * 80)
    print(f"Dataset  : {DATASET.upper()}")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Mode     : {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Baseline : {config.USE_BASELINE_REDUCTION}")
    print(f"Classes  : {config.NUM_CLASSES}  {config.IDX_TO_LABEL}")
    print("=" * 80)

    # Step 1: Load EEG data
    eeg_X_raw, eeg_y, eeg_subjects, eeg_trial_ids, eeg_label_map = load_eeg_data(
        config.DATA_ROOT, config
    )

    # Step 2: Extract features
    # Pass the dataset-correct sampling rate so PSD / DE bands are accurate
    eeg_X_features = extract_eeg_features(eeg_X_raw, config, fs=config.EEG_FS)

    # Step 3: Create data splits
    split_indices = create_data_splits(
        eeg_y, eeg_subjects, config, trial_ids=eeg_trial_ids
    )

    # Step 4: Train EEG model
    eeg_model, eeg_mu, eeg_sd = train_eeg_model(
        eeg_X_features, eeg_y, split_indices, eeg_label_map, config
    )

    print("\n" + "=" * 80)
    print("🎉 EEG PIPELINE COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {config.EEG_CHECKPOINT}")
    print("=" * 80)


if __name__ == "__main__":
    main()