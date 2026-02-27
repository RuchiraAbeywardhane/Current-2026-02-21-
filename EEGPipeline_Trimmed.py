"""
    EEG-Only Emotion Recognition Pipeline with Temporal Trimming
    =============================================================
    
    This script contains a complete EEG emotion recognition pipeline with:
    - Preprocessed EEG data loading (MUSE headband)
    - **TEMPORAL TRIMMING: Removes first 15s and last 15s from each recording**
    - Baseline reduction (InvBase method)
    - Feature extraction (26 features per channel)
    - BiLSTM classifier with attention
    - Subject-independent or subject-dependent splits
    
    KEY MODIFICATION:
    This pipeline removes the first 15 seconds and last 15 seconds from each
    EEG recording before preprocessing to eliminate potential artifacts from
    stimulus onset/offset periods, focusing analysis on the stable emotional
    response period.
    
    Author: Final Year Project
    Date: 2026-02-27
"""

import os
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import f1_score, classification_report

# Import configuration
from eeg_config import Config

# Import model architecture
from eeg_bilstm_model import SimpleBiLSTMClassifier

# Import data loading functions from TRIMMED module
from eeg_data_loader_trimmed import (
    load_eeg_data,
    create_data_splits
)

# Import feature extraction from standard module
from eeg_feature_extractor import extract_eeg_features

# Import training functions from separate module
from eeg_trainer import train_eeg_model


# Global config instance
config = Config()

# Set random seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

print(f"Device: {config.DEVICE}")


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """EEG-only emotion recognition pipeline with temporal trimming."""
    print("=" * 80)
    print("EEG-ONLY EMOTION RECOGNITION PIPELINE")
    print("WITH TEMPORAL TRIMMING (First 15s & Last 15s Removed)")
    print("=" * 80)
    print(f"Dataset: {config.DATA_ROOT}")
    print(f"Mode: {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Baseline Reduction: {config.USE_BASELINE_REDUCTION}")
    print(f"⏱️  Temporal Trimming: ENABLED (15s start + 15s end removed)")
    print("=" * 80)

    # Step 1: Load EEG data with temporal trimming
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(config.DATA_ROOT, config)
    
    # Step 2: Extract features
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    # Step 3: Create data splits
    print("\n" + "="*80)
    print("CREATING DATA SPLIT")
    print("="*80)
    
    split_indices = create_data_splits(eeg_y, eeg_subjects, config)
    
    print(f"\n📋 Split Summary:")
    print(f"   Train samples: {len(split_indices['train'])}")
    print(f"   Val samples: {len(split_indices['val'])}")
    print(f"   Test samples: {len(split_indices['test'])}")
    
    # Step 4: Train EEG model
    eeg_model, eeg_mu, eeg_sd = train_eeg_model(eeg_X_features, eeg_y, split_indices, eeg_label_map, config)
    
    print("\n" + "=" * 80)
    print("🎉 EEG PIPELINE WITH TEMPORAL TRIMMING COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {config.EEG_CHECKPOINT}")
    print(f"⏱️  Temporal Trimming: First 15s and Last 15s removed from all recordings")
    print(f"🎯 Focus: Stable emotional response period analyzed")
    print("=" * 80)

if __name__ == "__main__":
    main()
