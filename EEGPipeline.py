"""
    EEG-Only Emotion Recognition Pipeline
    ======================================
    
    This script contains a complete EEG emotion recognition pipeline with:
    - Preprocessed EEG data loading (MUSE headband)
    - Baseline reduction (InvBase method)
    - Feature extraction (26 features per channel)
    - BiLSTM classifier with attention
    - Subject-independent or subject-dependent splits
    
    Author: Final Year Project
    Date: 2026
"""

import os
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import f1_score, classification_report

# Import model architecture
from eeg_bilstm_model import SimpleBiLSTMClassifier

# # Import data loading functions from separate module
from eeg_data_loader import (
    load_eeg_data,
    extract_eeg_features,
    create_data_splits
)

# Import training functions from separate module
from eeg_trainer import train_eeg_model

# ==================================================
# CONFIGURATION
# ==================================================

class Config:
    """EEG-specific configuration."""
    # Paths
    DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"
    # DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognition"
    # DATA_ROOT = "/kaggle/input/datasets/ruchiabey/asr-outputv2-0/ASR_output"
    
    # Common parameters
    NUM_CLASSES = 4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Baseline reduction (InvBase method)
    USE_BASELINE_REDUCTION = True
    
    # Data split mode
    SUBJECT_INDEPENDENT = True
    CLIP_INDEPENDENT = False
    
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


# Global config instance
config = Config()

# Set random seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

print(f"Device: {config.DEVICE}")


# ==================================================
# MODEL ARCHITECTURE
# ==================================================

# Model moved to eeg_bilstm_model.py


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """EEG-only emotion recognition pipeline."""
    print("=" * 80)
    print("EEG-ONLY EMOTION RECOGNITION PIPELINE")
    print("=" * 80)
    print(f"Dataset: {config.DATA_ROOT}")
    print(f"Mode: {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Baseline Reduction: {config.USE_BASELINE_REDUCTION}")
    print("=" * 80)
    
    # # Check the JSON structure before loading data
    # check_json_structure(config.DATA_ROOT, num_samples=3)

    # Step 1: Load EEG data
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
    print("🎉 EEG PIPELINE COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {config.EEG_CHECKPOINT}")
    print("=" * 80)

if __name__ == "__main__":
    main()