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

from sklearn.metrics import f1_score, classification_report

# Import data loading utilities
from eeg_dataloader import load_eeg_data, extract_eeg_features, prepare_eeg_dataloaders

# Import model architecture
from eeg_model import SimpleBiLSTMClassifier

# Import training utilities
from eeg_trainer import train_eeg_model


# ==================================================
# CONFIGURATION
# ==================================================

class Config:
    """EEG-specific configuration."""
    # Paths
    DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"
    
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
        # Positive + High Arousal (Q1)
        "ENTHUSIASM": "Q1",
        "AMUSEMENT": "Q1",
        "JOY": "Q1",
        "EXCITEMENT": "Q1",
        
        # Negative + High Arousal (Q2)
        "FEAR": "Q2",
        "ANGER": "Q2",
        "DISGUST": "Q2",
        "ANXIETY": "Q2",
        
        # Negative + Low Arousal (Q3)
        "SADNESS": "Q3",
        "BOREDOM": "Q3",
        "DEPRESSION": "Q3",
        
        # Positive + Low Arousal (Q4)
        "NEUTRAL": "Q4",
        "CALMNESS": "Q4",
        "CONTENTMENT": "Q4",
        "RELAXATION": "Q4",
        "AWE": "Q4",
        "LIKING": "Q4",
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
    USE_MIXUP = CLIP_INDEPENDENT
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
    
    # Step 1: Load EEG data
    eeg_X_raw, eeg_y, eeg_subjects, eeg_label_map = load_eeg_data(config.DATA_ROOT, config)
    
    # Step 2: Extract features
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    # Step 3: Create data splits
    print("\n" + "="*80)
    print("CREATING DATA SPLIT")
    print("="*80)
    
    if config.SUBJECT_INDEPENDENT:
        print("  Strategy: SUBJECT-INDEPENDENT split")
        unique_subjects = np.unique(eeg_subjects)
        np.random.shuffle(unique_subjects)
        
        n_test = int(len(unique_subjects) * 0.15)
        n_val = int(len(unique_subjects) * 0.15)
        
        test_subjects = unique_subjects[:n_test]
        val_subjects = unique_subjects[n_test:n_test+n_val]
        train_subjects = unique_subjects[n_test+n_val:]
        
        train_mask = np.isin(eeg_subjects, train_subjects)
        val_mask = np.isin(eeg_subjects, val_subjects)
        test_mask = np.isin(eeg_subjects, test_subjects)
    else:
        print("  Strategy: RANDOM split")
        n_samples = len(eeg_y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_test = int(n_samples * 0.15)
        n_val = int(n_samples * 0.15)
        
        train_mask = np.zeros(n_samples, dtype=bool)
        val_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)
        
        test_mask[indices[:n_test]] = True
        val_mask[indices[n_test:n_test+n_val]] = True
        train_mask[indices[n_test+n_val:]] = True
    
    split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    print(f"\n📋 Split Summary:")
    print(f"   Train samples: {len(split_indices['train'])}")
    print(f"   Val samples: {len(split_indices['val'])}")
    print(f"   Test samples: {len(split_indices['test'])}")
    
    # Step 4: Prepare data loaders
    loaders, stats = prepare_eeg_dataloaders(eeg_X_features, eeg_y, split_indices, config)
    
    # Step 5: Train EEG model
    eeg_model, loaders, stats = train_eeg_model(loaders, eeg_label_map, config, stats)
    
    print("\n" + "=" * 80)
    print("🎉 EEG PIPELINE COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {config.EEG_CHECKPOINT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
