"""
    Preprocessed EEG-Only Emotion Recognition (Standalone Test)
    ============================================================
    
    This is a 100% EXACT copy of the EEG training code from BR_WithPP_FourEmotionsEEG.py
    Used to verify that preprocessed EEG training works identically before adding fusion.
    
    Dataset: /kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR
    
    Author: Final Year Project
    Date: 2026
"""

import random
import numpy as np
import torch

# Import modular components
from config import Config
from data_loaders import load_preprocessed_eeg_data
from preprocessing import extract_eeg_features
from models_eeg import SimpleBiLSTMClassifier
from training import train_eeg_model


# ==================================================
# CONFIGURATION & INITIALIZATION
# ==================================================

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
    """EEG-only training pipeline."""
    print("=" * 80)
    print("PREPROCESSED EEG-ONLY EMOTION RECOGNITION")
    print("=" * 80)
    print(f"Dataset: {config.DATA_ROOT}")
    print(f"Mode: {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Baseline Reduction: {config.USE_BASELINE_REDUCTION}")
    print("=" * 80)
    
    # Step 1: Load data
    eeg_X_raw, eeg_y, eeg_subjects = load_preprocessed_eeg_data(config.DATA_ROOT, config)
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    # Step 2: Create splits
    print("\n" + "="*80)
    print("CREATING EEG DATA SPLIT")
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
    
    eeg_split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    print(f"\n📋 Split Summary:")
    print(f"   Train samples: {len(eeg_split_indices['train'])}")
    print(f"   Val samples: {len(eeg_split_indices['val'])}")
    print(f"   Test samples: {len(eeg_split_indices['test'])}")
    
    # Step 3: Train EEG model
    eeg_model, eeg_mu, eeg_sd = train_eeg_model(
        eeg_X_features, 
        eeg_y, 
        eeg_split_indices, 
        config.SUPERCLASS_ID,
        SimpleBiLSTMClassifier,
        config
    )
    
    print("\n" + "=" * 80)
    print("🎉 EEG TRAINING COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {config.EEG_CHECKPOINT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
