"""
Configuration Module
====================
Centralized configuration for all training scripts.

Usage:
    from config import Config
    config = Config()
"""

import torch
import random
import numpy as np


class Config:
    """Shared configuration for all models and training scripts."""
    
    # ==================== PATHS ====================
    # HYBRID DATASETS
    EEG_DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"  # Preprocessed EEG
    BVP_DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognition"  # Raw BVP
    
    # ==================== DEVICE ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    # ==================== CLASSIFICATION MODE ====================
    NUM_CLASSES = 4
    USE_DUAL_BINARY = False  # If True, train separate arousal/valence classifiers
    CLASSIFY_WHOLE_CLIPS = False  # If True, aggregate window predictions per clip
    CLIP_AGGREGATION_METHOD = "mean"  # Options: "mean", "max", "voting"
    
    # ==================== DATA SPLIT ====================
    SUBJECT_INDEPENDENT = True  # Subject-independent split
    CLIP_INDEPENDENT = False  # Clip-independent (with overlap)
    USE_LOSO = False  # Leave-One-Subject-Out cross-validation
    LOSO_SAVE_ALL_FOLDS = True
    
    # Stratified split parameters
    USE_STRATIFIED_GROUP_SPLIT = True
    MIN_SAMPLES_PER_CLASS = 10
    
    # ==================== BASELINE REDUCTION ====================
    USE_BASELINE_REDUCTION = True  # InvBase method for EEG
    
    # ==================== LABEL MAPPINGS ====================
    # 4-class emotion system (Quadrants)
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",  # Positive-Active
        "FEAR": "Q2",         # Negative-Active
        "SADNESS": "Q3",      # Negative-Calm
        "NEUTRAL": "Q4",      # Positive-Calm
    }
    
    SUPERCLASS_ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    IDX_TO_LABEL = [
        "Q1_Positive_Active", 
        "Q2_Negative_Active", 
        "Q3_Negative_Calm", 
        "Q4_Positive_Calm"
    ]
    
    # Dual Binary Classification Mappings (Arousal/Valence)
    AROUSAL_MAP = {0: 1, 1: 1, 2: 0, 3: 0}  # High=1, Low=0
    VALENCE_MAP = {0: 1, 1: 0, 2: 0, 3: 1}  # Positive=1, Negative=0
    AROUSAL_LABELS = ["Low_Arousal", "High_Arousal"]
    VALENCE_LABELS = ["Negative_Valence", "Positive_Valence"]
    
    # ==================== EEG PARAMETERS ====================
    EEG_FS = 256.0  # Sampling frequency (Hz)
    EEG_CHANNELS = 4  # TP9, AF7, AF8, TP10
    EEG_FEATURES = 24  # Features per channel (8 time + 5 PSD + 5 DE + 6 temporal)
    EEG_WINDOW_SEC = 10.0  # Window length (seconds)
    EEG_OVERLAP = 0.5 if CLIP_INDEPENDENT else 0.0
    
    # EEG Training
    EEG_BATCH_SIZE = 32 if CLIP_INDEPENDENT else 64
    EEG_EPOCHS = 200 if CLIP_INDEPENDENT else 150
    EEG_LR = 5e-4 if CLIP_INDEPENDENT else 1e-3
    EEG_PATIENCE = 30 if CLIP_INDEPENDENT else 20
    EEG_CHECKPOINT = "best_eeg_model.pt"
    
    # EEG Model Architecture
    EEG_HIDDEN = 256
    EEG_LSTM_LAYERS = 3
    EEG_DROPOUT = 0.4
    
    # ==================== BVP PARAMETERS ====================
    BVP_FS = 64  # Sampling frequency (Hz)
    BVP_WINDOW_SEC = 10  # Window length (seconds)
    BVP_WINDOW_SIZE = BVP_FS * BVP_WINDOW_SEC
    BVP_FEATURES = 5  # Handcrafted features
    BVP_OVERLAP = 0.0
    
    # BVP Training
    BVP_BATCH_SIZE = 128
    BVP_EPOCHS = 50
    BVP_LR = 1e-3
    BVP_PATIENCE = 10
    BVP_CHECKPOINT = "best_bvp_model.pt"
    
    # ==================== FUSION PARAMETERS ====================
    FUSION_SHARED_DIM = 128  # Shared embedding dimension
    FUSION_NUM_HEADS = 4  # Multi-head attention heads
    FUSION_DROPOUT = 0.1
    
    # Fusion Training
    FUSION_BATCH_SIZE = 64
    FUSION_EPOCHS = 40
    FUSION_LR = 1e-3
    FUSION_PATIENCE = 10
    FUSION_CHECKPOINT = "best_fusion_model.pt"
    
    # ==================== AUGMENTATION ====================
    USE_MIXUP = CLIP_INDEPENDENT
    MIXUP_ALPHA = 0.2
    USE_LABEL_SMOOTHING = CLIP_INDEPENDENT
    LABEL_SMOOTHING = 0.1 if CLIP_INDEPENDENT else 0.0
    
    # ==================== EEG FREQUENCY BANDS ====================
    BANDS = [
        ("delta", (1, 3)),
        ("theta", (4, 7)),
        ("alpha", (8, 13)),
        ("beta", (14, 30)),
        ("gamma", (31, 45))
    ]
    
    # ==================== FILE OUTPUTS ====================
    SPLIT_FILE = "data_split_indices.npz"
    RESULTS_DIR = "results"
    
    def __init__(self):
        """Initialize config and set random seeds."""
        self.set_random_seeds()
    
    def set_random_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print(f"Device: {self.DEVICE}")
        print(f"Random Seed: {self.SEED}")
        print(f"\nDatasets:")
        print(f"  EEG (Preprocessed): {self.EEG_DATA_ROOT}")
        print(f"  BVP (Raw): {self.BVP_DATA_ROOT}")
        print(f"\nClassification:")
        print(f"  Mode: {'Subject-Independent' if self.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
        print(f"  Classes: {self.NUM_CLASSES}")
        print(f"  Baseline Reduction: {self.USE_BASELINE_REDUCTION}")
        print(f"\nEEG Settings:")
        print(f"  Window: {self.EEG_WINDOW_SEC}s, Overlap: {self.EEG_OVERLAP}")
        print(f"  Batch Size: {self.EEG_BATCH_SIZE}, Epochs: {self.EEG_EPOCHS}")
        print(f"  Learning Rate: {self.EEG_LR}")
        print(f"\nBVP Settings:")
        print(f"  Window: {self.BVP_WINDOW_SEC}s, Overlap: {self.BVP_OVERLAP}")
        print(f"  Batch Size: {self.BVP_BATCH_SIZE}, Epochs: {self.BVP_EPOCHS}")
        print(f"  Learning Rate: {self.BVP_LR}")
        print(f"\nFusion Settings:")
        print(f"  Shared Dim: {self.FUSION_SHARED_DIM}, Heads: {self.FUSION_NUM_HEADS}")
        print(f"  Batch Size: {self.FUSION_BATCH_SIZE}, Epochs: {self.FUSION_EPOCHS}")
        print("=" * 80)


# Create global config instance
config = Config()
