"""
    Configuration Module
    ====================
    Shared configuration for EEG emotion recognition.
"""

import torch


class Config:
    """Shared configuration for all models."""
    # Paths
    DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"  # Preprocessed dataset
    
    # Common parameters
    NUM_CLASSES = 4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Classification Mode
    USE_DUAL_BINARY = False
    CLASSIFY_WHOLE_CLIPS = False
    CLIP_AGGREGATION_METHOD = "mean"
    
    # Baseline reduction (InvBase method)
    USE_BASELINE_REDUCTION = True
    
    # Data split mode
    SUBJECT_INDEPENDENT = True
    CLIP_INDEPENDENT = False
    
    # LOSO Cross-Validation
    USE_LOSO = False
    LOSO_SAVE_ALL_FOLDS = True
    
    # Stratified split parameters
    USE_STRATIFIED_GROUP_SPLIT = True
    MIN_SAMPLES_PER_CLASS = 10
    
    # Label mappings (4-class system)
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",
        "FEAR": "Q2",
        "SADNESS": "Q3",
        "NEUTRAL": "Q4",
    }
    
    SUPERCLASS_ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    IDX_TO_LABEL = ["Q1_Positive_Active", "Q2_Negative_Active", "Q3_Negative_Calm", "Q4_Positive_Calm"]
    
    # Dual Binary Classification Mappings
    AROUSAL_MAP = {0: 1, 1: 1, 2: 0, 3: 0}
    VALENCE_MAP = {0: 1, 1: 0, 2: 0, 3: 1}
    AROUSAL_LABELS = ["Low_Arousal", "High_Arousal"]
    VALENCE_LABELS = ["Negative_Valence", "Positive_Valence"]
    
    # EEG parameters
    EEG_FS = 256.0
    EEG_CHANNELS = 4
    EEG_FEATURES = 26
    EEG_WINDOW_SEC = 10.0
    EEG_OVERLAP = 0.5 if CLIP_INDEPENDENT else 0.0
    EEG_BATCH_SIZE = 32 if CLIP_INDEPENDENT else 64
    EEG_EPOCHS = 200 if CLIP_INDEPENDENT else 150
    EEG_LR = 5e-4 if CLIP_INDEPENDENT else 1e-3
    EEG_PATIENCE = 30 if CLIP_INDEPENDENT else 20
    EEG_CHECKPOINT = "best_eeg_pp_only.pt"
    
    # Augmentation settings
    USE_MIXUP = CLIP_INDEPENDENT
    USE_LABEL_SMOOTHING = CLIP_INDEPENDENT
    LABEL_SMOOTHING = 0.1 if CLIP_INDEPENDENT else 0.0
    
    # File outputs
    SPLIT_FILE = "data_split_indices.npz"
    
    # Frequency bands for EEG
    BANDS = [("delta", (1, 3)), ("theta", (4, 7)), ("alpha", (8, 13)), 
            ("beta", (14, 30)), ("gamma", (31, 45))]
