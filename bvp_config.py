"""
BVP Pipeline Configuration
===========================

This module contains all configuration parameters for the BVP (Blood Volume Pulse)
emotion recognition pipeline from wearable devices (Samsung Watch / Empatica).

Author: Final Year Project
Date: 2026
"""

import torch


class BVPConfig:
    """BVP-specific configuration."""
    
    # ==========================================
    # PATHS
    # ==========================================
    DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
    
    # ==========================================
    # COMMON PARAMETERS
    # ==========================================
    NUM_CLASSES = 4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # BASELINE PREPROCESSING
    # ==========================================
    # Baseline Correction: Removes drift within the signal using local minima
    USE_BVP_BASELINE_CORRECTION = False
    
    # Baseline Reduction: Uses baseline recordings (InvBase method) to normalize
    # and reduce inter-subject variability (similar to EEG baseline reduction)
    USE_BVP_BASELINE_REDUCTION = True
    
    # ==========================================
    # DATA SPLIT MODE
    # ==========================================
    SUBJECT_INDEPENDENT = True  # Subject-independent evaluation
    
    # ==========================================
    # LABEL MAPPINGS
    # ==========================================
    # 4-class emotion quadrants (valence-arousal model)
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",  # Positive + High Arousal
        "FEAR": "Q2",         # Negative + High Arousal
        "SADNESS": "Q3",      # Negative + Low Arousal
        "NEUTRAL": "Q4",      # Positive + Low Arousal
    }
    
    SUPERCLASS_ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    
    IDX_TO_LABEL = [
        "Q1_Positive_Active",   # Enthusiasm
        "Q2_Negative_Active",   # Fear
        "Q3_Negative_Calm",     # Sadness
        "Q4_Positive_Calm"      # Neutral
    ]
    
    # ==========================================
    # BVP SIGNAL PARAMETERS
    # ==========================================
    BVP_FS = 64  # Samsung Watch sampling frequency (Hz)
    BVP_WINDOW_SEC = 10.0  # Window size in seconds
    BVP_OVERLAP = 0.0  # No overlap (0.0 = no overlap, 0.5 = 50% overlap)
    
    # ==========================================
    # BVP FILTERING PARAMETERS
    # ==========================================
    # Bandpass filter settings
    BVP_HIGHPASS_CUTOFF = 0.5   # Hz - Removes DC offset and very low-frequency drift
    BVP_LOWPASS_CUTOFF = 15.0   # Hz - Removes high-frequency noise
    BVP_FILTER_ORDER = 6        # Butterworth filter order
    
    # Wavelet denoising settings
    BVP_WAVELET = "db4"         # Daubechies 4 wavelet
    BVP_DENOISE_LEVEL = 4       # Wavelet decomposition level
    
    # Enable/disable highpass filtering
    USE_BVP_HIGHPASS = True     # Apply highpass filter (bandpass if True, lowpass only if False)
    
    # ==========================================
    # BVP ENCODER PARAMETERS
    # ==========================================
    BVP_INPUT_SIZE = 1          # Raw BVP signal (1D)
    BVP_HIDDEN_SIZE = 32        # LSTM hidden size
    BVP_OUTPUT_SIZE = 64        # Bidirectional LSTM output (hidden_size * 2)
    BVP_DROPOUT = 0.3           # Dropout rate
    
    # Use attention mechanism for BVP encoder
    USE_BVP_ATTENTION = True    # If True, use BVPEncoderWithAttention; else BVPEncoder
    
    # ==========================================
    # BVP TRAINING PARAMETERS
    # ==========================================
    BVP_BATCH_SIZE = 16
    BVP_EPOCHS = 100
    BVP_LR = 1e-3               # Learning rate
    BVP_WEIGHT_DECAY = 1e-4     # L2 regularization
    BVP_PATIENCE = 15           # Early stopping patience
    BVP_CHECKPOINT = "best_bvp_model.pt"
    
    # ==========================================
    # BVP FEATURE EXTRACTION
    # ==========================================
    # Number of handcrafted features extracted from BVP
    BVP_NUM_FEATURES = 5  # [mean, std, diff_std, hr_proxy, peak2peak]
    
    # ==========================================
    # CLASS WEIGHTS
    # ==========================================
    # Automatically computed from data if None
    BVP_CLASS_WEIGHTS = None    # Set to None for automatic computation
    
    # ==========================================
    # AUGMENTATION SETTINGS
    # ==========================================
    USE_BVP_AUGMENTATION = False  # Enable data augmentation for BVP
    BVP_NOISE_LEVEL = 0.01        # Gaussian noise standard deviation
    
    # ==========================================
    # PHYSIOLOGICAL RANGES
    # ==========================================
    # Heart rate range (for validation/sanity checks)
    MIN_HEART_RATE = 30  # bpm
    MAX_HEART_RATE = 240  # bpm
    
    # ==========================================
    # LOGGING AND VISUALIZATION
    # ==========================================
    VERBOSE = True              # Print detailed logs
    SAVE_PLOTS = True           # Save training plots
    PLOT_DIR = "bvp_plots"      # Directory for saving plots
    LOG_DIR = "bvp_logs"        # Directory for saving logs


# Create a default config instance
config = BVPConfig()


# Utility function to print config
def print_config():
    """Print all BVP configuration parameters."""
    print("=" * 80)
    print("BVP CONFIGURATION")
    print("=" * 80)
    
    print("\n📁 PATHS:")
    print(f"   Data root: {BVPConfig.DATA_ROOT}")
    
    print("\n🔧 PREPROCESSING:")
    print(f"   Baseline Correction: {BVPConfig.USE_BVP_BASELINE_CORRECTION}")
    print(f"   Baseline Reduction:  {BVPConfig.USE_BVP_BASELINE_REDUCTION}")
    print(f"   Highpass Filter:     {BVPConfig.USE_BVP_HIGHPASS}")
    
    print("\n📊 SIGNAL PARAMETERS:")
    print(f"   Sampling rate:    {BVPConfig.BVP_FS} Hz")
    print(f"   Window size:      {BVPConfig.BVP_WINDOW_SEC} sec")
    print(f"   Overlap:          {BVPConfig.BVP_OVERLAP * 100}%")
    print(f"   Highpass cutoff:  {BVPConfig.BVP_HIGHPASS_CUTOFF} Hz")
    print(f"   Lowpass cutoff:   {BVPConfig.BVP_LOWPASS_CUTOFF} Hz")
    
    print("\n🧠 ENCODER PARAMETERS:")
    print(f"   Hidden size:      {BVPConfig.BVP_HIDDEN_SIZE}")
    print(f"   Output size:      {BVPConfig.BVP_OUTPUT_SIZE}")
    print(f"   Use attention:    {BVPConfig.USE_BVP_ATTENTION}")
    print(f"   Dropout:          {BVPConfig.BVP_DROPOUT}")
    
    print("\n🎯 TRAINING:")
    print(f"   Batch size:       {BVPConfig.BVP_BATCH_SIZE}")
    print(f"   Epochs:           {BVPConfig.BVP_EPOCHS}")
    print(f"   Learning rate:    {BVPConfig.BVP_LR}")
    print(f"   Patience:         {BVPConfig.BVP_PATIENCE}")
    
    print("\n🏷️  LABELS:")
    print(f"   Num classes:      {BVPConfig.NUM_CLASSES}")
    print(f"   Class mapping:    {BVPConfig.SUPERCLASS_MAP}")
    
    print("\n💾 OTHER:")
    print(f"   Subject-independent: {BVPConfig.SUBJECT_INDEPENDENT}")
    print(f"   Device:              {BVPConfig.DEVICE}")
    print(f"   Random seed:         {BVPConfig.SEED}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config()
