"""
Test EEG Model with Preprocessed Dataset
=========================================
Simple testing pipeline using modular components.

Usage:
    python test_eeg_model.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# Import modular components
from config import config
from data_loaders import load_eeg_data, create_data_split, filter_data_by_clips
from datasets import EEGWindowDataset, create_data_loaders
from models_eeg import SimpleBiLSTMClassifier
from utils import train_model, evaluate_model, plot_confusion_matrix, plot_training_history, print_classification_report


def main():
    """Complete EEG testing pipeline."""
    
    print("\n" + "="*80)
    print("🧠 EEG MODEL TESTING PIPELINE")
    print("="*80)
    
    # Print configuration
    config.print_config()
    
    # ============================================================
    # STEP 1: LOAD PREPROCESSED EEG DATA
    # ============================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    X_features, y_labels, clip_names, subjects = load_eeg_data(
        data_root=config.EEG_DATA_ROOT,
        use_baseline_reduction=config.USE_BASELINE_REDUCTION
    )
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Shape: {X_features.shape}")
    print(f"   Samples: {len(y_labels)}")
    print(f"   Clips: {len(np.unique(clip_names))}")
    print(f"   Subjects: {len(np.unique(subjects))}")
    
    # ============================================================
    # STEP 2: CREATE DATA SPLIT (SUBJECT-INDEPENDENT)
    # ============================================================
    print("\n" + "="*80)
    print("STEP 2: CREATING DATA SPLIT")
    print("="*80)
    
    split = create_data_split(
        clip_names=clip_names,
        labels=y_labels,
        subjects=subjects,
        test_ratio=0.15,
        val_ratio=0.15
    )
    
    # ============================================================
    # STEP 3: PREPARE DATASETS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 3: PREPARING DATASETS")
    print("="*80)
    
    # Filter data by split
    train_mask = np.isin(clip_names, split['train_clips'])
    val_mask = np.isin(clip_names, split['val_clips'])
    test_mask = np.isin(clip_names, split['test_clips'])
    
    X_train = X_features[train_mask]
    y_train = y_labels[train_mask]
    clip_train = clip_names[train_mask]
    
    X_val = X_features[val_mask]
    y_val = y_labels[val_mask]
    clip_val = clip_names[val_mask]
    
    X_test = X_features[test_mask]
    y_test = y_labels[test_mask]
    clip_test = clip_names[test_mask]
    
    print(f"\n📊 Dataset Sizes:")
    print(f"   Train: {len(X_train)} windows")
    print(f"   Val:   {len(X_val)} windows")
    print(f"   Test:  {len(X_test)} windows")
    
    # Standardize features
    print(f"\n🔧 Standardizing features...")
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    
    X_train_std = (X_train - mu) / sd
    X_val_std = (X_val - mu) / sd
    X_test_std = (X_test - mu) / sd
    
    # Create PyTorch datasets
    train_dataset = EEGWindowDataset(X_train_std, y_train, clip_train)
    val_dataset = EEGWindowDataset(X_val_std, y_val, clip_val)
    test_dataset = EEGWindowDataset(X_test_std, y_test, clip_test)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.EEG_BATCH_SIZE,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ DataLoaders created!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    # ============================================================
    # STEP 4: CREATE MODEL
    # ============================================================
    print("\n" + "="*80)
    print("STEP 4: CREATING MODEL")
    print("="*80)
    
    model = SimpleBiLSTMClassifier(
        dx=config.EEG_FEATURES,
        n_channels=config.EEG_CHANNELS,
        hidden=256,
        layers=3,
        n_classes=config.NUM_CLASSES,
        p_drop=0.4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Architecture:")
    print(f"   Input: ({config.EEG_CHANNELS}, {config.EEG_FEATURES})")
    print(f"   Hidden size: 256")
    print(f"   LSTM layers: 3 (bidirectional)")
    print(f"   Output classes: {config.NUM_CLASSES}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ============================================================
    # STEP 5: TRAIN MODEL
    # ============================================================
    print("\n" + "="*80)
    print("STEP 5: TRAINING MODEL")
    print("="*80)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_path=config.EEG_CHECKPOINT,
        is_fusion=False
    )
    
    # Plot training history
    print(f"\n📈 Plotting training history...")
    plot_training_history(history, save_path="eeg_training_history.png")
    
    # ============================================================
    # STEP 6: TEST EVALUATION
    # ============================================================
    print("\n" + "="*80)
    print("STEP 6: FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load(config.EEG_CHECKPOINT))
    model = model.to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    test_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=config.DEVICE,
        is_fusion=False
    )
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"   Accuracy:  {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"   Precision: {test_results['precision']:.4f}")
    print(f"   Recall:    {test_results['recall']:.4f}")
    print(f"   F1-Score:  {test_results['f1']:.4f}")
    
    # Detailed classification report
    print_classification_report(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        class_names=config.IDX_TO_LABEL
    )
    
    # Plot confusion matrix
    print(f"\n📊 Plotting confusion matrix...")
    plot_confusion_matrix(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        class_names=config.IDX_TO_LABEL,
        title='EEG Model - Test Set Confusion Matrix',
        save_path='eeg_confusion_matrix.png'
    )
    
    # ============================================================
    # STEP 7: SAVE RESULTS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 7: SAVING RESULTS")
    print("="*80)
    
    results_dict = {
        'test_accuracy': test_results['accuracy'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'test_f1': test_results['f1'],
        'predictions': test_results['predictions'],
        'labels': test_results['labels'],
        'history': history,
        'config': {
            'num_classes': config.NUM_CLASSES,
            'eeg_features': config.EEG_FEATURES,
            'eeg_channels': config.EEG_CHANNELS,
            'batch_size': config.EEG_BATCH_SIZE,
            'learning_rate': config.EEG_LR,
            'epochs_trained': len(history['train_loss']),
            'baseline_reduction': config.USE_BASELINE_REDUCTION
        }
    }
    
    np.savez('eeg_test_results.npz', **results_dict)
    print(f"✅ Results saved to: eeg_test_results.npz")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("🎉 TESTING COMPLETE!")
    print("="*80)
    print(f"\n📁 Generated Files:")
    print(f"   ✅ {config.EEG_CHECKPOINT} - Best model checkpoint")
    print(f"   ✅ eeg_training_history.png - Training curves")
    print(f"   ✅ eeg_confusion_matrix.png - Confusion matrix")
    print(f"   ✅ eeg_test_results.npz - Complete results")
    print(f"\n🎯 Final Test Accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"🎯 Final Test F1-Score: {test_results['f1']:.4f}")
    print("="*80)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    import random
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    main()
