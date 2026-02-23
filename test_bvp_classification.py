"""
BVP Classification Test Script
===============================

This script tests the complete BVP emotion classification pipeline:
- Load REAL BVP data from dataset using bvp_data_loader
- BVP encoder (with/without attention)
- Classification head
- End-to-end training and evaluation
- Visualization of results

Usage:
    python test_bvp_classification.py

Author: Final Year Project
Date: 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Import modules
from bvp_encoder import BVPEncoder, BVPEncoderWithAttention
from classification_head import (
    SimpleClassificationHead, 
    DeepClassificationHead,
    ResidualClassificationHead,
    get_classification_head
)
from bvp_config import BVPConfig
from bvp_data_loader import load_bvp_data, create_data_splits


# ==================================================
# CONFIGURATION
# ==================================================

class TestConfig:
    """Configuration for BVP classification test."""
    # Data path - UPDATE THIS TO YOUR DATASET PATH
    DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"  # Change this!
    
    # BVP Device Selection - Choose 'samsung_watch' or 'empatica'
    BVP_DEVICE = 'samsung_watch'  # Options: 'samsung_watch', 'empatica', 'both'
    
    # Data parameters
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    
    # Model parameters
    BVP_HIDDEN_SIZE = 32
    BVP_OUTPUT_SIZE = 64
    DROPOUT = 0.3
    
    # Training parameters
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    # BVP signal parameters (from BVPConfig)
    BVP_FS = 64  # Samsung Watch sampling frequency
    BVP_WINDOW_SEC = 10.0
    BVP_OVERLAP = 0.0
    
    # Preprocessing flags
    USE_BVP_BASELINE_CORRECTION = False
    USE_BVP_BASELINE_REDUCTION = True  # Set to True if you have baseline recordings
    
    # Subject-independent split
    SUBJECT_INDEPENDENT = True
    
    # Label mappings
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",
        "FEAR": "Q2",
        "SADNESS": "Q3",
        "NEUTRAL": "Q4",
    }
    
    SUPERCLASS_ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    
    # Labels
    EMOTION_LABELS = [
        "Q1_Positive_Active",   # Enthusiasm
        "Q2_Negative_Active",   # Fear
        "Q3_Negative_Calm",     # Sadness
        "Q4_Positive_Calm"      # Neutral
    ]
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42


config = TestConfig()

# Set random seed
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)


# ==================================================
# COMPLETE BVP CLASSIFIER
# ==================================================

class BVPEmotionClassifier(nn.Module):
    """
    Complete BVP emotion classifier.
    
    Combines BVP encoder + classification head.
    
    Args:
        encoder_type (str): 'simple' or 'attention'
        classifier_type (str): 'simple', 'deep', or 'residual'
        num_classes (int): Number of emotion classes
    """
    
    def __init__(self, encoder_type='attention', classifier_type='deep', num_classes=4):
        super(BVPEmotionClassifier, self).__init__()
        
        self.encoder_type = encoder_type
        self.classifier_type = classifier_type
        
        # BVP Encoder
        if encoder_type == 'simple':
            self.encoder = BVPEncoder(
                input_size=1,
                hidden_size=config.BVP_HIDDEN_SIZE,
                dropout=config.DROPOUT
            )
        elif encoder_type == 'attention':
            self.encoder = BVPEncoderWithAttention(
                input_size=1,
                hidden_size=config.BVP_HIDDEN_SIZE,
                dropout=config.DROPOUT
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Classification Head
        encoder_output_dim = self.encoder.get_output_dim()
        
        if classifier_type == 'simple':
            self.classifier = SimpleClassificationHead(
                input_dim=encoder_output_dim,
                num_classes=num_classes,
                hidden_dim=128,
                dropout=config.DROPOUT
            )
        elif classifier_type == 'deep':
            self.classifier = DeepClassificationHead(
                input_dim=encoder_output_dim,
                num_classes=num_classes,
                hidden_dims=[256, 128],
                dropout=config.DROPOUT
            )
        elif classifier_type == 'residual':
            self.classifier = ResidualClassificationHead(
                input_dim=encoder_output_dim,
                num_classes=num_classes,
                hidden_dim=256,
                num_blocks=2,
                dropout=config.DROPOUT
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): BVP signal [B, T, 1]
        
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        # Encode BVP
        bvp_context = self.encoder(x, return_sequence=False)  # [B, 64]
        
        # Classify
        logits = self.classifier(bvp_context)  # [B, num_classes]
        
        return logits


# ==================================================
# TRAINING FUNCTIONS
# ==================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (bvp_data, labels) in enumerate(train_loader):
        bvp_data = bvp_data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(bvp_data)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for bvp_data, labels in test_loader:
            bvp_data = bvp_data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(bvp_data)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100.0 * accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, num_epochs, device):
    """Complete training loop."""
    print("\n" + "="*80)
    print("TRAINING BVP EMOTION CLASSIFIER")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=5, verbose=True)
    
    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_f1s = []
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Print progress
        if epoch % 5 == 0 or epoch <= 10:
            print(f"Epoch {epoch:03d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Training complete! Best Val F1: {best_f1:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'val_f1s': val_f1s,
        'best_f1': best_f1
    }


# ==================================================
# VISUALIZATION FUNCTIONS
# ==================================================

def plot_training_history(history, save_path='bvp_classification_training.png'):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_accs'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_accs'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, history['val_f1s'], 'g-', label='Val F1', linewidth=2)
    axes[2].axhline(y=history['best_f1'], color='r', linestyle='--', 
                    label=f"Best F1: {history['best_f1']:.4f}")
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training history saved: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path='bvp_classification_confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - BVP Emotion Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved: {save_path}")
    plt.close()


def plot_sample_predictions(model, test_data, test_labels, num_samples=8, 
                           save_path='bvp_classification_samples.png'):
    """Plot sample BVP signals with predictions."""
    model.eval()
    
    # Select random samples
    indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            bvp_signal = test_data[idx]
            true_label = test_labels[idx]
            
            # Get prediction
            bvp_input = torch.FloatTensor(bvp_signal).unsqueeze(0).unsqueeze(-1).to(config.DEVICE)
            logits = model(bvp_input)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()
            
            # Plot
            ax = axes[i]
            time = np.linspace(0, 10, len(bvp_signal))
            ax.plot(time, bvp_signal, 'b-', linewidth=0.8)
            
            # Title with prediction
            color = 'green' if pred_label == true_label else 'red'
            ax.set_title(f"True: {config.EMOTION_LABELS[true_label]}\n"
                        f"Pred: {config.EMOTION_LABELS[pred_label]} ({confidence:.2f})",
                        fontsize=10, fontweight='bold', color=color)
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('BVP', fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Sample predictions saved: {save_path}")
    plt.close()


# ==================================================
# MAIN TEST EXECUTION
# ==================================================

def main():
    """Run complete BVP classification test."""
    print("="*80)
    print("BVP EMOTION CLASSIFICATION TEST (REAL DATA)")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Data path: {config.DATA_ROOT}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Subject-independent: {config.SUBJECT_INDEPENDENT}")
    print("="*80)
    
    # Step 1: Load REAL BVP data from dataset
    print("\n📊 Loading REAL BVP data from dataset...")
    
    try:
        X_raw, y_labels, subject_ids, label_to_id = load_bvp_data(config.DATA_ROOT, config)
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Total samples: {len(X_raw)}")
        print(f"   Signal shape: {X_raw.shape}")
        print(f"   Unique subjects: {len(np.unique(subject_ids))}")
        print(f"   Label mapping: {label_to_id}")
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        print(f"\n💡 Make sure to update DATA_ROOT in the config to your dataset path!")
        print(f"   Current path: {config.DATA_ROOT}")
        print(f"\n   Expected file pattern: *_STIMULUS_SAMSUNG_WATCH.json or *_STIMULUS_EMPATICA.json")
        return
    
    # Step 2: Create data splits
    split_indices = create_data_splits(y_labels, subject_ids, config,
                                      train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    X_train = X_raw[train_idx]
    y_train = y_labels[train_idx]
    X_val = X_raw[val_idx]
    y_val = y_labels[val_idx]
    X_test = X_raw[test_idx]
    y_test = y_labels[test_idx]
    
    print(f"\n📂 Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Step 3: Create data loaders
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)  # [N, T, 1]
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatFloat(X_test).unsqueeze(-1)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Step 4: Test different model configurations
    model_configs = [
        ('attention', 'simple'),
        ('attention', 'deep'),
        ('simple', 'deep'),
    ]
    
    results = {}
    
    for encoder_type, classifier_type in model_configs:
        model_name = f"{encoder_type}_{classifier_type}"
        print(f"\n{'='*80}")
        print(f"TESTING: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Create model
        model = BVPEmotionClassifier(
            encoder_type=encoder_type,
            classifier_type=classifier_type,
            num_classes=config.NUM_CLASSES
        ).to(config.DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Train model
        history = train_model(model, train_loader, val_loader, config.NUM_EPOCHS, config.DEVICE)
        
        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_f1, test_preds, test_labels_eval = evaluate(
            model, test_loader, criterion, config.DEVICE
        )
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   Test F1 Score: {test_f1:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(test_labels_eval, test_preds, 
                                   target_names=config.EMOTION_LABELS,
                                   digits=3))
        
        # Save results
        results[model_name] = {
            'history': history,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_preds': test_preds,
            'test_labels': test_labels_eval,
            'model': model
        }
        
        # Visualizations for best model (attention_deep)
        if model_name == 'attention_deep':
            plot_training_history(history, f'bvp_classification_training_{model_name}.png')
            plot_confusion_matrix(test_labels_eval, test_preds, config.EMOTION_LABELS,
                                f'bvp_classification_confusion_matrix_{model_name}.png')
            plot_sample_predictions(model, X_test, y_test, num_samples=8,
                                  save_path=f'bvp_classification_samples_{model_name}.png')
    
    # Step 5: Compare models
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Test Acc (%)':<15} {'Test F1':<15} {'Parameters':<15}")
    print("-"*80)
    
    for model_name, result in results.items():
        model = result['model']
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name:<25} {result['test_acc']:<15.2f} "
              f"{result['test_f1']:<15.4f} {total_params:<15,}")
    
    print("="*80)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"   Test Accuracy: {best_result['test_acc']:.2f}%")
    print(f"   Test F1 Score: {best_result['test_f1']:.4f}")
    
    print("\n" + "="*80)
    print("🎉 BVP CLASSIFICATION TEST COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - bvp_classification_training_attention_deep.png")
    print("   - bvp_classification_confusion_matrix_attention_deep.png")
    print("   - bvp_classification_samples_attention_deep.png")
    print("="*80)


if __name__ == "__main__":
    main()
