"""
BVP Hybrid Encoder Pipeline - Emotion Recognition
==================================================

This pipeline uses BVP Hybrid Encoder (Deep + Handcrafted Features) 
for emotion recognition with real data.

The hybrid approach combines:
- Deep learned features: Conv1d + BiLSTM (64 features)
- Handcrafted features: Statistical + HRV + Pulse (11 features)
- Total: 75-dimensional hybrid representation

Usage:
    python bvp_pipeline.py --head deep --baseline_reduction
    python bvp_pipeline.py --head simple
    python bvp_pipeline.py --eval_only --checkpoint best_bvp_hybrid.pt

Author: Final Year Project
Date: 2026-02-24
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import custom modules
from bvp_config import BVPConfig
from bvp_data_loader import load_bvp_data, create_data_splits
from bvp_hybrid_encoder import BVPHybridEncoder
from classification_head import get_classification_head


# ==================================================
# DATASET CLASS
# ==================================================

class BVPDataset(Dataset):
    """PyTorch Dataset for BVP signals."""
    
    def __init__(self, X, y, subjects):
        """
        Args:
            X: BVP windows [N, T]
            y: Labels [N]
            subjects: Subject IDs [N]
        """
        self.X = torch.FloatTensor(X).unsqueeze(-1)  # [N, T, 1]
        self.y = torch.LongTensor(y)
        self.subjects = subjects
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================================================
# BVP HYBRID MODEL
# ==================================================

class BVPHybridModel(nn.Module):
    """
    BVP Hybrid Model for Emotion Recognition.
    
    Combines deep learning and handcrafted features:
    - Deep: Conv1d → BatchNorm → ReLU → BiLSTM → [B, 64]
    - Handcrafted: Statistical + HRV + Pulse → [B, 11]
    - Hybrid: Concatenated → [B, 75]
    - Classification Head → [B, num_classes]
    """
    
    def __init__(self, num_classes=4, hidden_size=32, dropout=0.3, 
                 sampling_rate=64.0, head_type='deep', head_params=None):
        super(BVPHybridModel, self).__init__()
        
        self.head_type = head_type
        
        # Hybrid encoder (deep + handcrafted features)
        self.encoder = BVPHybridEncoder(
            input_size=1,
            hidden_size=hidden_size,
            dropout=dropout,
            sampling_rate=sampling_rate,
            min_peak_distance=20
        )
        
        # Get encoder output dimension (75 = 64 + 11)
        encoder_dim = self.encoder.get_output_dim()
        
        # Classification head
        if head_params is None:
            head_params = {}
        
        self.classifier = get_classification_head(
            head_type=head_type,
            input_dim=encoder_dim,
            num_classes=num_classes,
            **head_params
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: BVP signal [B, T, 1]
        
        Returns:
            logits: [B, num_classes]
        """
        # Extract hybrid features (deep + handcrafted)
        hybrid_features = self.encoder(x)  # [B, 75]
        
        # Classify
        logits = self.classifier(hybrid_features)  # [B, num_classes]
        
        return logits
    
    def get_features(self, x, return_separate=False):
        """Get hybrid features without classification."""
        return self.encoder(x, return_separate=return_separate)


# ==================================================
# TRAINING FUNCTIONS
# ==================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, config, save_path='best_bvp_hybrid.pt'):
    """Training loop with early stopping."""
    
    device = config.DEVICE
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.BVP_LR, 
                          weight_decay=config.BVP_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print("\n" + "="*80)
    print("TRAINING BVP HYBRID MODEL")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: BVP Hybrid Encoder (Deep + Handcrafted)")
    print(f"Classification Head: {model.head_type}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Feature breakdown
    breakdown = model.encoder.get_feature_breakdown()
    print(f"\nFeature Breakdown:")
    print(f"  Deep features:        {breakdown['deep_features']}")
    print(f"  Handcrafted features: {breakdown['handcrafted_features']}")
    print(f"  Total hybrid:         {breakdown['total_features']}")
    print("="*80)
    
    for epoch in range(config.BVP_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.BVP_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"  📉 Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        
        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'head_type': model.head_type
            }, save_path)
            print(f"  ✅ Model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{config.BVP_PATIENCE}")
        
        if patience_counter >= config.BVP_PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*80)
    print(f"✅ TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print("="*80)
    
    return history, best_val_acc


# ==================================================
# EVALUATION FUNCTIONS
# ==================================================

def evaluate_model(model, test_loader, config, label_names=None):
    """Comprehensive model evaluation."""
    
    device = config.DEVICE
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "="*80)
    print("EVALUATING BVP HYBRID MODEL ON TEST SET")
    print("="*80)
    
    test_loss, test_acc, test_f1, all_preds, all_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📊 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   F1-Score (weighted): {test_f1:.4f}")
    
    if label_names is None:
        label_names = [f"Class {i}" for i in range(config.NUM_CLASSES)]
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - BVP Hybrid Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('bvp_hybrid_confusion_matrix.png', dpi=300)
    print("\n💾 Confusion matrix saved: bvp_hybrid_confusion_matrix.png")
    plt.close()
    
    print("="*80)
    
    return test_acc, test_f1


def plot_training_history(history, save_path='bvp_hybrid_training_history.png'):
    """Plot training history."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 score plot
    axes[2].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n💾 Training history saved: {save_path}")
    plt.close()


# ==================================================
# MAIN PIPELINE
# ==================================================

def main(args):
    """Main BVP Hybrid pipeline."""
    
    # Set random seeds
    random.seed(BVPConfig.SEED)
    np.random.seed(BVPConfig.SEED)
    torch.manual_seed(BVPConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BVPConfig.SEED)
    
    print("\n" + "="*80)
    print("BVP HYBRID ENCODER - EMOTION RECOGNITION PIPELINE")
    print("="*80)
    print(f"📋 Configuration:")
    print(f"   Model: BVP Hybrid Encoder (Deep + Handcrafted)")
    print(f"   Classification Head: {args.head}")
    print(f"   Baseline Reduction: {args.baseline_reduction}")
    print(f"   Device: {args.device}")
    print(f"   Data Root: {args.data_root}")
    print("="*80)
    
    # Update config
    BVPConfig.DATA_ROOT = args.data_root
    BVPConfig.USE_BVP_BASELINE_REDUCTION = args.baseline_reduction
    BVPConfig.USE_BVP_BASELINE_CORRECTION = False
    BVPConfig.BVP_DEVICE = args.bvp_device
    BVPConfig.DEVICE = torch.device(args.device)
    
    # Load real data
    print("\n📂 Loading real BVP data...")
    X_raw, y_labels, subject_ids, label_to_id = load_bvp_data(BVPConfig.DATA_ROOT, BVPConfig)
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Shape: {X_raw.shape}")
    print(f"   Subjects: {len(np.unique(subject_ids))}")
    print(f"   Classes: {BVPConfig.NUM_CLASSES}")
    print(f"   Labels: {BVPConfig.IDX_TO_LABEL}")
    
    # Create splits (subject-independent)
    split_indices = create_data_splits(y_labels, subject_ids, BVPConfig)
    
    # Create datasets
    train_dataset = BVPDataset(
        X_raw[split_indices['train']], 
        y_labels[split_indices['train']], 
        subject_ids[split_indices['train']]
    )
    val_dataset = BVPDataset(
        X_raw[split_indices['val']], 
        y_labels[split_indices['val']], 
        subject_ids[split_indices['val']]
    )
    test_dataset = BVPDataset(
        X_raw[split_indices['test']], 
        y_labels[split_indices['test']], 
        subject_ids[split_indices['test']]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\n📦 Dataloaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Create BVP Hybrid Model
    print(f"\n🧠 Creating BVP Hybrid Model...")
    
    head_params = {}
    if args.head == 'deep':
        head_params['hidden_dims'] = [256, 128]
    elif args.head == 'residual':
        head_params['hidden_dim'] = 256
        head_params['num_blocks'] = 2
    elif args.head == 'attention':
        head_params['hidden_dim'] = 256
        head_params['num_heads'] = 4
    
    model = BVPHybridModel(
        num_classes=BVPConfig.NUM_CLASSES,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        sampling_rate=64.0,
        head_type=args.head,
        head_params=head_params
    )
    
    print(f"✅ Model created!")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    if not args.eval_only:
        history, best_val_acc = train_model(
            model, train_loader, val_loader, BVPConfig, 
            save_path=args.checkpoint
        )
        
        # Plot training history
        plot_training_history(history)
    
    # Load best model for evaluation
    print(f"\n📥 Loading best model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=BVPConfig.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded (Val Acc: {checkpoint['val_acc']:.2f}%)")
    
    # Evaluate on test set
    test_acc, test_f1 = evaluate_model(model, test_loader, BVPConfig, 
                                       label_names=BVPConfig.IDX_TO_LABEL)
    
    # Save results
    results = {
        'model_type': 'BVP Hybrid Encoder',
        'head_type': args.head,
        'baseline_reduction': args.baseline_reduction,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'val_accuracy': checkpoint['val_acc'],
        'val_f1': checkpoint['val_f1']
    }
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print(f"📊 Final Results:")
    print(f"   Validation Accuracy: {results['val_accuracy']:.2f}%")
    print(f"   Validation F1-Score: {results['val_f1']:.4f}")
    print(f"   Test Accuracy:       {results['test_accuracy']:.2f}%")
    print(f"   Test F1-Score:       {results['test_f1']:.4f}")
    print("="*80)
    
    return model, results


# ==================================================
# COMMAND LINE INTERFACE
# ==================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='BVP Hybrid Encoder Pipeline - Emotion Recognition'
    )
    
    # Classification head
    parser.add_argument('--head', type=str, default='deep',
                       choices=['simple', 'deep', 'residual', 'attention'],
                       help='Classification head type (default: deep)')
    
    # Preprocessing
    parser.add_argument('--baseline_reduction', action='store_true',
                       help='Enable baseline reduction (InvBase method)')
    parser.add_argument('--bvp_device', type=str, default='samsung_watch',
                       choices=['samsung_watch', 'empatica', 'both'],
                       help='BVP device to use (default: samsung_watch)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--hidden_size', type=int, default=32,
                       help='LSTM hidden size (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    
    # Paths
    parser.add_argument('--data_root', type=str, 
                       default='E:/FInal Year Project/MyCodeSpace/Current(2026-02-21)/Data',
                       help='Path to data directory')
    parser.add_argument('--checkpoint', type=str, default='best_bvp_hybrid.pt',
                       help='Path to save/load model checkpoint')
    
    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, skip training')
    
    args = parser.parse_args()
    
    # Run pipeline
    model, results = main(args)
