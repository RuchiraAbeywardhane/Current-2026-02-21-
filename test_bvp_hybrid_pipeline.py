"""
BVP Hybrid Pipeline with Baseline Reduction - Test Script
===========================================================

This script demonstrates the complete BVP emotion recognition pipeline:
1. Load BVP data using the data loader
2. Apply baseline reduction (InvBase method) - NOT baseline correction
3. Extract hybrid features (deep + handcrafted) using the hybrid encoder
4. Classify emotions using a BiLSTM classification head

Pipeline Flow:
    Raw BVP → Preprocessing + Baseline Reduction → Hybrid Encoder → BiLSTM Classifier → Emotions

Author: Final Year Project
Date: 2026-02-24
"""

import os
import sys
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import BVP modules
from bvp_config import BVPConfig
from bvp_data_loader import load_bvp_data, create_data_splits
from bvp_hybrid_encoder import BVPHybridEncoder


# ==================================================
# BILSTM CLASSIFICATION HEAD
# ==================================================

class BiLSTMClassificationHead(nn.Module):
    """
    BiLSTM-based classification head for emotion recognition.
    
    This classifier takes hybrid features from the BVP encoder and
    uses a BiLSTM to capture temporal dependencies before classification.
    
    Architecture:
    1. Input: [batch_size, feature_dim] hybrid features
    2. Reshape to sequence: [batch_size, 1, feature_dim] (single timestep)
    3. BiLSTM layer: Captures bidirectional patterns
    4. Dropout: Regularization
    5. Fully connected: Maps to emotion classes
    
    Args:
        input_dim (int): Input feature dimension (75 for hybrid encoder)
        hidden_dim (int): LSTM hidden dimension (default: 64)
        num_classes (int): Number of emotion classes (default: 4)
        num_layers (int): Number of BiLSTM layers (default: 2)
        dropout (float): Dropout rate (default: 0.4)
        
    Input Shape:
        - x: [batch_size, input_dim] - hybrid features
        
    Output Shape:
        - logits: [batch_size, num_classes] - class scores
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=4, num_layers=2, dropout=0.4):
        super(BiLSTMClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer for classification
        # Input: hidden_dim * 2 (bidirectional)
        # Output: num_classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass through BiLSTM classifier.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Reshape to sequence: [batch_size, seq_len=1, input_dim]
        x = x.unsqueeze(1)
        
        # BiLSTM forward pass
        # lstm_out: [batch_size, seq_len=1, hidden_dim*2]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output from the single timestep
        # [batch_size, hidden_dim*2]
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layer for classification
        logits = self.fc(lstm_out)
        
        return logits


# ==================================================
# COMPLETE BVP HYBRID MODEL
# ==================================================

class BVPHybridModel(nn.Module):
    """
    Complete BVP emotion recognition model.
    
    Combines:
    1. BVP Hybrid Encoder (deep + handcrafted features)
    2. BiLSTM Classification Head
    
    Args:
        encoder_hidden_size (int): LSTM hidden size for encoder (default: 32)
        encoder_dropout (float): Dropout for encoder (default: 0.3)
        classifier_hidden_dim (int): LSTM hidden size for classifier (default: 64)
        classifier_num_layers (int): Number of BiLSTM layers (default: 2)
        classifier_dropout (float): Dropout for classifier (default: 0.4)
        num_classes (int): Number of emotion classes (default: 4)
        sampling_rate (float): BVP sampling rate in Hz (default: 64.0)
        
    Input Shape:
        - x: [batch_size, time_steps, 1] - preprocessed BVP signal
        
    Output Shape:
        - logits: [batch_size, num_classes] - class scores
    """
    
    def __init__(
        self,
        encoder_hidden_size=32,
        encoder_dropout=0.3,
        classifier_hidden_dim=64,
        classifier_num_layers=2,
        classifier_dropout=0.4,
        num_classes=4,
        sampling_rate=64.0
    ):
        super(BVPHybridModel, self).__init__()
        
        # Hybrid encoder (deep + handcrafted features)
        self.encoder = BVPHybridEncoder(
            input_size=1,
            hidden_size=encoder_hidden_size,
            dropout=encoder_dropout,
            sampling_rate=sampling_rate
        )
        
        # Get encoder output dimension (75 = 64 deep + 11 handcrafted)
        encoder_output_dim = self.encoder.get_output_dim()
        
        # BiLSTM classification head
        self.classifier = BiLSTMClassificationHead(
            input_dim=encoder_output_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            num_layers=classifier_num_layers,
            dropout=classifier_dropout
        )
        
    def forward(self, x):
        """
        Forward pass through the complete model.
        
        Args:
            x (torch.Tensor): Input BVP signal [batch_size, time_steps, 1]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        # Step 1: Extract hybrid features
        hybrid_features = self.encoder(x)  # [batch_size, 75]
        
        # Step 2: Classify using BiLSTM head
        logits = self.classifier(hybrid_features)  # [batch_size, num_classes]
        
        return logits
    
    def get_feature_breakdown(self):
        """Get detailed breakdown of model architecture."""
        return self.encoder.get_feature_breakdown()


# ==================================================
# TRAINING AND EVALUATION FUNCTIONS
# ==================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def compute_class_weights(y_labels, num_classes):
    """Compute class weights for imbalanced datasets."""
    class_counts = Counter(y_labels)
    total_samples = len(y_labels)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Avoid division by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


# ==================================================
# MAIN PIPELINE
# ==================================================

def main(args):
    """Main training pipeline."""
    
    print("\n" + "="*80)
    print("BVP HYBRID PIPELINE WITH BASELINE REDUCTION")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(BVPConfig.SEED)
    np.random.seed(BVPConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(BVPConfig.SEED)
    
    device = BVPConfig.DEVICE
    print(f"\n🖥️  Device: {device}")
    
    # ============================================================
    # STEP 1: LOAD DATA WITH BASELINE REDUCTION
    # ============================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    
    # Enable baseline reduction, disable baseline correction
    BVPConfig.USE_BVP_BASELINE_REDUCTION = True
    BVPConfig.USE_BVP_BASELINE_CORRECTION = False
    
    print(f"✅ Baseline Reduction (InvBase): ENABLED")
    print(f"❌ Baseline Correction (drift removal): DISABLED")
    
    # Load data
    X_raw, y_labels, subject_ids, label_to_id = load_bvp_data(
        args.data_root,
        BVPConfig
    )
    
    print(f"\n📊 Data loaded:")
    print(f"   Shape: {X_raw.shape}")
    print(f"   Labels: {Counter(y_labels)}")
    print(f"   Subjects: {len(np.unique(subject_ids))}")
    
    # ============================================================
    # STEP 2: CREATE TRAIN/VAL/TEST SPLITS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 2: DATA SPLITTING")
    print("="*80)
    
    split_indices = create_data_splits(
        y_labels,
        subject_ids,
        BVPConfig,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Extract splits
    X_train = X_raw[split_indices['train']]
    y_train = y_labels[split_indices['train']]
    
    X_val = X_raw[split_indices['val']]
    y_val = y_labels[split_indices['val']]
    
    X_test = X_raw[split_indices['test']]
    y_test = y_labels[split_indices['test']]
    
    print(f"\n✅ Splits created:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # ============================================================
    # STEP 3: CREATE PYTORCH DATALOADERS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 3: CREATING DATALOADERS")
    print("="*80)
    
    # Reshape for PyTorch: [N, T] → [N, T, 1]
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(-1).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    
    X_val_tensor = torch.from_numpy(X_val).unsqueeze(-1).float()
    y_val_tensor = torch.from_numpy(y_val).long()
    
    X_test_tensor = torch.from_numpy(X_test).unsqueeze(-1).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    # ============================================================
    # STEP 4: INITIALIZE MODEL
    # ============================================================
    print("\n" + "="*80)
    print("STEP 4: MODEL INITIALIZATION")
    print("="*80)
    
    model = BVPHybridModel(
        encoder_hidden_size=args.encoder_hidden,
        encoder_dropout=args.encoder_dropout,
        classifier_hidden_dim=args.classifier_hidden,
        classifier_num_layers=args.classifier_layers,
        classifier_dropout=args.classifier_dropout,
        num_classes=BVPConfig.NUM_CLASSES,
        sampling_rate=BVPConfig.BVP_FS
    ).to(device)
    
    # Display model info
    breakdown = model.get_feature_breakdown()
    print(f"\n🧠 Model Architecture:")
    print(f"   Encoder:")
    print(f"      Deep features:        {breakdown['deep_features']}")
    print(f"      Handcrafted features: {breakdown['handcrafted_features']}")
    print(f"      Total features:       {breakdown['total_features']}")
    print(f"   Classifier:")
    print(f"      BiLSTM hidden dim:    {args.classifier_hidden}")
    print(f"      BiLSTM layers:        {args.classifier_layers}")
    print(f"      Output classes:       {BVPConfig.NUM_CLASSES}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Parameters:")
    print(f"   Total:     {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # ============================================================
    # STEP 5: SETUP TRAINING
    # ============================================================
    print("\n" + "="*80)
    print("STEP 5: TRAINING SETUP")
    print("="*80)
    
    # Compute class weights for imbalanced data
    if args.use_class_weights:
        class_weights = compute_class_weights(y_train, BVPConfig.NUM_CLASSES)
        class_weights = class_weights.to(device)
        print(f"\n⚖️  Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
        print(f"\n⚖️  Class weights: None (uniform)")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Epochs:        {args.epochs}")
    print(f"   Batch size:    {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Weight decay:  {args.weight_decay}")
    print(f"   Early stop:    {args.patience} epochs")
    
    # ============================================================
    # STEP 6: TRAINING LOOP
    # ============================================================
    print("\n" + "="*80)
    print("STEP 6: TRAINING")
    print("="*80)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, args.checkpoint)
            
            print(f"   ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
            break
    
    # ============================================================
    # STEP 7: FINAL EVALUATION ON TEST SET
    # ============================================================
    print("\n" + "="*80)
    print("STEP 7: TEST SET EVALUATION")
    print("="*80)
    
    # Load best model
    print(f"\n📂 Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n🎯 Final Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Acc:  {test_acc:.2f}%")
    
    # Per-class accuracy
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    print(f"\n📊 Per-Class Accuracy:")
    for class_idx in range(BVPConfig.NUM_CLASSES):
        mask = test_labels == class_idx
        if mask.sum() > 0:
            class_acc = 100.0 * (test_preds[mask] == test_labels[mask]).sum() / mask.sum()
            class_name = BVPConfig.IDX_TO_LABEL[class_idx]
            print(f"   {class_name:25s}: {class_acc:.2f}%")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n📋 Confusion Matrix:")
    print(cm)
    
    print(f"\n📄 Classification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=BVPConfig.IDX_TO_LABEL
    ))
    
    print("\n" + "="*80)
    print("✅ BVP HYBRID PIPELINE COMPLETE!")
    print("="*80)
    
    print(f"\n📈 Summary:")
    print(f"   Best Validation Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"   Final Test Acc:      {test_acc:.2f}%")
    print(f"   Model saved to:      {args.checkpoint}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVP Hybrid Pipeline with Baseline Reduction")
    
    # Data
    parser.add_argument('--data_root', type=str, 
                        default=BVPConfig.DATA_ROOT,
                        help='Root directory for BVP data')
    
    # Model architecture
    parser.add_argument('--encoder_hidden', type=int, default=32,
                        help='Hidden size for encoder LSTM')
    parser.add_argument('--encoder_dropout', type=float, default=0.3,
                        help='Dropout for encoder')
    parser.add_argument('--classifier_hidden', type=int, default=64,
                        help='Hidden size for classifier BiLSTM')
    parser.add_argument('--classifier_layers', type=int, default=2,
                        help='Number of BiLSTM layers in classifier')
    parser.add_argument('--classifier_dropout', type=float, default=0.4,
                        help='Dropout for classifier')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='best_bvp_hybrid.pt',
                        help='Checkpoint file path')
    
    args = parser.parse_args()
    
    main(args)
