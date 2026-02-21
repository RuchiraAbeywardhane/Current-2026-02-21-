"""
Utilities Module
================
Training, evaluation, and visualization utilities.

Usage:
    from utils import train_model, evaluate_model, EarlyStopping
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience (int): How many epochs to wait after last improvement
        delta (float): Minimum change to qualify as an improvement
        verbose (bool): Print messages
    """
    
    def __init__(self, patience=10, delta=0.0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss, model, path='checkpoint.pt'):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {path}')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_epoch(model, dataloader, criterion, optimizer, device, is_fusion=False):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        is_fusion (bool): Whether this is a fusion model
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if is_fusion:
            eeg_x, bvp_x, labels = batch
            eeg_x = eeg_x.to(device)
            bvp_x = tuple(x.to(device) for x in bvp_x)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg_x, bvp_x)
        else:
            inputs, labels = batch
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(device) for x in inputs)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device, is_fusion=False):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        criterion: Loss function
        device: Device
        is_fusion (bool): Whether this is a fusion model
    
    Returns:
        dict: {'loss', 'accuracy', 'precision', 'recall', 'f1', 'predictions', 'labels'}
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if is_fusion:
                eeg_x, bvp_x, labels = batch
                eeg_x = eeg_x.to(device)
                bvp_x = tuple(x.to(device) for x in bvp_x)
                labels = labels.to(device)
                outputs = model(eeg_x, bvp_x)
            else:
                inputs, labels = batch
                if isinstance(inputs, tuple):
                    inputs = tuple(x.to(device) for x in inputs)
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }


def train_model(model, train_loader, val_loader, config, checkpoint_path='best_model.pt', is_fusion=False):
    """
    Complete training loop with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object with training params
        checkpoint_path (str): Path to save best model
        is_fusion (bool): Whether this is a fusion model
    
    Returns:
        dict: Training history {'train_loss', 'train_acc', 'val_loss', 'val_acc'}
    """
    device = config.DEVICE
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR if not is_fusion else config.FUSION_LR)
    early_stopping = EarlyStopping(patience=config.EEG_PATIENCE if not is_fusion else config.FUSION_PATIENCE, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    epochs = config.EEG_EPOCHS if not is_fusion else config.FUSION_EPOCHS
    
    print(f"\n{'='*80}")
    print(f"TRAINING {'FUSION' if is_fusion else 'MODEL'}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    print(f"Batch Size: {train_loader.batch_size}")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, is_fusion)
        
        # Validate
        val_results = evaluate_model(model, val_loader, criterion, device, is_fusion)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping
        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"\n✅ Training complete! Best model saved to {checkpoint_path}")
    
    return history


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): Class names
        title (str): Plot title
        save_path (str): Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history (dict): Training history from train_model
        save_path (str): Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): Class names
    """
    from sklearn.metrics import classification_report
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {per_class_acc[i]:.4f}")
    
    print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("="*80)


def save_results(results, filepath):
    """
    Save evaluation results to file.
    
    Args:
        results (dict): Results dictionary
        filepath (str): Output file path
    """
    np.savez(filepath, **results)
    print(f"Results saved to {filepath}")


def standardize_features(X_train, X_val, X_test):
    """
    Standardize features using training statistics.
    
    Args:
        X_train (np.ndarray): Training features
        X_val (np.ndarray): Validation features
        X_test (np.ndarray): Test features
    
    Returns:
        tuple: (X_train_std, X_val_std, X_test_std, mean, std)
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std
    
    return X_train_std, X_val_std, X_test_std, mean, std
