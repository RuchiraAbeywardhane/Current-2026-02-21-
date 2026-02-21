"""
PyTorch Datasets Module
========================
Custom Dataset classes for EEG, BVP, and multimodal fusion.

Usage:
    from datasets import EEGWindowDataset, BVPWindowDataset, MultimodalFusionDataset
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class EEGWindowDataset(Dataset):
    """
    PyTorch Dataset for EEG windows.
    
    Args:
        X (np.ndarray): EEG features (N, C, dx)
        y (np.ndarray): Labels (N,)
        clip_names (np.ndarray): Clip names for each window (N,)
    """
    
    def __init__(self, X, y, clip_names=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.clip_names = clip_names if clip_names is not None else np.arange(len(X))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_clip_name(self, idx):
        """Get clip name for a given index."""
        return self.clip_names[idx]


class BVPWindowDataset(Dataset):
    """
    PyTorch Dataset for BVP windows with multi-scale inputs.
    
    Args:
        X1 (np.ndarray): Original signal (N, 1, T)
        X2 (np.ndarray): Smoothed signal (N, 1, T)
        X3 (np.ndarray): Downsampled signal (N, 1, T//2)
        feats (np.ndarray): Handcrafted features (N, feat_dim)
        y (np.ndarray): Labels (N,)
        clip_names (np.ndarray): Clip names for each window (N,)
    """
    
    def __init__(self, X1, X2, X3, feats, y, clip_names=None):
        self.X1 = torch.from_numpy(X1).float()
        self.X2 = torch.from_numpy(X2).float()
        self.X3 = torch.from_numpy(X3).float()
        self.feats = torch.from_numpy(feats).float()
        self.y = torch.from_numpy(y).long()
        self.clip_names = clip_names if clip_names is not None else np.arange(len(y))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X1[idx], self.X2[idx], self.X3[idx], self.feats[idx]), self.y[idx]
    
    def get_clip_name(self, idx):
        """Get clip name for a given index."""
        return self.clip_names[idx]


class MultimodalFusionDataset(Dataset):
    """
    PyTorch Dataset for multimodal EEG+BVP fusion.
    
    Aligns EEG and BVP windows by clip name and ensures consistent labeling.
    
    Args:
        eeg_dataset (EEGWindowDataset): EEG dataset
        bvp_dataset (BVPWindowDataset): BVP dataset
        common_clips (list): List of clip names common to both modalities
    """
    
    def __init__(self, eeg_dataset, bvp_dataset, common_clips):
        self.samples = []
        
        # Create clip-to-index mappings
        eeg_clip_to_idx = {clip: [] for clip in common_clips}
        bvp_clip_to_idx = {clip: [] for clip in common_clips}
        
        for i in range(len(eeg_dataset)):
            clip = eeg_dataset.get_clip_name(i)
            if clip in eeg_clip_to_idx:
                eeg_clip_to_idx[clip].append(i)
        
        for i in range(len(bvp_dataset)):
            clip = bvp_dataset.get_clip_name(i)
            if clip in bvp_clip_to_idx:
                bvp_clip_to_idx[clip].append(i)
        
        # Align windows by clip
        for clip in common_clips:
            eeg_indices = eeg_clip_to_idx.get(clip, [])
            bvp_indices = bvp_clip_to_idx.get(clip, [])
            
            if not eeg_indices or not bvp_indices:
                continue
            
            # Take minimum number of windows
            min_windows = min(len(eeg_indices), len(bvp_indices))
            
            for i in range(min_windows):
                eeg_idx = eeg_indices[i]
                bvp_idx = bvp_indices[i]
                
                # Get data
                eeg_x, eeg_y = eeg_dataset[eeg_idx]
                bvp_x, bvp_y = bvp_dataset[bvp_idx]
                
                # Verify labels match
                if eeg_y.item() != bvp_y.item():
                    print(f"⚠️ WARNING: Label mismatch for clip {clip}: EEG={eeg_y.item()}, BVP={bvp_y.item()}")
                
                self.samples.append((eeg_x, bvp_x, eeg_y, clip))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        eeg_x, bvp_x, y, clip = self.samples[idx]
        return eeg_x, bvp_x, y


class ClipAggregationDataset(Dataset):
    """
    Dataset wrapper for clip-level predictions (aggregating window predictions).
    
    Args:
        base_dataset: Base window-level dataset
        aggregation_method (str): "mean", "max", or "voting"
    """
    
    def __init__(self, base_dataset, aggregation_method="mean"):
        self.base_dataset = base_dataset
        self.aggregation_method = aggregation_method
        
        # Group windows by clip
        self.clip_to_windows = {}
        for i in range(len(base_dataset)):
            clip = base_dataset.get_clip_name(i)
            if clip not in self.clip_to_windows:
                self.clip_to_windows[clip] = []
            self.clip_to_windows[clip].append(i)
        
        self.clips = list(self.clip_to_windows.keys())
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        """
        Returns all windows for a clip.
        
        Returns:
            tuple: (windows, labels, clip_name)
                windows: List of window tensors
                labels: List of label tensors
                clip_name: Clip identifier
        """
        clip = self.clips[idx]
        window_indices = self.clip_to_windows[clip]
        
        windows = []
        labels = []
        
        for i in window_indices:
            x, y = self.base_dataset[i]
            windows.append(x)
            labels.append(y)
        
        return windows, labels, clip


class AugmentedEEGDataset(Dataset):
    """
    EEG Dataset with on-the-fly augmentation (e.g., Mixup).
    
    Args:
        base_dataset (EEGWindowDataset): Base EEG dataset
        mixup_alpha (float): Mixup interpolation parameter (0 = no mixup)
    """
    
    def __init__(self, base_dataset, mixup_alpha=0.0):
        self.base_dataset = base_dataset
        self.mixup_alpha = mixup_alpha
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        
        if self.mixup_alpha > 0 and np.random.rand() < 0.5:
            # Apply mixup
            idx2 = np.random.randint(len(self.base_dataset))
            x2, y2 = self.base_dataset[idx2]
            
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = lam * x + (1 - lam) * x2
            
            # Return mixed labels (for loss computation)
            return x, (y, y2, lam)
        
        return x, y


def create_data_loaders(train_dataset, val_dataset, test_dataset, 
                       batch_size=64, num_workers=0, pin_memory=True):
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        pin_memory (bool): Pin memory for faster GPU transfer
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
