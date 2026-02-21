"""
EEG Model Architecture
======================
BiLSTM with attention mechanism for EEG emotion recognition.

Usage:
    from models_eeg import SimpleBiLSTMClassifier
    
    model = SimpleBiLSTMClassifier(
        dx=26,           # Features per channel
        n_channels=4,    # Number of EEG channels
        hidden=256,      # Hidden dimension
        layers=3,        # Number of LSTM layers
        n_classes=4,     # Number of emotion classes
        p_drop=0.4       # Dropout rate
    )
"""

import torch
import torch.nn as nn


class SimpleBiLSTMClassifier(nn.Module):
    """
    3-layer BiLSTM with attention for EEG emotion recognition.
    
    Architecture:
    1. Input projection (Linear + BatchNorm + ReLU + Dropout)
    2. Bidirectional LSTM (3 layers)
    3. Layer normalization + Dropout
    4. Attention mechanism (Tanh-based)
    5. Classifier head (3 FC layers with BatchNorm)
    
    Args:
        dx (int): Number of features per channel (default: 26)
        n_channels (int): Number of EEG channels (default: 4)
        hidden (int): Hidden dimension size (default: 256)
        layers (int): Number of LSTM layers (default: 3)
        n_classes (int): Number of output classes (default: 4)
        p_drop (float): Dropout probability (default: 0.4)
    """
    
    def __init__(self, dx=26, n_channels=4, hidden=256, layers=3, n_classes=4, p_drop=0.4):
        super().__init__()
        self.n_channels = n_channels
        self.hidden = hidden
        
        # Input projection: (B, C, dx) -> (B, C, hidden)
        self.input_proj = nn.Sequential(
            nn.Linear(dx, hidden),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
            nn.Dropout(p_drop * 0.5)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=p_drop if layers > 1 else 0
        )
        
        # Post-LSTM normalization
        d_lstm = 2 * hidden  # Bidirectional doubles the hidden size
        self.norm = nn.LayerNorm(d_lstm)
        self.drop = nn.Dropout(p_drop)

        # Attention mechanism
        self.attn = nn.Sequential(
            nn.Linear(d_lstm, d_lstm // 2),
            nn.Tanh(),
            nn.Linear(d_lstm // 2, 1)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_lstm, d_lstm),
            nn.BatchNorm1d(d_lstm),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_lstm, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, dx)
                B = batch size
                C = number of channels
                dx = number of features per channel
        
        Returns:
            torch.Tensor: Output logits of shape (B, n_classes)
        """
        B, C, dx = x.shape
        
        # Project input features
        x = self.input_proj(x)  # (B, C, hidden)
        
        # LSTM encoding
        h, _ = self.lstm(x)  # (B, C, 2*hidden)
        h = self.drop(self.norm(h))

        # Attention pooling
        scores = self.attn(h)  # (B, C, 1)
        alpha = torch.softmax(scores, dim=1)  # (B, C, 1)
        h_pooled = (alpha * h).sum(dim=1)  # (B, 2*hidden)

        # Classification
        return self.classifier(h_pooled)  # (B, n_classes)
    
    def get_encoder(self):
        """
        Returns the encoder part (without classifier) for fusion.
        
        Returns:
            nn.Module: Encoder module
        """
        encoder = nn.Sequential(
            self.input_proj,
            nn.ModuleDict({
                'lstm': self.lstm,
                'norm': self.norm,
                'drop': self.drop,
                'attn': self.attn
            })
        )
        return encoder


class EEGEncoder(nn.Module):
    """
    EEG encoder for fusion (without classifier head).
    
    This is used when you want to extract features from EEG
    without classification, typically for multimodal fusion.
    """
    
    def __init__(self, dx=26, n_channels=4, hidden=256, layers=3, p_drop=0.4):
        super().__init__()
        self.n_channels = n_channels
        self.hidden = hidden
        
        self.input_proj = nn.Sequential(
            nn.Linear(dx, hidden),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
            nn.Dropout(p_drop * 0.5)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=p_drop if layers > 1 else 0
        )
        
        d_lstm = 2 * hidden
        self.norm = nn.LayerNorm(d_lstm)
        self.drop = nn.Dropout(p_drop)

        self.attn = nn.Sequential(
            nn.Linear(d_lstm, d_lstm // 2),
            nn.Tanh(),
            nn.Linear(d_lstm // 2, 1)
        )
        
        self.out_dim = d_lstm  # Output dimension for fusion
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass (returns features, not logits).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, dx)
        
        Returns:
            torch.Tensor: Feature vector of shape (B, 2*hidden)
        """
        B, C, dx = x.shape
        x = self.input_proj(x)
        h, _ = self.lstm(x)
        h = self.drop(self.norm(h))
        scores = self.attn(h)
        alpha = torch.softmax(scores, dim=1)
        h_pooled = (alpha * h).sum(dim=1)
        return h_pooled


def load_eeg_classifier(checkpoint_path, device='cuda'):
    """
    Load a trained EEG classifier from checkpoint.
    
    Args:
        checkpoint_path (str): Path to saved model checkpoint
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        SimpleBiLSTMClassifier: Loaded model
    """
    model = SimpleBiLSTMClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def convert_classifier_to_encoder(classifier_model):
    """
    Convert a trained classifier to an encoder (remove classifier head).
    
    Args:
        classifier_model (SimpleBiLSTMClassifier): Trained classifier
    
    Returns:
        SimpleBiLSTMClassifier: Model with Identity classifier (acts as encoder)
    """
    # Replace classifier with Identity
    classifier_model.classifier = nn.Identity()
    
    # Freeze parameters
    for param in classifier_model.parameters():
        param.requires_grad = False
    
    classifier_model.eval()
    return classifier_model
