"""
    EEG Model Architecture
    ======================
    
    This module contains the BiLSTM-based model architecture for EEG emotion recognition.
    
    Features:
    - Multi-layer Bidirectional LSTM
    - Attention mechanism for temporal pooling
    - Batch normalization and dropout for regularization
    
    Author: Final Year Project
    Date: 2026
"""

import torch
import torch.nn as nn


class SimpleBiLSTMClassifier(nn.Module):
    """3-layer BiLSTM with attention for EEG emotion recognition."""
    
    def __init__(self, dx=26, n_channels=4, hidden=256, layers=3, n_classes=4, p_drop=0.4):
        """
        Initialize BiLSTM classifier.
        
        Args:
            dx: Number of features per channel (default: 26)
            n_channels: Number of EEG channels (default: 4 for MUSE)
            hidden: Hidden dimension size (default: 256)
            layers: Number of LSTM layers (default: 3)
            n_classes: Number of output classes (default: 4)
            p_drop: Dropout probability (default: 0.4)
        """
        super().__init__()
        self.n_channels = n_channels
        self.hidden = hidden
        
        # Input projection layer
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
        
        # Layer normalization and dropout
        d_lstm = 2 * hidden
        self.norm = nn.LayerNorm(d_lstm)
        self.drop = nn.Dropout(p_drop)

        # Attention mechanism
        self.attn = nn.Sequential(
            nn.Linear(d_lstm, d_lstm // 2),
            nn.Tanh(),
            nn.Linear(d_lstm // 2, 1)
        )
        
        # Classification head
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
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, dx)
        
        Returns:
            logits: Output tensor of shape (batch_size, n_classes)
        """
        B, C, dx = x.shape
        
        # Project input features
        x = self.input_proj(x)
        
        # BiLSTM encoding
        h, _ = self.lstm(x)
        h = self.drop(self.norm(h))

        # Attention pooling
        scores = self.attn(h)
        alpha = torch.softmax(scores, dim=1)
        h_pooled = (alpha * h).sum(dim=1)

        # Classification
        return self.classifier(h_pooled)
