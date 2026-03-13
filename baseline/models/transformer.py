"""
Phase 2: Transformer Encoder for sequence modeling.

Replaces BiLSTM from Phase 1:
- Better parallelization (no sequential dependency)
- Global receptive field via self-attention
- Positional encoding for sequence order awareness
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""
    
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSequenceEncoder(nn.Module):
    """
    Transformer encoder for sequence modeling in OCR.
    
    Replaces BiLSTM with multi-layer Transformer encoder.
    Uses pre-norm architecture for better training stability.
    """
    
    def __init__(
        self, 
        d_model=512, 
        nhead=8, 
        num_layers=4, 
        dim_feedforward=1024,
        dropout=0.1,
        output_dim=None
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection (if needed)
        self.output_proj = None
        if output_dim is not None and output_dim != d_model:
            self.output_proj = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] sequence features
        
        Returns:
            [B, T, D] or [B, T, output_dim] encoded features
        """
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        if self.output_proj is not None:
            x = self.output_proj(x)
        
        return x
