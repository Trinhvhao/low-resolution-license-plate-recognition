"""
Phase 2: Multi-Head Temporal Attention Fusion.

Upgrades from Phase 1 AttentionFusion:
- Multi-head attention instead of single-head score network
- Cross-frame attention for capturing inter-frame relationships
- Learnable quality-aware weighting per frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadTemporalFusion(nn.Module):
    """
    Multi-head attention-based temporal fusion across multiple frames.
    
    Uses self-attention to model inter-frame relationships and learn
    which frames contain the most useful features for recognition.
    """
    
    def __init__(self, channels, num_heads=8, num_frames=5, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_frames = num_frames
        
        # Frame quality estimator (per-frame global quality score)
        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )
        
        # Multi-head self-attention across frames
        self.frame_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm for attention output
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B*T, C, H, W] (T frames per sample)
        
        Returns:
            Fused tensor [B, C, W]
        """
        bt, c, h, w = x.size()
        b = bt // self.num_frames
        t = self.num_frames
        
        # 1. Pool height dimension
        if h > 1:
            x = F.adaptive_avg_pool2d(x, (1, w))  # [B*T, C, 1, W]
        x = x.squeeze(2)  # [B*T, C, W]
        
        # 2. Compute quality scores for each frame
        # Use the feature map before height pooling
        x_4d = x.unsqueeze(2)  # [B*T, C, 1, W] for quality_net
        quality = self.quality_net(x_4d)  # [B*T, 1]
        quality = quality.view(b, t, 1)  # [B, T, 1]
        
        # 3. Reshape for multi-head attention: process each spatial position
        x = x.view(b, t, c, w)  # [B, T, C, W]
        
        # Process each spatial position with frame attention
        x = x.permute(0, 3, 1, 2)  # [B, W, T, C]
        x_flat = x.reshape(b * w, t, c)  # [B*W, T, C]
        
        # Self-attention across frames
        attn_out, _ = self.frame_attention(x_flat, x_flat, x_flat)  # [B*W, T, C]
        attn_out = self.norm1(attn_out + x_flat)  # residual + norm
        
        # FFN
        ffn_out = self.ffn(attn_out)
        attn_out = self.norm2(attn_out + ffn_out)  # [B*W, T, C]
        
        # Reshape back
        attn_out = attn_out.view(b, w, t, c)  # [B, W, T, C]
        attn_out = attn_out.permute(0, 2, 3, 1)  # [B, T, C, W]
        
        # 4. Quality-weighted aggregation across frames
        quality_weights = F.softmax(quality, dim=1)  # [B, T, 1]
        quality_weights = quality_weights.unsqueeze(-1)  # [B, T, 1, 1]
        
        fused = (attn_out * quality_weights).sum(dim=1)  # [B, C, W]
        
        return fused
