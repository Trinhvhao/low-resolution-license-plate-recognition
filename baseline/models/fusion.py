"""
Fusion modules for multi-frame license plate recognition.

2.1 Spatial-Temporal Attention Fusion (Phase 2):
    - Channel attention (SE-style): which features are important per frame
    - Spatial attention: which spatial locations are sharp/clear per frame
    - Temporal attention: position-aware frame selection
      → solves "frame 1 clear on left, frame 3 clear on right"

AttentionFusion: original Phase 1 (kept for backward compat)
SpatialTemporalAttentionFusion: Phase 2 upgrade
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """Original Phase 1 attention-based temporal fusion. Kept for backward compat."""
    def __init__(self, channels):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x):
        b_frames, c, h, w = x.size()
        b_size = b_frames // 5
        if h > 1:
            x = F.adaptive_avg_pool2d(x, (1, w))
        x = x.squeeze(2)
        x_view = x.view(b_size, 5, c, w)
        x_4d = x.unsqueeze(2)
        scores = self.score_net(x_4d).view(b_size, 5, 1, w)
        return torch.sum(x_view * F.softmax(scores, dim=1), dim=1)


class SpatialTemporalAttentionFusion(nn.Module):
    """
    Phase 2: Spatial-Temporal Attention Fusion.

    Three-stage fusion strategy:
      1. Channel Attention (Squeeze-Excitation per frame)
      2. Spatial Attention (sharpness/quality score per H×W location)
      3. Temporal Attention (position-aware frame weighting at each W column)

    Handles scenario: "frame 1 clear on left, frame 3 clear on right"
    by computing independent frame weights at each horizontal position.

    Args:
        channels:   backbone output channels (default 512)
        num_frames: frames per track (default 5)
        reduction:  SE reduction ratio (default 16)
        dropout:    dropout rate (default 0.1)
    """

    def __init__(self, channels=512, num_frames=5, reduction=16, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels
        mid = max(channels // reduction, 8)

        # Stage 1: Channel Attention (SE block)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

        # Stage 2: Spatial Attention (sharpness score per H,W cell)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1)
        )

        # Stage 3: Temporal Attention (per-W-position frame weighting)
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, 1, kernel_size=1)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Args:
            x: [B*T, C, H, W]
        Returns:
            fused: [B, C, W]
        """
        bt, c, h, w = x.shape
        b = bt // self.num_frames
        t = self.num_frames

        # Stage 1: Channel Attention
        ca = self.channel_attn(x)                          # [B*T, C]
        x = x * ca.view(bt, c, 1, 1)                      # [B*T, C, H, W]

        # Stage 2: Spatial Attention
        sa = torch.sigmoid(self.spatial_attn(x))           # [B*T, 1, H, W]
        x = x * sa                                         # [B*T, C, H, W]

        # Height pooling -> sequence
        x = F.adaptive_avg_pool2d(x, (1, w)).squeeze(2)   # [B*T, C, W]

        # Stage 3: Temporal Attention (position-aware)
        ta_scores = self.temporal_attn(x)                  # [B*T, 1, W]
        ta_scores = ta_scores.view(b, t, 1, w)             # [B, T, 1, W]
        ta_weights = F.softmax(ta_scores, dim=1)           # softmax over T frames

        x_frames = x.view(b, t, c, w)                     # [B, T, C, W]
        fused = (x_frames * ta_weights).sum(dim=1)         # [B, C, W]

        # LayerNorm + Dropout
        fused = self.norm(fused.permute(0, 2, 1)).permute(0, 2, 1)
        fused = self.dropout(fused)

        return fused
