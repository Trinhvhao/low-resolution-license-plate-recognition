"""
SR Enhancement Front-end — Lightweight image enhancer with Channel Attention.

PURPOSE: Runs at BOTH training AND inference time.
    - Training: learns to enhance LR → closer to HR quality (supervised by HR targets)
    - Inference: automatically enhances test LR images before recognition

Unlike AuxSRBranch (training-only backbone regularizer), this module:
    1. Sits BEFORE STN/backbone (enhances raw pixel input)
    2. Runs at inference (improves actual recognition)
    3. Uses residual learning (output = input + learned_enhancement)
    4. Very lightweight (~450K params, adds <10ms per sample)

Architecture:
    Input: [B, 3, H, W]  (resized LR frame, typically 64×224)
    → Conv 3×3 → channels → ReLU  (shallow feature extraction)
    → N × CAResBlock (Channel Attention ResBlock)
    → Conv 3×3 → 3ch  (reconstruction)
    → + Input  (global residual skip connection)
    → Output: [B, 3, H, W]  (enhanced frame)

Key design choices:
    - Channel Attention (SE-style): focuses on feature channels containing 
      high-frequency info (edges, character strokes)
    - Residual learning: network only needs to learn the "delta" 
      (enhancement amount), not the full image reconstruction
    - Global residual skip: ensures output is always close to input,
      prevents hallucination of wrong characters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention.

    Learns which feature channels are important for each input frame.
    Focuses on channels containing high-frequency details (edges, text strokes).

    Args:
        channels:  number of feature channels
        reduction: reduction ratio for SE bottleneck (default: 8)
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: [B, C, H, W] → [B, C]
        y = self.squeeze(x).view(b, c)
        # Excitation: [B, C] → [B, C] (channel weights)
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: modulate features by channel importance
        return x * y


class CAResBlock(nn.Module):
    """
    Residual Block with Channel Attention.

    Conv → ReLU → Conv → ChannelAttention → + input

    Args:
        channels: number of feature channels
    """

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return x + res


class SREnhancer(nn.Module):
    """
    Lightweight per-frame image enhancer with Channel Attention ResBlocks.

    Pipeline:
        1. Shallow feature extraction: Conv 3×3 (3 → channels)
        2. Deep feature extraction: N × CAResBlock
        3. Reconstruction: Conv 3×3 (channels → 3)
        4. Global residual: output = input + reconstruction

    Only learns the "enhancement delta" — input passes through unchanged
    when the network outputs zeros (identity-initialized).

    Args:
        in_channels:  input channels (default: 3 for RGB)
        channels:     intermediate feature channels (default: 64)
        num_blocks:   number of CAResBlocks (default: 6)
    """

    def __init__(self, in_channels=3, channels=64, num_blocks=6):
        super().__init__()

        # Shallow feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Deep feature extraction with Channel Attention
        self.body = nn.Sequential(
            *[CAResBlock(channels) for _ in range(num_blocks)]
        )

        # Reconstruction (back to image space)
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, in_channels, kernel_size=3, padding=1),
        )

        # Initialize tail to near-zero for stable start (residual = ~0)
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] — input LR frame (already resized to model input)

        Returns:
            enhanced: [B, 3, H, W] — enhanced frame (same resolution)
        """
        shallow = self.head(x)
        deep = self.body(shallow)
        residual = self.tail(deep)
        return x + residual  # global residual skip
