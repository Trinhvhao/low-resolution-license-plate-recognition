"""
Phase 7 backbone: ResNet50 with controlled downsampling.

ResNet50 uses Bottleneck blocks (3-layer) vs ResNet34's BasicBlock (2-layer).
More capacity to learn subtle character differences (8↔6, V↔Y, etc.).

Key changes from ResNet34Backbone:
  - ResNet50 Bottleneck blocks → 4× more channels per layer
  - layer3 outputs 1024ch (vs 256ch in ResNet34)
  - adapt: 1024 → 512 (heavier projection for richer features)
  - Same downsampling strategy: skip layer4 to preserve spatial resolution

Output: [B, 512, H/8, W/8] — identical spatial dims to ResNet34Backbone.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone for license plate feature extraction.

    Compared to ResNet34Backbone:
      - Bottleneck blocks give deeper per-layer representation
      - layer3 has 1024 channels (vs 256)
      - Stronger 1024→512 adaptation layer
      - ~2× more total parameters in backbone

    Same downsampling scheme as ResNet34:
      - maxpool stride=1 (preserve spatial)
      - Skip layer4 (avoid /16 downsampling)
      - Final output: [B, 512, H/8, W/8]
    """

    def __init__(self, pretrained=True):
        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Input layers
        self.conv1 = resnet.conv1      # 3 → 64, stride=2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # stride=1 instead of 2 to preserve spatial dims
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = resnet.layer1    # 64 → 256 (Bottleneck)
        self.layer2 = resnet.layer2    # 256 → 512 (Bottleneck), stride=2
        self.layer3 = resnet.layer3    # 512 → 1024 (Bottleneck), stride=2

        # Skip layer4 to prevent /16 downsampling (same as ResNet34)
        # Total downsampling: conv1(/2) × layer2(/2) × layer3(/2) = /8

        # Richer adaptation: 1024 → 512 with residual-style projection
        self.adapt = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),   # 1×1 channel reduction
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),  # 3×3 for local context
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]  (e.g., 64×224)
        Returns:
            [B, 512, H/8, W/8]  (e.g., 8×28)
        """
        x = self.conv1(x)       # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # stride=1, no downsampling

        x = self.layer1(x)      # same spatial
        x = self.layer2(x)      # /2  → total /4
        x = self.layer3(x)      # /2  → total /8

        x = self.adapt(x)       # 1024 → 512

        return x

    @property
    def output_channels(self):
        return 512
