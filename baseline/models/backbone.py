"""
ResNet-34 Backbone adapted for License Plate Recognition.

Uses pretrained ResNet-34 from torchvision, adapted for plate aspect ratio.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet34Backbone(nn.Module):
    """
    ResNet-34 backbone for feature extraction from license plates.
    
    Modifications:
    - Remove avgpool and fc layers
    - Adapt first conv for plate images (small height)
    - Output feature maps suitable for sequence modeling
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet-34
        resnet = models.resnet34(pretrained=pretrained)
        
        # Keep only feature extraction layers (remove avgpool, fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # MODIFIED: Use stride=1 instead of stride=2 to preserve spatial dimensions
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.layer1 = resnet.layer1  # Output: 64 channels
        self.layer2 = resnet.layer2  # Output: 128 channels
        self.layer3 = resnet.layer3  # Output: 256 channels
        # MODIFIED: Skip layer4 to prevent too much downsampling
        # With layer4, total downsampling = 32x (too much for 48x160 input)
        # Without layer4, downsampling = 16x (better for sequence length)
        
        # Add adaptation layer to match expected 512 channels
        self.adapt = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Feature maps [B, 512, H', W'] where H'~H/8, W'~W/8 (reduced downsampling)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # stride=1, so H,W preserved
        
        x = self.layer1(x)  # /2
        x = self.layer2(x)  # /4
        x = self.layer3(x)  # /8
        # Skip layer4 to avoid /16 downsampling
        x = self.adapt(x)  # 256 -> 512 channels
        
        return x
    
    @property
    def output_channels(self):
        """Number of output channels"""
        return 512


class VanillaCNN(nn.Module):
    """
    Original vanilla CNN backbone (for comparison/fallback).
    Same as in the original MultiFrameCRNN.
    """
    
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.cnn(x)
    
    @property
    def output_channels(self):
        return 512
