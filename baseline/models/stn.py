"""
Spatial Transformer Network (STN) for license plate alignment.

Automatically learns to rectify perspective distortion, rotation, skewing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    """
    Spatial Transformer Network for geometric rectification.
    
    Learns affine transformation to align license plates.
    Helps handle rotation, skewing, perspective distortion.
    """
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Localization network - predicts transformation parameters
        self.localization = nn.Sequential(
            # Input: [B, 3, H, W]
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        
        # Calculate size after convolutions for adaptive pooling
        # After 3 maxpools with stride 2: H/8, W/8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Regressor for the 2x3 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 6)  # 6 params for 2x3 affine matrix
        )
        
        # Initialize identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x):
        """
        Args:
            x: Input images [B, C, H, W]
        
        Returns:
            Transformed images [B, C, H, W]
        """
        # Get transformation parameters
        xs = self.localization(x)
        xs = self.adaptive_pool(xs)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Generate sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        
        # Sample input using grid
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        return x_transformed


class IdentitySTN(nn.Module):
    """Identity STN for ablation studies - does nothing"""
    
    def forward(self, x):
        return x
