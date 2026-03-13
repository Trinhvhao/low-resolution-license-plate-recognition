"""
2.3 Auxiliary Super-Resolution Branch (training only).

During training, forces backbone to learn fine-grained texture features
by reconstructing HR images from low-resolution multi-frame input.

Architecture:
    backbone features [B, 512, H', W']
    → ConvTranspose2d upsampling (progressive ×2 stages)
    → [B, 3, H, W]   (original input resolution, e.g. 48×160)

Loss: SRLoss = λ_l1 × L1 + λ_perc × PerceptualLoss

Usage:
    # Training
    preds, sr_out = model(images, return_sr=True)
    ctc_loss = ctc_criterion(preds, ...)
    sr_loss  = sr_criterion(sr_out, hr_images)
    loss = ctc_loss + λ * sr_loss

    # Inference — no SR overhead
    preds = model(images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual Loss (VGG16 relu2_2 features, fixed)
# ─────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 relu2_2 features.
    Parameters are frozen – used only for computing similarity.
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        # relu2_2 = first 9 layers (0-based indices 0..8)
        self.features = nn.Sequential(*list(vgg.features)[:9])
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization for VGG input
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        """Normalize to ImageNet stats. Assumes x ∈ [0, 1]."""
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, 3, H, W]  – SR output
            target: [B, 3, H, W]  – HR ground-truth images (values ∈ [0, 1])
        Returns:
            scalar perceptual loss
        """
        f_pred   = self.features(self.normalize(pred.clamp(0, 1)))
        f_target = self.features(self.normalize(target.clamp(0, 1)))
        return F.mse_loss(f_pred, f_target)


# ─────────────────────────────────────────────────────────────────────────────
# SR Loss  (L1 + Perceptual)
# ─────────────────────────────────────────────────────────────────────────────

class SRLoss(nn.Module):
    """
    Combined SR loss: L1 + λ_perceptual × PerceptualLoss.

    Args:
        lambda_l1:          weight for pixel-wise L1 loss
        lambda_perceptual:  weight for VGG perceptual loss
    """

    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.perceptual = PerceptualLoss()

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, 3, H, W]  SR prediction
            target: [B, 3, H, W]  HR ground-truth (values ∈ [0, 1])
        Returns:
            total_loss, (l1_loss, perceptual_loss)
        """
        l1_loss   = F.l1_loss(pred, target)
        perc_loss = self.perceptual(pred, target)
        total     = self.lambda_l1 * l1_loss + self.lambda_perceptual * perc_loss
        return total, (l1_loss.item(), perc_loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary SR Branch (decoder)
# ─────────────────────────────────────────────────────────────────────────────

class AuxSRBranch(nn.Module):
    """
    Lightweight progressive upsampling decoder.

    Given backbone features [B, in_channels, H', W'] (e.g. [B, 512, 6, 20])
    produces [B, 3, target_h, target_w] (e.g. [B, 3, 48, 160]).

    The decoder uses bilinear interpolation + conv (sub-pixel-free, robust):
        512 → 256 (×2 upsample)
        256 → 128 (×2 upsample)
        128 →  64 (×2 upsample)
         64 →   3 (final 1×1 conv + Sigmoid)

    Note: TRAINING ONLY.  Set model.eval() or pass return_sr=False to skip.
    """

    def __init__(self, in_channels=512, target_h=43, target_w=120):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w

        def up_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.up1 = up_block(512, 256)
        self.up2 = up_block(256, 128)
        self.up3 = up_block(128,  64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()              # output ∈ [0, 1] for L1 / VGG loss
        )

    def forward(self, feat):
        """
        Args:
            feat: [B, 512, H', W']
        Returns:
            sr:   [B, 3, target_h, target_w]
        """
        x = feat
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(x)

        # Final resize to exact target resolution
        x = F.interpolate(x, size=(self.target_h, self.target_w),
                          mode='bilinear', align_corners=False)
        return self.final(x)
