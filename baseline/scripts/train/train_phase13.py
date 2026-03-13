"""
Phase 13 — SR-Enhanced Recognition Pipeline.

╔══════════════════════════════════════════════════════════════════════════╗
║  WHY THIS PHASE EXISTS                                                  ║
║                                                                          ║
║  Root cause of plateau at ~79%: LR images are 17×48 pixels.             ║
║  Naive resize to 64×224 = very blurry. The backbone tries to recognize  ║
║  characters from blurry input — no amount of LR tuning can fix this.    ║
║                                                                          ║
║  Phase 8-12 all fine-tuned the SAME architecture with different LR      ║
║  schedules. Results: 79.15% → 79.33% → 79.22% → ~79%. Plateaued.      ║
║                                                                          ║
║  FIX: Add SREnhancer BEFORE backbone. Learns to denoise/sharpen LR     ║
║  frames using HR supervision. Runs at BOTH training AND inference.      ║
║                                                                          ║
║  Current: LR(17×48) → resize(64×224) → STN → Backbone → BiLSTM → CTC  ║
║  Phase 13: LR(17×48) → resize(64×224) → SREnhancer → STN → Backbone   ║
║            → BiLSTM → CTC  + SR loss (enhanced vs HR)                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Pipeline:
    ┌─────────┐   ┌─────────────┐   ┌─────┐   ┌──────────┐   ┌────────┐   ┌─────┐
    │ LR×5    │→  │ SREnhancer  │→  │ STN │→  │ ResNet34 │→  │ Fusion │→  │LSTM │→ CTC
    │ frames  │   │ (inference) │   │     │   │ backbone │   │        │   │     │
    └─────────┘   └──────┬──────┘   └─────┘   └────┬─────┘   └────────┘   └─────┘
                         │                          │
                    SR front loss              SR branch loss
                   (enhanced vs HR)         (training-only regularizer)

Losses:
    L_total = L_CTC + λ_sr_front × L_SR_front + λ_sr_branch × L_SR_branch
"""

import os
import sys
import math
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from dataset import AdvancedMultiFrameDataset
from utils import (decode_predictions, calculate_accuracy, calculate_cer,
                   get_prediction_confidence, calculate_confidence_gap,
                   seed_everything)
from models.recognizer_v3 import Phase3Recognizer
from models.sr_branch import SRLoss
from models.sr_frontend import SREnhancer

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════
IMG_HEIGHT = 64
IMG_WIDTH  = 224

CFG = {
    'epochs':                50,
    'batch_size':            16,
    'gradient_accumulation': 6,         # eff batch = 96

    # Discriminative LR (4 groups now)
    'lr_sr_frontend':        5e-4,      # NEW module — needs faster learning
    'lr_backbone':           5e-6,      # pretrained — very conservative
    'lr_lstm':               5e-5,      # pretrained — moderate
    'lr_head':               1e-4,      # pretrained — moderate-high
    'lr_min':                1e-7,
    'weight_decay':          1e-4,
    'max_grad_norm':         1.0,

    # SR Front-end config
    'sr_frontend_channels':  64,        # feature channels in SREnhancer
    'sr_frontend_blocks':    6,         # number of CAResBlocks
    'lambda_sr_frontend':    0.5,       # weight for SR front-end loss
    'lambda_sr_frontend_perceptual': 0.1,  # perceptual component

    # BiLSTM
    'hidden_size':           256,
    'num_lstm_layers':       2,
    'dropout':               0.25,

    # Fusion
    'fusion_reduction':      16,

    # SR branch (training-only backbone regularizer — keep from P10)
    'use_sr_branch':         True,
    'sr_target_h':           43,
    'sr_target_w':           120,
    'lambda_sr_branch':      0.1,
    'lambda_perceptual':     0.1,

    # Focal CTC
    'focal_ctc_gamma':       2.0,

    # Fine-tune from Phase 10 best (79.33%)
    'pretrained_path':       'checkpoints/best_model_phase10.pth',

    # Early stopping
    'patience':              30,

    # Regularization
    'label_smoothing':       0.05,

    # SWA config
    'swa_start_epoch':       15,
    'swa_interval':          3,

    # TTA config (6 views)
    'use_tta':               True,
    'tta_n_augs':            6,

    # Workers
    'num_workers':           4,

    # Save
    'model_save_name':       'checkpoints/best_model_phase13.pth',
    'swa_model_save_name':   'checkpoints/best_model_phase13_swa.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Phase13Model — SR-Enhanced Recognizer
# ════════════════════════════════════════════════════════════════════════════
class Phase13Model(nn.Module):
    """
    Wraps Phase3Recognizer with SREnhancer front-end.

    SREnhancer sits BEFORE everything else:
        SREnhancer → STN → Backbone → Fusion → BiLSTM → CTC

    SR front-end runs at BOTH training and inference.
    """

    def __init__(self, base_model, sr_channels=64, sr_blocks=6):
        super().__init__()
        self.sr_frontend = SREnhancer(
            in_channels=3,
            channels=sr_channels,
            num_blocks=sr_blocks,
        )
        self.base = base_model

    def forward(self, x, return_sr=False):
        """
        Args:
            x:         [B, T, C, H, W] — T=5 LR frames (resized)
            return_sr: if True, also return SR branch output (training)

        Returns:
            log_probs: [B, W', num_classes]
            sr_branch_out: [B, 3, sr_H, sr_W] (only if return_sr)
        """
        b, t, c, h, w = x.size()

        # Enhance each frame with SR front-end
        x_flat = x.view(b * t, c, h, w)
        x_enhanced = self.sr_frontend(x_flat)   # [B*T, 3, H, W]
        x_5d = x_enhanced.view(b, t, c, h, w)

        # Pass enhanced frames through base recognizer
        if return_sr:
            return self.base(x_5d, return_sr=True)
        return self.base(x_5d)

    def enhance_frames(self, x):
        """
        Enhance frames only (for SR front-end loss computation).

        Args:
            x: [B, C, H, W] — clean/raw frame(s)

        Returns:
            enhanced: [B, C, H, W]
        """
        return self.sr_frontend(x)

    @property
    def sr_branch(self):
        return self.base.sr_branch

    def get_model_info(self):
        info = self.base.get_model_info()
        info['sr_frontend'] = {
            'channels': self.sr_frontend.head[0].out_channels,
            'num_blocks': len(self.sr_frontend.body),
        }
        return info


# ════════════════════════════════════════════════════════════════════════════
# Augmentation
# ════════════════════════════════════════════════════════════════════════════
def get_train_transforms():
    """Conservative augmentation for fine-tuning."""
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        # Gentle geometric
        A.Affine(scale=(0.92, 1.08), translate_percent=(0.06, 0.06),
                 rotate=(-6, 6), shear=(-4, 4), p=0.5, fill=128),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        # Color
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=20, p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=3.0, p=1.0),
            A.Equalize(p=1.0),
        ], p=0.2),
        # Blur/Noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.4),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        # Occlusion
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(3, 12),
                        hole_width_range=(4, 16), fill=128, p=0.3),
        A.ImageCompression(quality_range=(30, 70), p=0.2),
        # Normalize
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_clean_transforms():
    """Clean transform for SR loss — only resize and normalize."""
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_tta_transforms():
    """6-view TTA for inference."""
    return [
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.RandomBrightnessContrast(brightness_limit=(0.08, 0.12),
                                       contrast_limit=(0.05, 0.08), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.RandomBrightnessContrast(brightness_limit=(-0.12, -0.08),
                                       contrast_limit=(0.05, 0.08), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.9, 1.1), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.RandomBrightnessContrast(brightness_limit=0,
                                       contrast_limit=(0.1, 0.15), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
    ]


def get_degradation_transforms():
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.5), p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(std_range=(0.03, 0.15), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), p=1.0),
        ], p=0.8),
        A.ImageCompression(quality_range=(10, 40), p=0.5),
        A.Downscale(scale_range=(0.2, 0.5), p=0.5),
    ])


# ════════════════════════════════════════════════════════════════════════════
# Dataset — returns BOTH augmented frames AND clean LR/HR for SR loss
# ════════════════════════════════════════════════════════════════════════════
class Phase13Dataset(Dataset):
    """
    Dataset that provides:
    1. Augmented LR frames (5 frames) → for CTC recognition
    2. Clean LR frame → for SR front-end loss input
    3. Clean HR frame → for SR front-end loss target
    4. HR image (43×120) → for SR branch loss (existing)
    """

    def __init__(self, root_dir, mode='train', split_ratio=0.8,
                 sr_target_h=43, sr_target_w=120):
        self.mode = mode
        self.sr_h = sr_target_h
        self.sr_w = sr_target_w

        if mode == 'train':
            self.transform = get_train_transforms()
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms()
            self.degrade = None

        self.clean_transform = get_clean_transforms()

        base = AdvancedMultiFrameDataset(root_dir, mode=mode,
                                         split_ratio=split_ratio)
        self.samples = base.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']

        # 1. Load augmented LR frames (for recognition)
        use_hr = (self.mode == 'train' and len(item['hr_paths']) > 0
                  and random.random() < 0.5)
        if use_hr:
            images_list = self._load_frames(item['hr_paths'],
                                            apply_degradation=True)
        else:
            images_list = self._load_frames(item['lr_paths'],
                                            apply_degradation=False)
        images_tensor = torch.stack(images_list, dim=0)

        # 2. Target
        target = [Config.CHAR2IDX[c] for c in label if c in Config.CHAR2IDX]
        if len(target) == 0:
            target = [0]

        # 3. HR image for SR branch (43×120)
        hr_tensor = self._load_hr_for_branch(item.get('hr_paths', []))

        # 4. Clean LR frame for SR front-end loss (64×224, no augmentation)
        clean_lr = self._load_clean_frame(item['lr_paths'])

        # 5. Clean HR frame for SR front-end loss target (64×224, no augmentation)
        clean_hr = self._load_clean_hr(item.get('hr_paths', []))

        return (images_tensor,
                torch.tensor(target, dtype=torch.long),
                len(target),
                label,
                hr_tensor,      # [3, 43, 120] for SR branch
                clean_lr,       # [3, 64, 224] for SR front loss
                clean_hr)       # [3, 64, 224] for SR front target

    def _load_frames(self, paths, apply_degradation=False):
        if len(paths) < 5:
            paths = paths + [paths[-1]] * (5 - len(paths))
        else:
            paths = paths[:5]
        images = []
        for p in paths:
            image = cv2.imread(p)
            if image is None:
                image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if apply_degradation and self.degrade:
                image = self.degrade(image=image)['image']
            image = self.transform(image=image)['image']
            images.append(image)
        return images

    def _load_clean_frame(self, lr_paths):
        """Load first LR frame with ONLY resize+normalize (no augmentation)."""
        if lr_paths:
            img = cv2.imread(lr_paths[0])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return self.clean_transform(image=img)['image']
        return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH)

    def _load_clean_hr(self, hr_paths):
        """Load first HR frame at model input size (64×224) for SR front loss."""
        if hr_paths:
            img = cv2.imread(hr_paths[0])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return self.clean_transform(image=img)['image']
        return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH)

    def _load_hr_for_branch(self, hr_paths):
        """Load HR at SR branch target size (43×120)."""
        if hr_paths:
            img = cv2.imread(hr_paths[0])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.sr_w, self.sr_h))
                return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return torch.zeros(3, self.sr_h, self.sr_w)

    @staticmethod
    def collate_fn(batch):
        (images, targets, target_lengths, labels_text,
         hr_images, clean_lrs, clean_hrs) = zip(*batch)
        images = torch.stack(images)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        hr_images = torch.stack(hr_images)
        clean_lrs = torch.stack(clean_lrs)
        clean_hrs = torch.stack(clean_hrs)
        return (images, targets, target_lengths, list(labels_text),
                hr_images, clean_lrs, clean_hrs)


# ════════════════════════════════════════════════════════════════════════════
# Focal CTC Loss
# ════════════════════════════════════════════════════════════════════════════
class FocalCTCLoss(nn.Module):
    def __init__(self, blank=0, gamma=2.0, zero_infinity=True):
        super().__init__()
        self.gamma = gamma
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity,
                              reduction='none')

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        with torch.no_grad():
            p_easy = torch.exp(-ctc_loss.detach())
            focal_weight = (1.0 - p_easy) ** self.gamma
        return (focal_weight * ctc_loss).mean()


# ════════════════════════════════════════════════════════════════════════════
# SR Front-end Loss (L1 + Perceptual)
# ════════════════════════════════════════════════════════════════════════════
class SRFrontLoss(nn.Module):
    """
    Loss for SR front-end: L1 + perceptual loss.

    Compares enhanced LR frame with HR ground truth (both at 64×224).
    Input to this loss should be clean (no augmentation) for meaningful comparison.
    """

    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual

        # Reuse perceptual loss from sr_branch
        from models.sr_branch import PerceptualLoss
        self.perceptual = PerceptualLoss()

    def forward(self, enhanced, target):
        """
        Args:
            enhanced: [B, 3, H, W] — SR-enhanced LR frame (normalized -1..1)
            target:   [B, 3, H, W] — HR ground truth (normalized -1..1)

        Returns:
            total_loss, (l1_loss, perceptual_loss)
        """
        # Denormalize for perceptual loss (expects 0..1)
        enhanced_01 = (enhanced * 0.5 + 0.5).clamp(0, 1)
        target_01 = (target * 0.5 + 0.5).clamp(0, 1)

        l1_loss = F.l1_loss(enhanced_01, target_01)
        perc_loss = self.perceptual(enhanced_01, target_01)
        total = self.lambda_l1 * l1_loss + self.lambda_perceptual * perc_loss

        return total, (l1_loss.item(), perc_loss.item())


# ════════════════════════════════════════════════════════════════════════════
# SWA Helper
# ════════════════════════════════════════════════════════════════════════════
class SWAModel:
    def __init__(self, model):
        self.n_models = 0
        self.avg_params = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }
        self.avg_buffers = {}
        for name, buf in model.named_buffers():
            if 'running_mean' in name or 'running_var' in name:
                self.avg_buffers[name] = buf.detach().cpu().clone()

    def update(self, model):
        self.n_models += 1
        for name, param in model.named_parameters():
            self.avg_params[name] += (
                param.detach().cpu() - self.avg_params[name]
            ) / self.n_models
        for name, buf in model.named_buffers():
            if name in self.avg_buffers:
                self.avg_buffers[name] += (
                    buf.detach().cpu().float() - self.avg_buffers[name].float()
                ) / self.n_models

    def apply(self, model):
        device = next(model.parameters()).device
        state = model.state_dict()
        for name in self.avg_params:
            if name in state:
                state[name] = self.avg_params[name].to(device)
        for name in self.avg_buffers:
            if name in state:
                state[name] = self.avg_buffers[name].to(device)
        model.load_state_dict(state)
        return model


def update_bn(model, loader, device, num_batches=200):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()
            module.momentum = None
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            model(images)


# ════════════════════════════════════════════════════════════════════════════
# Discriminative LR — 4 Param Groups
# ════════════════════════════════════════════════════════════════════════════
def get_param_groups(model, cfg):
    """
    Split into 4 groups:
    1. SR front-end (new, needs higher LR)
    2. Backbone (pretrained, very low LR)
    3. LSTM + Fusion (pretrained, moderate LR)
    4. Head + STN + SR branch (pretrained, moderate-high LR)
    """
    sr_params = []
    backbone_params = []
    lstm_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'sr_frontend' in name:
            sr_params.append(param)
        elif 'base.backbone' in name or 'base.adapt' in name or 'backbone' in name and 'sr' not in name:
            backbone_params.append(param)
        elif 'base.rnn' in name or 'base.fusion' in name or 'rnn' in name or 'fusion' in name:
            lstm_params.append(param)
        else:
            head_params.append(param)

    groups = [
        {'params': sr_params,       'lr': cfg['lr_sr_frontend'], 'name': 'sr_frontend'},
        {'params': backbone_params,  'lr': cfg['lr_backbone'],    'name': 'backbone'},
        {'params': lstm_params,      'lr': cfg['lr_lstm'],        'name': 'lstm'},
        {'params': head_params,      'lr': cfg['lr_head'],        'name': 'head'},
    ]

    total = sum(p.numel() for g in groups for p in g['params'])
    for g in groups:
        n = sum(p.numel() for p in g['params'])
        trainable = sum(p.numel() for p in g['params'] if p.requires_grad)
        print(f"  Param group '{g['name']}': {n:,} params ({trainable:,} trainable), lr={g['lr']:.2e}")
    print(f"  Total: {total:,}")

    return groups


# ════════════════════════════════════════════════════════════════════════════
# TTA
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def tta_predict(model, frames_raw, device, tta_transforms):
    """Run TTA with multiple augmented views."""
    all_logits = []
    for tta_tf in tta_transforms:
        augmented = []
        for img in frames_raw:
            aug_img = tta_tf(image=img)['image']
            augmented.append(aug_img)
        while len(augmented) < 5:
            augmented.append(augmented[-1])
        augmented = augmented[:5]
        x = torch.stack(augmented).unsqueeze(0).to(device)
        preds = model(x)
        all_logits.append(preds)

    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    decoded = decode_predictions(torch.argmax(avg_logits, dim=2),
                                 Config.IDX2CHAR)
    probs = torch.softmax(avg_logits, dim=-1)
    conf = probs.max(dim=-1).values.mean().item()
    return decoded[0], conf


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, ctc_criterion, sr_branch_criterion,
                    sr_front_criterion, optimizer, scaler, device, epoch, cfg):
    """
    Train one epoch with 3 losses:
    1. CTC loss (recognition)
    2. SR branch loss (backbone regularizer, training-only)
    3. SR front-end loss (image enhancement, clean LR vs HR)
    """
    model.train()
    total_ctc, total_sr_branch, total_sr_front = 0.0, 0.0, 0.0
    n = 0
    ga = cfg['gradient_accumulation']
    use_sr = cfg['use_sr_branch'] and model.sr_branch is not None
    label_smooth = cfg.get('label_smoothing', 0.0)

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
    for bi, batch in enumerate(pbar):
        (images, targets, tgt_lens, _, hr_images,
         clean_lrs, clean_hrs) = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        hr_images = hr_images.to(device, non_blocking=True)
        clean_lrs = clean_lrs.to(device, non_blocking=True)
        clean_hrs = clean_hrs.to(device, non_blocking=True)

        with autocast('cuda'):
            # ── CTC + SR branch loss ──────────────────────────────────
            if use_sr:
                preds, sr_out = model(images, return_sr=True)
            else:
                preds = model(images)

            T_out = preds.size(1)
            input_lengths = torch.full((images.size(0),), T_out,
                                       dtype=torch.long)
            ctc_loss = ctc_criterion(preds.permute(1, 0, 2), targets,
                                     input_lengths, tgt_lens)

            if label_smooth > 0:
                confidence_penalty = -preds.mean()
                ctc_loss = ctc_loss + label_smooth * confidence_penalty

            if use_sr and sr_out is not None:
                sr_branch_loss, _ = sr_branch_criterion(sr_out, hr_images)
            else:
                sr_branch_loss = torch.tensor(0.0, device=device)

            # ── SR front-end loss (clean LR → enhanced vs clean HR) ──
            enhanced_clean = model.enhance_frames(clean_lrs)
            sr_front_loss, _ = sr_front_criterion(enhanced_clean, clean_hrs)

            # ── Total loss ────────────────────────────────────────────
            loss = (ctc_loss
                    + cfg['lambda_sr_branch'] * sr_branch_loss
                    + cfg['lambda_sr_frontend'] * sr_front_loss) / ga

        scaler.scale(loss).backward()

        if (bi + 1) % ga == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_ctc += ctc_loss.item()
        total_sr_branch += sr_branch_loss.item()
        total_sr_front += sr_front_loss.item()
        n += 1

        pbar.set_postfix({
            'ctc': f"{ctc_loss.item():.3f}",
            'sr_f': f"{sr_front_loss.item():.3f}",
            'lr_sr': f"{optimizer.param_groups[0]['lr']:.1e}",
        })

    # Flush remaining gradients
    if n % ga != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=cfg['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_ctc / n, total_sr_branch / n, total_sr_front / n


@torch.no_grad()
def validate(model, loader, ctc_criterion_val, device):
    model.eval()
    val_loss = 0.0
    all_preds, all_targets, all_confs = [], [], []

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        tgt_lens = batch[2]
        labels_text = batch[3]

        preds = model(images)
        T_out = preds.size(1)
        input_lengths = torch.full((images.size(0),), T_out, dtype=torch.long)
        loss = ctc_criterion_val(preds.permute(1, 0, 2), targets,
                                 input_lengths, tgt_lens)
        val_loss += loss.item()

        decoded = decode_predictions(torch.argmax(preds, dim=2),
                                     Config.IDX2CHAR)
        all_preds.extend(decoded)
        all_targets.extend(labels_text)
        all_confs.extend(get_prediction_confidence(preds).tolist())

    avg_loss = val_loss / len(loader)
    acc = calculate_accuracy(all_preds, all_targets) * 100
    cer = calculate_cer(all_preds, all_targets)
    is_c = [p == t for p, t in zip(all_preds, all_targets)]
    gap = calculate_confidence_gap(all_confs, is_c)

    confusion_stats = {}
    for gt, pred in zip(all_targets, all_preds):
        for pos in range(min(len(gt), len(pred))):
            gc, pc = gt[pos], pred[pos]
            if gc != pc:
                pair = f"{gc}→{pc}"
                confusion_stats[pair] = confusion_stats.get(pair, 0) + 1

    return {'loss': avg_loss, 'accuracy': acc, 'cer': cer,
            'confidence_gap': gap, 'outputs/predictions/predictions': all_preds,
            'targets': all_targets, 'confusion_stats': confusion_stats}


@torch.no_grad()
def validate_tta(model, val_dataset, device, tta_transforms):
    model.eval()
    all_preds, all_targets = [], []
    print(f"  Running TTA validation ({len(tta_transforms)} views)...")
    for idx in tqdm(range(len(val_dataset)), desc="TTA Val"):
        item = val_dataset.samples[idx]
        label = item['label']
        lr_paths = item['lr_paths']
        if len(lr_paths) < 5:
            lr_paths = lr_paths + [lr_paths[-1]] * (5 - len(lr_paths))
        else:
            lr_paths = lr_paths[:5]

        frames_raw = []
        for p in lr_paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_raw.append(img)

        decoded, conf = tta_predict(model, frames_raw, device, tta_transforms)
        all_preds.append(decoded)
        all_targets.append(label)

    acc = calculate_accuracy(all_preds, all_targets) * 100
    cer = calculate_cer(all_preds, all_targets)
    return {'accuracy': acc, 'cer': cer}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG
    seed_everything(Config.SEED)
    device = Config.DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase13_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 13 — SR-Enhanced Recognition Pipeline")
    print("  NEW: SREnhancer front-end (runs at inference!)")
    print("=" * 80)
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = Phase13Dataset(Config.DATA_ROOT, mode='train', split_ratio=0.8,
                              sr_target_h=cfg['sr_target_h'],
                              sr_target_w=cfg['sr_target_w'])
    val_ds = Phase13Dataset(Config.DATA_ROOT, mode='val', split_ratio=0.8,
                            sr_target_h=cfg['sr_target_h'],
                            sr_target_w=cfg['sr_target_w'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,
                              collate_fn=Phase13Dataset.collate_fn,
                              num_workers=cfg['num_workers'], pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, collate_fn=Phase13Dataset.collate_fn,
                            num_workers=cfg['num_workers'], pin_memory=True,
                            persistent_workers=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Base Model (Phase3Recognizer) ────────────────────────────────────
    base_model = Phase3Recognizer(
        num_classes=Config.NUM_CLASSES,
        use_stn=Config.USE_STN,
        use_resnet_backbone=Config.USE_RESNET_BACKBONE,
        hidden_size=cfg['hidden_size'],
        num_lstm_layers=cfg['num_lstm_layers'],
        dropout=cfg['dropout'],
        fusion_reduction=cfg['fusion_reduction'],
        use_sr_branch=cfg['use_sr_branch'],
        sr_target_h=cfg['sr_target_h'],
        sr_target_w=cfg['sr_target_w'],
    ).to(device)

    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Base model params: {base_params:,}")

    # Load Phase 10 best (79.33%)
    p10_path = cfg['pretrained_path']
    p10_acc = 0.0
    if os.path.exists(p10_path):
        ckpt = torch.load(p10_path, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt['model_state_dict'])
        p10_acc = ckpt.get('accuracy', 0)
        p10_epoch = ckpt.get('epoch', -1) + 1
        print(f"  ✅ Loaded base model: {p10_path} (acc={p10_acc:.2f}%, epoch {p10_epoch})")
    else:
        print(f"  ❌ {p10_path} not found!")

    # ── Wrap with SR Front-end ───────────────────────────────────────────
    model = Phase13Model(
        base_model,
        sr_channels=cfg['sr_frontend_channels'],
        sr_blocks=cfg['sr_frontend_blocks'],
    ).to(device)

    sr_params = sum(p.numel() for p in model.sr_frontend.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  SR front-end params: {sr_params:,} (NEW, randomly initialized)")
    print(f"  Total params: {total_params:,} (+{sr_params:,} from SR)")

    # ── Losses ───────────────────────────────────────────────────────────
    ctc_criterion = FocalCTCLoss(blank=0, gamma=cfg['focal_ctc_gamma'])
    ctc_criterion_val = nn.CTCLoss(blank=0, zero_infinity=True)
    sr_branch_criterion = SRLoss(
        lambda_l1=1.0, lambda_perceptual=cfg['lambda_perceptual']).to(device)
    sr_front_criterion = SRFrontLoss(
        lambda_l1=1.0,
        lambda_perceptual=cfg['lambda_sr_frontend_perceptual']).to(device)

    # ── Optimizer with 4 Discriminative LR Groups ────────────────────────
    param_groups = get_param_groups(model, cfg)
    optimizer = optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])

    # ── CosineAnnealingLR (proven stable) ────────────────────────────────
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['epochs'],
        eta_min=cfg['lr_min'],
    )

    scaler = GradScaler()

    # ── SWA ──────────────────────────────────────────────────────────────
    swa = SWAModel(model)
    swa_updated = False
    swa_count = 0
    print(f"  ⚡ SWA: start at epoch {cfg['swa_start_epoch']}, "
          f"every {cfg['swa_interval']} epochs")

    # ── Training loop ────────────────────────────────────────────────────
    patience_counter = 0
    best_acc = 0.0
    best_cer = float('inf')
    best_epoch = -1

    print(f"\n{'='*80}")
    print("  Starting Phase 13 SR-Enhanced training from Phase 10...")
    print(f"  {cfg['epochs']} epochs | patience={cfg['patience']}")
    print(f"  4 LR groups: sr_frontend={cfg['lr_sr_frontend']:.1e}, "
          f"backbone={cfg['lr_backbone']:.1e}, "
          f"lstm={cfg['lr_lstm']:.1e}, head={cfg['lr_head']:.1e}")
    print(f"  Losses: CTC + {cfg['lambda_sr_frontend']}×SR_front + "
          f"{cfg['lambda_sr_branch']}×SR_branch")
    print(f"  Scheduler: CosineAnnealingLR (monotonic decay)")
    print(f"  TTA: {cfg['use_tta']} ({cfg['tta_n_augs']} views)")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        ctc_l, sr_branch_l, sr_front_l = train_one_epoch(
            model, train_loader, ctc_criterion, sr_branch_criterion,
            sr_front_criterion, optimizer, scaler, device, epoch, cfg)

        val = validate(model, val_loader, ctc_criterion_val, device)

        # Step scheduler per epoch
        scheduler.step()

        # ── SWA snapshot ────────────────────────────────────────────────
        if epoch >= cfg['swa_start_epoch'] and (epoch + 1) % cfg['swa_interval'] == 0:
            swa.update(model)
            swa_count += 1
            swa_updated = True
            print(f"  📸 SWA snapshot #{swa_count} collected (epoch {epoch+1})")

        # Logging
        writer.add_scalar('Loss/train_ctc', ctc_l, epoch)
        writer.add_scalar('Loss/sr_branch', sr_branch_l, epoch)
        writer.add_scalar('Loss/sr_frontend', sr_front_l, epoch)
        writer.add_scalar('Loss/val', val['loss'], epoch)
        writer.add_scalar('Accuracy/val', val['accuracy'], epoch)
        writer.add_scalar('CER/val', val['cer'], epoch)
        for i, g in enumerate(optimizer.param_groups):
            writer.add_scalar(f'LR/group_{i}', g['lr'], epoch)

        print(f"Epoch {epoch+1}/{cfg['epochs']}:")
        print(f"  Train CTC: {ctc_l:.4f} | SR_branch: {sr_branch_l:.4f} | "
              f"SR_front: {sr_front_l:.4f}")
        print(f"  Val   Loss: {val['loss']:.4f} | Acc: {val['accuracy']:.2f}% | "
              f"CER: {val['cer']:.4f} | Gap: {val['confidence_gap']:.4f}")
        lr_str = ', '.join(f"{g.get('name','?')}={g['lr']:.2e}"
                           for g in optimizer.param_groups)
        print(f"  LR: {lr_str} | SWA: {swa_count}")

        if val['confusion_stats']:
            top_conf = sorted(val['confusion_stats'].items(),
                              key=lambda x: -x[1])[:5]
            conf_str = ', '.join(f"{k}:{v}" for k, v in top_conf)
            print(f"  Top confusions: {conf_str}")

        # Save best
        if val['accuracy'] > best_acc:
            best_acc = val['accuracy']
            best_cer = val['cer']
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val['accuracy'],
                'cer': val['cer'],
                'confidence_gap': val['confidence_gap'],
                'config': model.get_model_info(),
                'phase13_config': cfg,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
                'has_sr_frontend': True,
                'sr_frontend_channels': cfg['sr_frontend_channels'],
                'sr_frontend_blocks': cfg['sr_frontend_blocks'],
            }, cfg['model_save_name'])
            print(f"  -> Saved {cfg['model_save_name']}  "
                  f"(Acc {val['accuracy']:.2f}%, CER {val['cer']:.4f})")
            if p10_acc > 0:
                delta = val['accuracy'] - p10_acc
                print(f"     vs Phase 10: {delta:+.2f}%"
                      f" {'🎯 NEW BEST!' if delta > 0 else ''}")
        else:
            patience_counter += 1

        # Samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n  Sample predictions (epoch {epoch+1}):")
            for i in range(min(8, len(val['outputs/predictions/predictions']))):
                p, t = val['outputs/predictions/predictions'][i], val['targets'][i]
                m = "+" if p == t else "x"
                print(f"    [{m}] GT: {t:8s}  Pred: {p:8s}")
            print()

        # Early stopping
        if patience_counter >= cfg['patience']:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(best epoch {best_epoch+1}, acc {best_acc:.2f}%)")
            break

    # ════════════════════════════════════════════════════════════════════
    # SWA
    # ════════════════════════════════════════════════════════════════════
    swa_acc = 0.0
    if swa_updated and swa_count >= 2:
        print(f"\n{'='*80}")
        print(f"  Applying SWA ({swa_count} snapshots)...")
        print(f"{'='*80}")

        best_ckpt = torch.load(cfg['model_save_name'], map_location=device,
                               weights_only=False)

        swa.apply(model)
        print("  Recomputing BatchNorm statistics...")
        update_bn(model, train_loader, device, num_batches=200)

        swa_val = validate(model, val_loader, ctc_criterion_val, device)
        swa_acc = swa_val['accuracy']

        print(f"  SWA Accuracy: {swa_acc:.2f}% | CER: {swa_val['cer']:.4f}")
        print(f"  Best single:  {best_acc:.2f}%")
        print(f"  Δ SWA:        {swa_acc - best_acc:+.2f}%")

        if swa_val['confusion_stats']:
            top_conf = sorted(swa_val['confusion_stats'].items(),
                              key=lambda x: -x[1])[:8]
            conf_str = ', '.join(f"{k}:{v}" for k, v in top_conf)
            print(f"  SWA confusions: {conf_str}")

        if swa_acc > best_acc:
            print(f"  ✅ SWA improved! Saving as {cfg['swa_model_save_name']}")
            torch.save({
                'epoch': -1,
                'model_state_dict': model.state_dict(),
                'accuracy': swa_acc,
                'cer': swa_val['cer'],
                'config': model.get_model_info(),
                'phase13_config': cfg,
                'swa_snapshots': swa_count,
                'has_sr_frontend': True,
                'sr_frontend_channels': cfg['sr_frontend_channels'],
                'sr_frontend_blocks': cfg['sr_frontend_blocks'],
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['swa_model_save_name'])
            torch.save({
                'epoch': -1,
                'model_state_dict': model.state_dict(),
                'accuracy': swa_acc,
                'cer': swa_val['cer'],
                'config': model.get_model_info(),
                'phase13_config': cfg,
                'swa_snapshots': swa_count,
                'has_sr_frontend': True,
                'sr_frontend_channels': cfg['sr_frontend_channels'],
                'sr_frontend_blocks': cfg['sr_frontend_blocks'],
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['model_save_name'])
            best_acc = swa_acc
            best_cer = swa_val['cer']
        else:
            print(f"  ❌ SWA didn't improve. Keeping best single model.")
            model.load_state_dict(best_ckpt['model_state_dict'])
    else:
        print(f"\n  ⚠️ SWA: {swa_count} snapshots (need >= 2).")

    # ════════════════════════════════════════════════════════════════════
    # TTA Evaluation (6 views)
    # ════════════════════════════════════════════════════════════════════
    if cfg['use_tta']:
        print(f"\n{'='*80}")
        print(f"  Running Enhanced TTA evaluation ({cfg['tta_n_augs']} views)...")
        print(f"{'='*80}")
        tta_transforms = get_tta_transforms()
        tta_result = validate_tta(model, val_ds, device, tta_transforms)
        tta_acc = tta_result['accuracy']
        print(f"  TTA Accuracy: {tta_acc:.2f}% | CER: {tta_result['cer']:.4f}")
        print(f"  vs Best:      {best_acc:.2f}%")
        print(f"  Δ TTA:        {tta_acc - best_acc:+.2f}%")

        if tta_acc > best_acc:
            print(f"  ✅ TTA improved! (inference-only)")
            best_acc = tta_acc

    writer.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Phase 13 Training Completed")
    print(f"{'='*80}")
    print(f"  Best Accuracy:       {best_acc:.2f}%")
    print(f"  Best CER:            {best_cer:.4f}")
    print(f"  Best Epoch:          {best_epoch+1}/{cfg['epochs']}")
    print(f"  SWA Snapshots:       {swa_count}")
    print(f"  SR Frontend:         {sr_params:,} params "
          f"({cfg['sr_frontend_channels']}ch × {cfg['sr_frontend_blocks']} blocks)")
    print(f"  Model:               {cfg['model_save_name']}")
    print(f"  Based on:            {cfg['pretrained_path']}")
    print(f"  TensorBoard:         {log_dir}")

    if p10_acc > 0:
        delta = best_acc - p10_acc
        print(f"\n  Phase 10 -> Phase 13: {p10_acc:.2f}% -> {best_acc:.2f}% "
              f"({delta:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
