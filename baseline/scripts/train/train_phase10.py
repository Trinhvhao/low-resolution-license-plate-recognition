"""
Phase 10 — Fine-tune from Phase 8 SWA (79.15%) with Discriminative LR + TTA + SWA.

Key difference from Phase 9:
  Phase 9: From P4v2, balanced sampling, enhanced augmentation, CosineWarmRestart
  Phase 10: From P8 SWA (best), discriminative LR, OneCycleLR, TTA evaluation, SWA

Analysis of Phase 8 SWA (79.15%):
  ┌─────────────────┬────────┬──────────────────────────────────────────┐
  │ Scenario        │ Acc    │ Key Issues                               │
  ├─────────────────┼────────┼──────────────────────────────────────────┤
  │ A_Brazilian     │ 77.86% │ Regression from P4v2 (83.97%)            │
  │ A_Mercosur      │ 84.96% │ OK                                       │
  │ B_Brazilian     │ 54.11% │ Major bottleneck, O/D/N/Q/V all <65%     │
  │ B_Mercosur      │ 82.62% │ Good improvement (+3.34% from P4v2)      │
  └─────────────────┴────────┴──────────────────────────────────────────┘
  - Position 1 & 2 have most errors (B_Braz: P1=80.5%, P2=78.1%)
  - Confidence gap small (model confident even when wrong)
  - Top confusions: 8↔6, M↔H, V↔Y, D↔B, 5↔6

Strategy:
  1. Discriminative LR — backbone lr/10, LSTM lr, head lr*2
     → Preserve backbone features while adapting recognition head
  2. OneCycleLR (different from CosineWarmRestart in P8/P9)
     → Better convergence for fine-tuning, natural super-convergence
  3. Conservative augmentation — P8 SWA already good, don't disrupt
  4. Test-Time Augmentation (TTA) for final evaluation
     → Horizontal flip + brightness variations → ensemble decode
  5. SWA (proven +0.82% in P8)
  6. Freeze backbone 5 epochs → unfreeze gradually (progressive unfreezing)
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

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════
IMG_HEIGHT = 64
IMG_WIDTH  = 224

CFG = {
    'epochs':                60,
    'batch_size':            24,        # reduced for GPU sharing with Phase 9
    'gradient_accumulation': 4,         # eff batch = 96

    # Discriminative LR
    'lr_backbone':           2e-5,      # backbone: very low (features already good)
    'lr_lstm':               2e-4,      # lstm: moderate
    'lr_head':               4e-4,      # head: highest (adapt recognition)
    'lr_min':                1e-6,
    'weight_decay':          1e-4,
    'max_grad_norm':         1.0,

    # OneCycleLR config
    'pct_start':             0.3,       # 30% warmup
    'anneal_strategy':       'cos',
    'div_factor':            10,        # initial_lr = max_lr / div_factor
    'final_div_factor':      1000,      # final_lr = initial_lr / final_div_factor

    # BiLSTM
    'hidden_size':           256,
    'num_lstm_layers':       2,
    'dropout':               0.25,

    # Fusion
    'fusion_reduction':      16,

    # SR branch
    'use_sr_branch':         True,
    'sr_target_h':           43,
    'sr_target_w':           120,
    'lambda_sr':             0.1,
    'lambda_perceptual':     0.1,

    # Focal CTC
    'focal_ctc_gamma':       2.0,

    # Fine-tune from Phase 8 SWA (BEST model)
    'pretrained_path':       'checkpoints/best_model_phase8_swa.pth',

    # Progressive unfreezing
    'freeze_backbone_epochs': 5,        # freeze backbone for first 5 epochs

    # Early stopping
    'patience':              30,

    # Regularization
    'label_smoothing':       0.05,

    # SWA config
    'swa_start_epoch':       20,        # start collecting later (model needs to settle)
    'swa_interval':          5,         # collect every 5 epochs (not tied to cycle)

    # TTA config
    'use_tta':               True,      # test-time augmentation
    'tta_n_augs':            4,         # number of augmented views

    # Workers
    'num_workers':           4,

    # Save
    'model_save_name':       'checkpoints/best_model_phase10.pth',
    'swa_model_save_name':   'checkpoints/best_model_phase10_swa.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Augmentation — Conservative (P8 SWA is already strong)
# ════════════════════════════════════════════════════════════════════════════
def get_train_transforms():
    """Conservative augmentation — don't disrupt P8 SWA's learned features."""
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
        # Blur/Noise — lighter
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


def get_tta_transforms():
    """TTA augmentation variants for test-time ensemble."""
    return [
        # Original (no augmentation)
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
        # Slight brightness up
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.15),
                                       contrast_limit=(0.05, 0.1), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
        # Slight brightness down
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.RandomBrightnessContrast(brightness_limit=(-0.15, -0.1),
                                       contrast_limit=(0.05, 0.1), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]),
        # Sharpen
        A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.9, 1.1), p=1.0),
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
# Dataset
# ════════════════════════════════════════════════════════════════════════════
class Phase10Dataset(Dataset):
    def __init__(self, root_dir, mode='train', split_ratio=0.8,
                 sr_target_h=43, sr_target_w=120, custom_transform=None):
        self.mode = mode
        self.sr_h = sr_target_h
        self.sr_w = sr_target_w
        if custom_transform is not None:
            self.transform = custom_transform
            self.degrade = None
        elif mode == 'train':
            self.transform = get_train_transforms()
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms()
            self.degrade = None

        base = AdvancedMultiFrameDataset(root_dir, mode=mode,
                                         split_ratio=split_ratio)
        self.samples = base.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']
        use_hr = (self.mode == 'train' and len(item['hr_paths']) > 0
                  and random.random() < 0.5)
        if use_hr:
            images_list = self._load_frames(item['hr_paths'],
                                            apply_degradation=True)
        else:
            images_list = self._load_frames(item['lr_paths'],
                                            apply_degradation=False)
        images_tensor = torch.stack(images_list, dim=0)
        target = [Config.CHAR2IDX[c] for c in label if c in Config.CHAR2IDX]
        if len(target) == 0:
            target = [0]
        hr_tensor = self._load_hr(item.get('hr_paths', []))
        return (images_tensor, torch.tensor(target, dtype=torch.long),
                len(target), label, hr_tensor)

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

    def _load_hr(self, hr_paths):
        if hr_paths:
            img = cv2.imread(hr_paths[0])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.sr_w, self.sr_h))
                return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return torch.zeros(3, self.sr_h, self.sr_w)

    @staticmethod
    def collate_fn(batch):
        images, targets, target_lengths, labels_text, hr_images = zip(*batch)
        images = torch.stack(images)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        hr_images = torch.stack(hr_images)
        return images, targets, target_lengths, list(labels_text), hr_images


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
# SWA Helper (proven in Phase 8)
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
# Discriminative LR — Param Groups
# ════════════════════════════════════════════════════════════════════════════
def get_param_groups(model, cfg):
    """
    Split model parameters into 3 groups with different learning rates:
    1. Backbone (ResNet34) — lowest LR (features already good)
    2. LSTM + Fusion — moderate LR
    3. Head (FC) + STN + SR — highest LR (adapt to task)
    
    Always includes all parameters regardless of requires_grad.
    """
    backbone_params = []
    lstm_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name or 'adapt' in name:
            backbone_params.append(param)
        elif 'rnn' in name or 'fusion' in name:
            lstm_params.append(param)
        else:
            head_params.append(param)

    groups = [
        {'params': backbone_params, 'lr': cfg['lr_backbone'], 'name': 'backbone'},
        {'params': lstm_params,     'lr': cfg['lr_lstm'],     'name': 'lstm'},
        {'params': head_params,     'lr': cfg['lr_head'],     'name': 'head'},
    ]

    total = sum(p.numel() for g in groups for p in g['params'])
    for g in groups:
        n = sum(p.numel() for p in g['params'])
        trainable = sum(p.numel() for p in g['params'] if p.requires_grad)
        print(f"  Param group '{g['name']}': {n:,} params ({trainable:,} trainable), lr={g['lr']:.2e}")
    print(f"  Total: {total:,}")

    return groups


# ════════════════════════════════════════════════════════════════════════════
# Progressive Unfreezing
# ════════════════════════════════════════════════════════════════════════════
def freeze_backbone(model):
    """Freeze backbone parameters."""
    count = 0
    for name, param in model.named_parameters():
        if 'backbone' in name or 'adapt' in name:
            param.requires_grad = False
            count += 1
    print(f"  🔒 Froze {count} backbone parameters")


def unfreeze_backbone(model):
    """Unfreeze backbone parameters."""
    count = 0
    for name, param in model.named_parameters():
        if 'backbone' in name or 'adapt' in name:
            param.requires_grad = True
            count += 1
    print(f"  🔓 Unfroze {count} backbone parameters")


# ════════════════════════════════════════════════════════════════════════════
# Test-Time Augmentation
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def tta_predict(model, frames_raw, device, tta_transforms):
    """
    Run TTA: apply multiple augmentations to the raw frames,
    average the logits, then decode.

    frames_raw: list of numpy images (H, W, 3) RGB, already read
    Returns: (decoded_text, confidence)
    """
    all_logits = []

    for tta_tf in tta_transforms:
        # Apply this TTA transform to all frames
        augmented = []
        for img in frames_raw:
            aug_img = tta_tf(image=img)['image']
            augmented.append(aug_img)

        # Pad to 5 frames
        while len(augmented) < 5:
            augmented.append(augmented[-1])
        augmented = augmented[:5]

        x = torch.stack(augmented).unsqueeze(0).to(device)  # [1, 5, C, H, W]
        preds = model(x)  # [1, T, num_classes]
        all_logits.append(preds)

    # Average logits across TTA views
    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # [1, T, num_classes]

    # Decode
    decoded = decode_predictions(torch.argmax(avg_logits, dim=2),
                                 Config.IDX2CHAR)
    probs = torch.softmax(avg_logits, dim=-1)
    conf = probs.max(dim=-1).values.mean().item()

    return decoded[0], conf


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, ctc_criterion, sr_criterion,
                    optimizer, scaler, scheduler, device, epoch, cfg):
    model.train()
    total_ctc, total_sr = 0.0, 0.0
    n = 0
    ga = cfg['gradient_accumulation']
    use_sr = cfg['use_sr_branch'] and model.sr_branch is not None
    label_smooth = cfg.get('label_smoothing', 0.0)

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
    for bi, batch in enumerate(pbar):
        images, targets, tgt_lens, _, hr_images = batch
        hr_images = hr_images.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast('cuda'):
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
                sr_loss_val, _ = sr_criterion(sr_out, hr_images)
                loss = (ctc_loss + cfg['lambda_sr'] * sr_loss_val) / ga
            else:
                sr_loss_val = torch.tensor(0.0)
                loss = ctc_loss / ga

        scaler.scale(loss).backward()

        if (bi + 1) % ga == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # OneCycleLR steps per optimizer step
            scheduler.step()

        total_ctc += ctc_loss.item()
        total_sr += (sr_loss_val.item() if isinstance(sr_loss_val, torch.Tensor)
                     else sr_loss_val)
        n += 1

        pbar.set_postfix({
            'ctc': f"{ctc_loss.item():.4f}",
            'lr_b': f"{optimizer.param_groups[0]['lr']:.2e}",
            'lr_h': f"{optimizer.param_groups[2]['lr']:.2e}",
        })

    # Flush remaining gradients
    if n % ga != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=cfg['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    return total_ctc / n, total_sr / n


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
    """Validate with Test-Time Augmentation."""
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

        # Load raw frames
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
    return {'accuracy': acc, 'cer': cer, 'outputs/predictions/predictions': all_preds,
            'targets': all_targets}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG
    seed_everything(Config.SEED)
    device = Config.DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase10_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 10 — From P8 SWA + Discriminative LR + OneCycleLR + TTA + SWA")
    print("=" * 80)
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = Phase10Dataset(Config.DATA_ROOT, mode='train', split_ratio=0.8,
                              sr_target_h=cfg['sr_target_h'],
                              sr_target_w=cfg['sr_target_w'])
    val_ds = Phase10Dataset(Config.DATA_ROOT, mode='val', split_ratio=0.8,
                            sr_target_h=cfg['sr_target_h'],
                            sr_target_w=cfg['sr_target_w'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,
                              collate_fn=Phase10Dataset.collate_fn,
                              num_workers=cfg['num_workers'], pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, collate_fn=Phase10Dataset.collate_fn,
                            num_workers=cfg['num_workers'], pin_memory=True,
                            persistent_workers=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────
    model = Phase3Recognizer(
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

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_p:,}")

    # Load Phase 8 SWA checkpoint (BEST model at 79.15%)
    p8_path = cfg['pretrained_path']
    p8_acc = 0.0
    if os.path.exists(p8_path):
        ckpt = torch.load(p8_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        p8_acc = ckpt.get('accuracy', 0)
        swa_snaps = ckpt.get('swa_snapshots', 0)
        print(f"  ✅ Loaded {p8_path}: acc={p8_acc:.2f}%, SWA snapshots={swa_snaps}")
    else:
        print(f"  ❌ {p8_path} not found! Training from scratch.")

    # ── Progressive Unfreezing ───────────────────────────────────────────
    if cfg['freeze_backbone_epochs'] > 0:
        freeze_backbone(model)

    # ── Loss ─────────────────────────────────────────────────────────────
    ctc_criterion = FocalCTCLoss(blank=0, gamma=cfg['focal_ctc_gamma'])
    ctc_criterion_val = nn.CTCLoss(blank=0, zero_infinity=True)
    sr_criterion = SRLoss(lambda_l1=1.0,
                          lambda_perceptual=cfg['lambda_perceptual']).to(device)

    # ── Optimizer with Discriminative LR ─────────────────────────────────
    param_groups = get_param_groups(model, cfg)
    optimizer = optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])

    # ── OneCycleLR (different from CosineWarmRestart in P8/P9) ───────────
    total_steps = cfg['epochs'] * len(train_loader) // cfg['gradient_accumulation']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg['lr_backbone'], cfg['lr_lstm'], cfg['lr_head']],
        total_steps=total_steps,
        pct_start=cfg['pct_start'],
        anneal_strategy=cfg['anneal_strategy'],
        div_factor=cfg['div_factor'],
        final_div_factor=cfg['final_div_factor'],
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
    best_gap = 0.0
    best_epoch = -1

    print(f"\n{'='*80}")
    print("  Starting Phase 10 fine-tuning from Phase 8 SWA...")
    print(f"  {cfg['epochs']} epochs | patience={cfg['patience']}")
    print(f"  Discriminative LR: backbone={cfg['lr_backbone']:.2e}, "
          f"lstm={cfg['lr_lstm']:.2e}, head={cfg['lr_head']:.2e}")
    print(f"  OneCycleLR | Progressive unfreezing at epoch {cfg['freeze_backbone_epochs']}")
    print(f"  TTA: {cfg['use_tta']} ({cfg['tta_n_augs']} views)")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        # ── Progressive unfreezing ──────────────────────────────────────
        if epoch == cfg['freeze_backbone_epochs'] and cfg['freeze_backbone_epochs'] > 0:
            unfreeze_backbone(model)
            # Rebuild param groups with proper requires_grad
            param_groups = get_param_groups(model, cfg)
            optimizer = optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])
            # Rebuild scheduler for remaining epochs
            remaining_steps = (cfg['epochs'] - epoch) * len(train_loader) // cfg['gradient_accumulation']
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[cfg['lr_backbone'], cfg['lr_lstm'], cfg['lr_head']],
                total_steps=remaining_steps,
                pct_start=cfg['pct_start'],
                anneal_strategy=cfg['anneal_strategy'],
                div_factor=cfg['div_factor'],
                final_div_factor=cfg['final_div_factor'],
            )
            scaler = GradScaler()

        ctc_l, sr_l = train_one_epoch(
            model, train_loader, ctc_criterion, sr_criterion,
            optimizer, scaler, scheduler, device, epoch, cfg)

        val = validate(model, val_loader, ctc_criterion_val, device)

        # ── SWA snapshot ────────────────────────────────────────────────
        if epoch >= cfg['swa_start_epoch'] and (epoch + 1) % cfg['swa_interval'] == 0:
            swa.update(model)
            swa_count += 1
            swa_updated = True
            print(f"  📸 SWA snapshot #{swa_count} collected (epoch {epoch+1})")

        # Logging
        writer.add_scalar('Loss/train_ctc', ctc_l, epoch)
        writer.add_scalar('Loss/train_sr', sr_l, epoch)
        writer.add_scalar('Loss/val', val['loss'], epoch)
        writer.add_scalar('Accuracy/val', val['accuracy'], epoch)
        writer.add_scalar('CER/val', val['cer'], epoch)
        writer.add_scalar('ConfGap/val', val['confidence_gap'], epoch)
        for i, g in enumerate(optimizer.param_groups):
            writer.add_scalar(f'LR/group_{i}', g['lr'], epoch)

        print(f"Epoch {epoch+1}/{cfg['epochs']}:")
        print(f"  Train CTC: {ctc_l:.4f} | SR: {sr_l:.4f}")
        print(f"  Val   Loss: {val['loss']:.4f} | Acc: {val['accuracy']:.2f}% | "
              f"CER: {val['cer']:.4f} | Gap: {val['confidence_gap']:.4f}")
        lr_str = ', '.join(f"{g.get('name','?')}={g['lr']:.2e}" for g in optimizer.param_groups)
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
            best_gap = val['confidence_gap']
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
                'phase10_config': cfg,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['model_save_name'])
            print(f"  -> Saved {cfg['model_save_name']}  "
                  f"(Acc {val['accuracy']:.2f}%, CER {val['cer']:.4f})")
            if p8_acc > 0:
                delta = val['accuracy'] - p8_acc
                print(f"     vs Phase 8 SWA: {delta:+.2f}%"
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
                'confidence_gap': swa_val['confidence_gap'],
                'config': model.get_model_info(),
                'phase10_config': cfg,
                'swa_snapshots': swa_count,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['swa_model_save_name'])
            torch.save({
                'epoch': -1,
                'model_state_dict': model.state_dict(),
                'accuracy': swa_acc,
                'cer': swa_val['cer'],
                'confidence_gap': swa_val['confidence_gap'],
                'config': model.get_model_info(),
                'phase10_config': cfg,
                'swa_snapshots': swa_count,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['model_save_name'])
            best_acc = swa_acc
            best_cer = swa_val['cer']
        else:
            print(f"  ❌ SWA didn't improve. Keeping best single model.")
            model.load_state_dict(best_ckpt['model_state_dict'])
    else:
        print(f"\n  ⚠️ Not enough SWA snapshots ({swa_count}). Need >= 2.")

    # ════════════════════════════════════════════════════════════════════
    # TTA Evaluation
    # ════════════════════════════════════════════════════════════════════
    if cfg['use_tta']:
        print(f"\n{'='*80}")
        print("  Running Test-Time Augmentation evaluation...")
        print(f"{'='*80}")
        tta_transforms = get_tta_transforms()
        tta_result = validate_tta(model, val_ds, device, tta_transforms)
        tta_acc = tta_result['accuracy']
        print(f"  TTA Accuracy: {tta_acc:.2f}% | CER: {tta_result['cer']:.4f}")
        print(f"  vs Best:      {best_acc:.2f}%")
        print(f"  Δ TTA:        {tta_acc - best_acc:+.2f}%")

        if tta_acc > best_acc:
            print(f"  ✅ TTA improved! (note: TTA is inference-only, no model save needed)")
            best_acc = tta_acc

    writer.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Phase 10 Training Completed")
    print(f"{'='*80}")
    print(f"  Best Accuracy:       {best_acc:.2f}%")
    print(f"  Best CER:            {best_cer:.4f}")
    print(f"  Best Epoch:          {best_epoch+1}/{cfg['epochs']}")
    print(f"  SWA Snapshots:       {swa_count}")
    print(f"  Model:               {cfg['model_save_name']}")
    print(f"  Based on:            {cfg['pretrained_path']}")
    print(f"  TensorBoard:         {log_dir}")

    if p8_acc > 0:
        delta = best_acc - p8_acc
        print(f"\n  Phase 8 SWA -> Phase 10: {p8_acc:.2f}% -> {best_acc:.2f}% "
              f"({delta:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
