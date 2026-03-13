"""
Phase 12 — Goldilocks LR: Start at P10's sweet-spot LR + cosine decay.

╔═══════════════════════════════════════════════════════════════════════════╗
║  LESSON LEARNED from P10 & P11:                                          ║
║                                                                           ║
║  P10: OneCycleLR peak 4e-4  → CRASH at epoch 21 (76.83%)                ║
║  P11: CosineAnnealing 1e-4  → TOO LOW, stuck at 79.22%                  ║
║  P10 epoch 12 (best 79.33%): backbone=9e-6, head=1.78e-4 ← SWEET SPOT  ║
║                                                                           ║
║  FIX: Start at sweet-spot LR and ONLY decay from there.                  ║
║  CosineAnnealingLR: head 2e-4 → 1e-7 over 50 epochs                    ║
║  No peaks, no restarts, strong enough to improve.                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

Strategy:
  1. Start from Phase 10 best (79.33%)
  2. LR at P10's sweet spot: backbone=1e-5, lstm=1e-4, head=2e-4
  3. CosineAnnealingLR — monotonic decay, NO peaks
  4. dropout=0.25, label_smoothing=0.05 (same as P10 — P11's 0.3/0.03 hurt)
  5. SWA from epoch 15, every 3 epochs
  6. Enhanced TTA 6 views
  7. 50 epochs with patience 30
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
    'epochs':                50,
    'batch_size':            24,
    'gradient_accumulation': 4,         # eff batch = 96

    # Goldilocks LR — at P10's sweet spot (epoch 12 values)
    'lr_backbone':           1e-5,      # P10@ep12: 8.9e-6
    'lr_lstm':               1e-4,      # P10@ep12: 8.9e-5
    'lr_head':               2e-4,      # P10@ep12: 1.78e-4
    'lr_min':                1e-7,
    'weight_decay':          1e-4,
    'max_grad_norm':         1.0,

    # BiLSTM (same as P10)
    'hidden_size':           256,
    'num_lstm_layers':       2,
    'dropout':               0.25,      # reverted from P11's 0.3

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

    # Fine-tune from Phase 10 best (79.33%)
    'pretrained_path':       'checkpoints/best_model_phase10.pth',

    # Early stopping
    'patience':              30,

    # Regularization (same as P10)
    'label_smoothing':       0.05,

    # SWA config — more snapshots near convergence
    'swa_start_epoch':       15,
    'swa_interval':          3,

    # Enhanced TTA — 6 views
    'use_tta':               True,
    'tta_n_augs':            6,

    # Workers
    'num_workers':           4,

    # Save
    'model_save_name':       'checkpoints/best_model_phase12.pth',
    'swa_model_save_name':   'checkpoints/best_model_phase12_swa.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Augmentation — Same as P10 (conservative)
# ════════════════════════════════════════════════════════════════════════════
def get_train_transforms():
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Affine(scale=(0.92, 1.08), translate_percent=(0.06, 0.06),
                 rotate=(-6, 6), shear=(-4, 4), p=0.5, fill=128),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=20, p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=3.0, p=1.0),
            A.Equalize(p=1.0),
        ], p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.4),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(3, 12),
                        hole_width_range=(4, 16), fill=128, p=0.3),
        A.ImageCompression(quality_range=(30, 70), p=0.2),
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
    """Enhanced TTA — 6 views."""
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
# Dataset
# ════════════════════════════════════════════════════════════════════════════
class Phase12Dataset(Dataset):
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
# Discriminative LR — Param Groups
# ════════════════════════════════════════════════════════════════════════════
def get_param_groups(model, cfg):
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
# Test-Time Augmentation
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def tta_predict(model, frames_raw, device, tta_transforms):
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
def train_one_epoch(model, loader, ctc_criterion, sr_criterion,
                    optimizer, scaler, device, epoch, cfg):
    """Train one epoch. Scheduler steps per epoch (outside this function)."""
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
    log_dir = os.path.join(Config.LOG_DIR, f'phase12_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 12 — Goldilocks LR from P10 sweet-spot + cosine decay")
    print("=" * 80)
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = Phase12Dataset(Config.DATA_ROOT, mode='train', split_ratio=0.8,
                              sr_target_h=cfg['sr_target_h'],
                              sr_target_w=cfg['sr_target_w'])
    val_ds = Phase12Dataset(Config.DATA_ROOT, mode='val', split_ratio=0.8,
                            sr_target_h=cfg['sr_target_h'],
                            sr_target_w=cfg['sr_target_w'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,
                              collate_fn=Phase12Dataset.collate_fn,
                              num_workers=cfg['num_workers'], pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, collate_fn=Phase12Dataset.collate_fn,
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

    # Load Phase 10 best checkpoint (79.33%)
    p10_path = cfg['pretrained_path']
    p10_acc = 0.0
    if os.path.exists(p10_path):
        ckpt = torch.load(p10_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        p10_acc = ckpt.get('accuracy', 0)
        p10_epoch = ckpt.get('epoch', -1) + 1
        print(f"  ✅ Loaded {p10_path}: acc={p10_acc:.2f}% (epoch {p10_epoch})")
    else:
        print(f"  ❌ {p10_path} not found! Training from scratch.")

    # ── Loss ─────────────────────────────────────────────────────────────
    ctc_criterion = FocalCTCLoss(blank=0, gamma=cfg['focal_ctc_gamma'])
    ctc_criterion_val = nn.CTCLoss(blank=0, zero_infinity=True)
    sr_criterion = SRLoss(lambda_l1=1.0,
                          lambda_perceptual=cfg['lambda_perceptual']).to(device)

    # ── Optimizer with Discriminative LR ─────────────────────────────────
    param_groups = get_param_groups(model, cfg)
    optimizer = optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])

    # ── CosineAnnealingLR — MONOTONIC DECAY from sweet-spot LR ──────────
    # Starts at: backbone=1e-5, lstm=1e-4, head=2e-4
    # Decays to: 1e-7 over 50 epochs
    # NO peaks, NO restarts — only goes DOWN
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
    best_gap = 0.0
    best_epoch = -1

    print(f"\n{'='*80}")
    print("  Starting Phase 12 — Goldilocks LR fine-tuning...")
    print(f"  {cfg['epochs']} epochs | patience={cfg['patience']}")
    print(f"  Discriminative LR: backbone={cfg['lr_backbone']:.2e}, "
          f"lstm={cfg['lr_lstm']:.2e}, head={cfg['lr_head']:.2e}")
    print(f"  Scheduler: CosineAnnealingLR → {cfg['lr_min']:.1e}")
    print(f"  NO peaks, NO restarts — monotonic decay only")
    print(f"  TTA: {cfg['use_tta']} ({cfg['tta_n_augs']} views)")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        ctc_l, sr_l = train_one_epoch(
            model, train_loader, ctc_criterion, sr_criterion,
            optimizer, scaler, device, epoch, cfg)

        val = validate(model, val_loader, ctc_criterion_val, device)

        # Step scheduler ONCE per epoch
        scheduler.step()

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
                'phase12_config': cfg,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
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
                'confidence_gap': swa_val['confidence_gap'],
                'config': model.get_model_info(),
                'phase12_config': cfg,
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
                'phase12_config': cfg,
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
        if not swa_updated:
            print(f"\n  ⚠️ No SWA snapshots collected.")
        else:
            print(f"\n  ⚠️ Not enough SWA snapshots ({swa_count}). Need >= 2.")

    # ════════════════════════════════════════════════════════════════════
    # TTA Evaluation
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
            print(f"  ✅ TTA improved! (inference-only, no model save needed)")
            best_acc = tta_acc

    writer.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Phase 12 Training Completed")
    print(f"{'='*80}")
    print(f"  Best Accuracy:       {best_acc:.2f}%")
    print(f"  Best CER:            {best_cer:.4f}")
    print(f"  Best Epoch:          {best_epoch+1}/{cfg['epochs']}")
    print(f"  SWA Snapshots:       {swa_count}")
    print(f"  Model:               {cfg['model_save_name']}")
    print(f"  Based on:            {cfg['pretrained_path']}")
    print(f"  TensorBoard:         {log_dir}")

    if p10_acc > 0:
        delta = best_acc - p10_acc
        print(f"\n  Phase 10 -> Phase 12: {p10_acc:.2f}% -> {best_acc:.2f}% "
              f"({delta:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
