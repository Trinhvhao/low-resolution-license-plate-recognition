"""
Phase 8 — SWA + Confusion-Aware Fine-tuning from Phase 4 v2.

COMPLETELY DIFFERENT direction from Phase 6:
  Phase 6 tried: format constraints + balanced sampling + position-type loss → FAILED
  Phase 8 tries: SWA + confusion-pair penalty + cosine warm restarts

Why Phase 6 failed:
  - Model stuck in local minimum, OneCycleLR fine-tune just circles around it
  - Format constraints useless (errors are within-type: digit↔digit, letter↔letter)
  - Balanced sampling didn't help the fundamental confusion problem

Phase 8 KEY innovations:
  1. SWA (Stochastic Weight Averaging)
     - Average model weights from multiple training snapshots
     - Each snapshot sits in a different local minimum
     - The average generalizes better than any single point
     - Proven +0.3-1% in literature, zero inference cost

  2. Confusion-Pair Penalty Loss
     - Extra loss term that heavily penalizes known confusions:
       8↔6, V↔Y, 5↔6, 9↔5, M↔H, Q↔O, D↔B, 0↔Q, 1↔I, B↔8
     - Applied at CTC output level — when model predicts one of a
       confusion pair and GT is the other, penalty is 3× the normal CTC
     - NOT position-type loss (that was Phase 6's failed idea)

  3. CosineAnnealingWarmRestarts (NOT OneCycleLR)
     - Cyclical LR: repeatedly warm up → cool down
     - Each cycle explores a different region of weight space
     - Model escapes local minima that OneCycleLR couldn't
     - SWA collects snapshots at end of each cycle (when LR is lowest)

  4. Calibration-aware label smoothing
     - Increase label smoothing from 0.05 → 0.10
     - All wrong predictions have confidence >0.8 (overconfident)
     - Higher smoothing reduces overconfidence → better calibration

Architecture: Same Phase3Recognizer as Phase 4 v2 (no changes needed)
Starting point: best_model_phase4_v2.pth (78.47%)
"""

import os
import sys
import math
import copy
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
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    'epochs':                60,        # shorter — SWA averages over cycles
    'batch_size':            24,
    'gradient_accumulation': 4,         # eff batch = 96
    'weight_decay':          1e-4,
    'max_grad_norm':         1.0,

    # BiLSTM (same as Phase 4 v2)
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

    # Fine-tune from Phase 4 v2
    'from_scratch':          False,
    'pretrained_path':       'checkpoints/best_model_phase4_v2.pth',

    # Early stopping
    'patience':              30,        # generous — SWA needs many cycles

    # ── SWA config ───────────────────────────────────────────────────────
    'swa_start_epoch':       10,        # start collecting SWA after epoch 10
    'swa_lr':                5e-5,      # SWA base LR (low)

    # ── Cosine Warm Restarts ─────────────────────────────────────────────
    'lr_max':                5e-4,      # max LR per cycle (lower than scratch)
    'lr_min':                1e-6,      # min LR at cycle end
    'T_0':                   10,        # cycle length (epochs)
    'T_mult':                1,         # constant cycle length

    # ── Confusion-Pair Penalty ───────────────────────────────────────────
    'use_confusion_penalty': True,
    'confusion_penalty_weight': 0.5,    # λ for confusion penalty

    # Regularization
    'label_smoothing':       0.10,      # increased from 0.05 (reduce overconfidence)

    # Focal CTC
    'focal_ctc_gamma':       2.0,

    # MixUp
    'mixup_alpha':           0.2,
    'mixup_prob':            0.3,

    # Workers
    'num_workers':           4,

    # Output
    'model_save_name':       'checkpoints/best_model_phase8.pth',
    'swa_model_save_name':   'checkpoints/best_model_phase8_swa.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Confusion Pair Penalty
# ════════════════════════════════════════════════════════════════════════════
# Top confusion pairs from error analysis (861/4000 wrong):
#   8↔6 (most common), V↔Y, 5↔6, 9↔5, M↔H, Q↔O, D↔B, 0↔Q, 1↔I, B↔8
# These are CHAR indices (1-based: digit 0=idx1, ..., 9=idx10, A=idx11, ..., Z=idx36)

def _char_to_idx(c):
    """Convert character to CTC index (1-based, 0=blank)."""
    return Config.CHAR2IDX[c]

# Confusion pairs as (idx_a, idx_b) — model frequently confuses a↔b
CONFUSION_PAIRS = [
    (_char_to_idx('8'), _char_to_idx('6')),   # 8↔6
    (_char_to_idx('V'), _char_to_idx('Y')),   # V↔Y
    (_char_to_idx('5'), _char_to_idx('6')),   # 5↔6
    (_char_to_idx('9'), _char_to_idx('5')),   # 9↔5
    (_char_to_idx('M'), _char_to_idx('H')),   # M↔H
    (_char_to_idx('Q'), _char_to_idx('O')),   # Q↔O
    (_char_to_idx('D'), _char_to_idx('B')),   # D↔B
    (_char_to_idx('0'), _char_to_idx('Q')),   # 0↔Q
    (_char_to_idx('1'), _char_to_idx('I')),   # 1↔I
    (_char_to_idx('B'), _char_to_idx('8')),   # B↔8
    (_char_to_idx('0'), _char_to_idx('O')),   # 0↔O
    (_char_to_idx('2'), _char_to_idx('Z')),   # 2↔Z
    (_char_to_idx('4'), _char_to_idx('A')),   # 4↔A
]


class ConfusionPenaltyLoss(nn.Module):
    """
    Extra penalty when model predicts one char of a confusion pair
    and the GT is the other.

    Works at CTC output timestep level:
    - For each timestep, get the predicted char (argmax)
    - Find timesteps aligned to GT chars (via CTC alignment)
    - If pred is one of a confusion pair and GT is the other,
      add a penalty: -log(prob_of_correct_char)

    This is simpler and more effective than position-type loss
    because it directly targets the SPECIFIC character confusions.
    """

    def __init__(self, confusion_pairs, num_classes=37, penalty_scale=3.0):
        super().__init__()
        self.penalty_scale = penalty_scale

        # Build confusion lookup: for each char, what chars are it confused with?
        # confusion_map[idx] = set of indices it gets confused with
        self.confusion_map = {}
        for a, b in confusion_pairs:
            if a not in self.confusion_map:
                self.confusion_map[a] = set()
            if b not in self.confusion_map:
                self.confusion_map[b] = set()
            self.confusion_map[a].add(b)
            self.confusion_map[b].add(a)

        # Build penalty matrix: [num_classes, num_classes]
        # penalty_matrix[gt_char, pred_char] = penalty_scale if confusion pair, else 1.0
        penalty_matrix = torch.ones(num_classes, num_classes)
        for a, b in confusion_pairs:
            penalty_matrix[a, b] = penalty_scale
            penalty_matrix[b, a] = penalty_scale
        self.register_buffer('penalty_matrix', penalty_matrix)

    def forward(self, log_probs, targets, target_lengths):
        """
        Args:
            log_probs: [B, T, C] — log probabilities from model
            targets: [sum(target_lengths)] — concatenated GT indices
            target_lengths: [B] — length of each GT

        Returns:
            penalty_loss: scalar — extra penalty for confusion pair errors
        """
        B, T, C = log_probs.shape
        device = log_probs.device

        total_penalty = 0.0
        n_penalties = 0
        offset = 0

        for b in range(B):
            tgt_len = target_lengths[b].item()
            gt_chars = targets[offset:offset + tgt_len]  # [tgt_len]
            offset += tgt_len

            # Simple alignment: distribute GT chars evenly across T timesteps
            # Then only compute penalty at those aligned positions
            positions = torch.linspace(0, T - 1, tgt_len).long().to(device)

            for i, pos in enumerate(positions):
                gt_idx = gt_chars[i].item()
                if gt_idx == 0:  # blank
                    continue

                # Get log probs at this position
                lp = log_probs[b, pos]  # [C]

                # Check if GT char is in a confusion pair
                if gt_idx in self.confusion_map:
                    # For each confused char, add penalty weighted by its probability
                    for confused_idx in self.confusion_map[gt_idx]:
                        # Probability model assigns to the confused char
                        p_confused = torch.exp(lp[confused_idx])
                        # Penalty: higher when model gives high prob to confused char
                        # We want: -log(1 - p_confused) ≈ penalize giving mass to wrong char
                        # But simpler: just use p_confused * scale as penalty
                        penalty = p_confused * self.penalty_scale
                        total_penalty += penalty
                        n_penalties += 1

        if n_penalties > 0:
            return total_penalty / n_penalties
        return torch.tensor(0.0, device=device, requires_grad=True)


# ════════════════════════════════════════════════════════════════════════════
# Augmentation (same as Phase 4 v2)
# ════════════════════════════════════════════════════════════════════════════
def get_train_transforms():
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.08, 0.08),
                 rotate=(-8, 8), shear=(-5, 5), p=0.6, fill=128),
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                             val_shift_limit=30, p=0.4),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.Equalize(p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),
        A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(4, 16),
                        hole_width_range=(6, 24), fill=128, p=0.4),
        A.ImageCompression(quality_range=(20, 60), p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


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
# Dataset (same as Phase 4 v2)
# ════════════════════════════════════════════════════════════════════════════
class Phase8Dataset(Dataset):
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
    """
    Stochastic Weight Averaging.

    Maintains running average of model parameters.
    Call update() at the end of each LR cycle (when LR is lowest).
    Call apply() to copy averaged weights into the model.
    """

    def __init__(self, model):
        self.n_models = 0
        # Store average as dict of tensors (CPU to save GPU memory)
        self.avg_params = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }
        # Also store BN running stats
        self.avg_buffers = {}
        for name, buf in model.named_buffers():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                self.avg_buffers[name] = buf.detach().cpu().clone()

    def update(self, model):
        """Add current model weights to running average."""
        self.n_models += 1
        for name, param in model.named_parameters():
            self.avg_params[name] += (param.detach().cpu() - self.avg_params[name]) / self.n_models
        # Update BN stats (skip num_batches_tracked — it's a Long counter)
        for name, buf in model.named_buffers():
            if name in self.avg_buffers and 'num_batches_tracked' not in name:
                self.avg_buffers[name] += (buf.detach().cpu().float() - self.avg_buffers[name].float()) / self.n_models

    def apply(self, model):
        """Copy averaged weights into the model."""
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


def update_bn(model, loader, device, num_batches=100):
    """
    Recompute BatchNorm running stats after SWA weight averaging.
    SWA averages parameters but BN stats need to be recomputed.
    """
    # Reset BN stats
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()
            module.momentum = None  # Use cumulative moving average

    model.train()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            model(images)


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, ctc_criterion, sr_criterion,
                    confusion_criterion, optimizer, scaler, device,
                    epoch, cfg):
    model.train()
    total_ctc, total_sr, total_conf = 0.0, 0.0, 0.0
    n = 0
    ga = cfg['gradient_accumulation']
    use_sr = cfg['use_sr_branch'] and model.sr_branch is not None
    use_conf = cfg['use_confusion_penalty'] and confusion_criterion is not None
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

            # Confusion pair penalty
            conf_loss = torch.tensor(0.0, device=device)
            if use_conf:
                conf_loss = confusion_criterion(preds, targets, tgt_lens)

            # SR loss
            if use_sr and sr_out is not None:
                sr_loss_val, _ = sr_criterion(sr_out, hr_images)
                loss = (ctc_loss
                        + cfg['lambda_sr'] * sr_loss_val
                        + cfg['confusion_penalty_weight'] * conf_loss
                       ) / ga
            else:
                sr_loss_val = torch.tensor(0.0)
                loss = (ctc_loss
                        + cfg['confusion_penalty_weight'] * conf_loss
                       ) / ga

        scaler_scale_before = scaler.get_scale()
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
        total_conf += conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss
        n += 1

        pbar.set_postfix({
            'ctc': f"{ctc_loss.item():.4f}",
            'conf': f"{conf_loss.item():.4f}" if isinstance(conf_loss, torch.Tensor) else "0",
            'lr':  f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    # Flush remaining gradients
    if n % ga != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=cfg['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_ctc / n, total_sr / n, total_conf / n


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

    # Per-confusion-pair accuracy
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


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG
    seed_everything(Config.SEED)
    device = Config.DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase8_{ts}')
    writer = SummaryWriter(log_dir)

    print("🔒 Đã cố định Seed:", Config.SEED)
    print("=" * 80)
    print("  PHASE 8 — SWA + Confusion-Pair Penalty + Cosine Warm Restarts")
    print("=" * 80)
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = Phase8Dataset(Config.DATA_ROOT, mode='train', split_ratio=0.8,
                             sr_target_h=cfg['sr_target_h'],
                             sr_target_w=cfg['sr_target_w'])
    val_ds = Phase8Dataset(Config.DATA_ROOT, mode='val', split_ratio=0.8,
                           sr_target_h=cfg['sr_target_h'],
                           sr_target_w=cfg['sr_target_w'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, collate_fn=Phase8Dataset.collate_fn,
                              num_workers=cfg['num_workers'], pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, collate_fn=Phase8Dataset.collate_fn,
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

    # Load Phase 4 v2 checkpoint
    p4v2_path = cfg['pretrained_path']
    p4v2_acc = 0.0
    if os.path.exists(p4v2_path):
        ckpt = torch.load(p4v2_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        p4v2_acc = ckpt.get('accuracy', 0)
        p4v2_epoch = ckpt.get('epoch', 0)
        print(f"  ✅ Loaded {p4v2_path}: acc={p4v2_acc:.2f}%, epoch={p4v2_epoch+1}")
    else:
        print(f"  ❌ {p4v2_path} not found! Training from scratch.")

    # ── Loss ─────────────────────────────────────────────────────────────
    ctc_criterion = FocalCTCLoss(blank=0, gamma=cfg['focal_ctc_gamma'])
    ctc_criterion_val = nn.CTCLoss(blank=0, zero_infinity=True)
    sr_criterion = SRLoss(lambda_l1=1.0,
                          lambda_perceptual=cfg['lambda_perceptual']).to(device)

    # Confusion-pair penalty
    confusion_criterion = None
    if cfg['use_confusion_penalty']:
        confusion_criterion = ConfusionPenaltyLoss(
            CONFUSION_PAIRS,
            num_classes=Config.NUM_CLASSES,
            penalty_scale=3.0,
        ).to(device)
        print(f"  📐 Confusion penalty: {len(CONFUSION_PAIRS)} pairs, "
              f"λ={cfg['confusion_penalty_weight']}")

    # ── Optimizer + CosineAnnealingWarmRestarts ──────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['lr_max'],
                            weight_decay=cfg['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg['T_0'],
        T_mult=cfg['T_mult'],
        eta_min=cfg['lr_min'],
    )

    scaler = GradScaler()

    # ── SWA ──────────────────────────────────────────────────────────────
    swa = SWAModel(model)
    swa_updated = False
    swa_count = 0
    print(f"  ⚡ SWA: start collecting at epoch {cfg['swa_start_epoch']}, "
          f"snapshot every {cfg['T_0']} epochs")

    # ── Training loop ────────────────────────────────────────────────────
    patience_counter = 0
    best_acc = 0.0
    best_cer = float('inf')
    best_gap = 0.0
    best_epoch = -1

    print(f"\n{'='*80}")
    print("  Starting Phase 8 fine-tuning from Phase 4 v2...")
    print(f"  {cfg['epochs']} epochs | patience={cfg['patience']} "
          f"| SWA from epoch {cfg['swa_start_epoch']}")
    print(f"  LR: {cfg['lr_max']} → {cfg['lr_min']} "
          f"(cosine restart every {cfg['T_0']} epochs)")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        ctc_l, sr_l, conf_l = train_one_epoch(
            model, train_loader, ctc_criterion, sr_criterion,
            confusion_criterion, optimizer, scaler, device, epoch, cfg)

        # Step scheduler per epoch
        scheduler.step()

        val = validate(model, val_loader, ctc_criterion_val, device)

        # ── SWA snapshot at end of each LR cycle ────────────────────────
        # Snapshot when we're at the end of a cycle (LR is near minimum)
        is_cycle_end = ((epoch + 1) % cfg['T_0'] == 0)
        if epoch >= cfg['swa_start_epoch'] and is_cycle_end:
            swa.update(model)
            swa_count += 1
            swa_updated = True
            print(f"  📸 SWA snapshot #{swa_count} collected (epoch {epoch+1})")

        # Logging
        writer.add_scalar('Loss/train_ctc', ctc_l, epoch)
        writer.add_scalar('Loss/train_sr', sr_l, epoch)
        writer.add_scalar('Loss/confusion_penalty', conf_l, epoch)
        writer.add_scalar('Loss/val', val['loss'], epoch)
        writer.add_scalar('Accuracy/val', val['accuracy'], epoch)
        writer.add_scalar('CER/val', val['cer'], epoch)
        writer.add_scalar('ConfGap/val', val['confidence_gap'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{cfg['epochs']}:")
        print(f"  Train CTC: {ctc_l:.4f} | SR: {sr_l:.4f} | Conf: {conf_l:.4f}")
        print(f"  Val   Loss: {val['loss']:.4f} | Acc: {val['accuracy']:.2f}% | "
              f"CER: {val['cer']:.4f} | Gap: {val['confidence_gap']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}"
              f" | SWA snapshots: {swa_count}")

        # Top confusion pairs this epoch
        if val['confusion_stats']:
            top_conf = sorted(val['confusion_stats'].items(),
                              key=lambda x: -x[1])[:5]
            conf_str = ', '.join(f"{k}:{v}" for k, v in top_conf)
            print(f"  Top confusions: {conf_str}")

        # Save best (non-SWA) model
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
                'phase8_config': cfg,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['model_save_name'])
            print(f"  -> Saved {cfg['model_save_name']}  "
                  f"(Acc {val['accuracy']:.2f}%, CER {val['cer']:.4f})")
            if p4v2_acc > 0:
                delta = val['accuracy'] - p4v2_acc
                print(f"     vs Phase 4 v2: {delta:+.2f}%"
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

        # Early stopping (generous for SWA)
        if patience_counter >= cfg['patience']:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(best epoch {best_epoch+1}, acc {best_acc:.2f}%)")
            break

    # ════════════════════════════════════════════════════════════════════
    # SWA: Apply averaged weights and evaluate
    # ════════════════════════════════════════════════════════════════════
    swa_acc = 0.0
    if swa_updated and swa_count >= 2:
        print(f"\n{'='*80}")
        print(f"  Applying SWA ({swa_count} snapshots)...")
        print(f"{'='*80}")

        # Save original best model's state (in case SWA is worse)
        best_ckpt = torch.load(cfg['model_save_name'], map_location=device,
                               weights_only=False)

        # Apply SWA weights
        swa.apply(model)

        # Recompute BN stats with SWA weights
        print("  Recomputing BatchNorm statistics...")
        update_bn(model, train_loader, device, num_batches=200)

        # Evaluate SWA model
        swa_val = validate(model, val_loader, ctc_criterion_val, device)
        swa_acc = swa_val['accuracy']

        print(f"  SWA Accuracy: {swa_acc:.2f}% | CER: {swa_val['cer']:.4f}")
        print(f"  Best single:  {best_acc:.2f}%")
        print(f"  Δ SWA:        {swa_acc - best_acc:+.2f}%")

        if swa_acc > best_acc:
            print(f"  ✅ SWA improved! Saving as {cfg['swa_model_save_name']}")
            torch.save({
                'epoch': -1,  # SWA averaged
                'model_state_dict': model.state_dict(),
                'accuracy': swa_acc,
                'cer': swa_val['cer'],
                'confidence_gap': swa_val['confidence_gap'],
                'config': model.get_model_info(),
                'phase8_config': cfg,
                'swa_snapshots': swa_count,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['swa_model_save_name'])

            # Also overwrite main save if SWA is better
            torch.save({
                'epoch': -1,
                'model_state_dict': model.state_dict(),
                'accuracy': swa_acc,
                'cer': swa_val['cer'],
                'confidence_gap': swa_val['confidence_gap'],
                'config': model.get_model_info(),
                'phase8_config': cfg,
                'swa_snapshots': swa_count,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['model_save_name'])
            best_acc = swa_acc
            best_cer = swa_val['cer']
        else:
            print(f"  ❌ SWA didn't improve. Keeping best single model.")
            # Restore best single model
            model.load_state_dict(best_ckpt['model_state_dict'])
    else:
        print(f"\n  ⚠️ Not enough SWA snapshots ({swa_count}). "
              f"Need at least 2.")

    writer.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Phase 8 Training Completed")
    print(f"{'='*80}")
    print(f"  Best Accuracy:       {best_acc:.2f}% "
          f"{'(SWA)' if swa_acc > 0 and swa_acc >= best_acc else '(single)'}")
    print(f"  Best CER:            {best_cer:.4f}")
    print(f"  Best Confidence Gap: {best_gap:.4f}")
    print(f"  Best Epoch:          {best_epoch+1}/{cfg['epochs']}")
    print(f"  SWA Snapshots:       {swa_count}")
    print(f"  Model:               {cfg['model_save_name']}")
    print(f"  Based on:            {cfg['pretrained_path']}")
    print(f"  TensorBoard:         {log_dir}")

    if p4v2_acc > 0:
        delta = best_acc - p4v2_acc
        print(f"\n  Phase 4 v2 -> Phase 8: {p4v2_acc:.2f}% -> {best_acc:.2f}% "
              f"({delta:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
