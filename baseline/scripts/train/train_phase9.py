"""
Phase 9 — SWA + Scenario-Balanced Sampling + Enhanced Augmentation.

Fine-tune from Phase 4 v2 (78.47%).

Key changes from Phase 8 (79.15% SWA):
  ✅ KEEP:  SWA (proven +0.82%)
  ✅ KEEP:  CosineAnnealingWarmRestarts (T_0=10)
  ✅ KEEP:  Focal CTC Loss
  ✅ KEEP:  SR branch

  ❌ REMOVE: ConfusionPenaltyLoss (caused A_Brazilian -6.11% regression)
  ❌ REMOVE: label_smoothing=0.10 (too aggressive → back to 0.05)

  🆕 ADD:   Scenario-balanced WeightedRandomSampler
            — B_Brazilian only 2000 samples (10%) but worst scenario (54%)
            — Oversample minority scenarios for balanced learning
  🆕 ADD:   Enhanced augmentation
            — GridDistortion (plate bending/curvature)
            — ElasticTransform (warp/deformation)
            — Sharpen (improve character edge definition)
            — ChannelShuffle (color invariance)
  🆕 ADD:   Lower LR (3e-4 vs 5e-4) — more stable fine-tuning
  🆕 ADD:   80 epochs + patience=40 (more SWA snapshots, up to 7)

Expected: Recover A_Brazilian from regression + improve B_Brazilian via oversampling.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from collections import Counter
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
    'epochs':                80,        # more room for SWA (was 60)
    'batch_size':            48,
    'gradient_accumulation': 2,         # eff batch = 96
    'lr_max':                3e-4,      # lower than P8 (was 5e-4) for stability
    'lr_min':                1e-6,
    'weight_decay':          1e-4,
    'max_grad_norm':         1.0,

    # Cosine warm restarts
    'T_0':                   10,        # same as P8
    'T_mult':                1,

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

    # Fine-tune from Phase 4 v2
    'pretrained_path':       'checkpoints/best_model_phase4_v2.pth',

    # Early stopping
    'patience':              40,        # generous for SWA (was 30)

    # Regularization
    'label_smoothing':       0.05,      # back to P4v2 level (was 0.10 in P8)

    # SWA config
    'swa_start_epoch':       10,        # start collecting after epoch 10
    'swa_lr':                5e-5,

    # Scenario balancing
    'balance_scenarios':     True,      # NEW: oversample minority scenarios

    # Workers
    'num_workers':           4,

    # Save
    'model_save_name':       'checkpoints/best_model_phase9.pth',
    'swa_model_save_name':   'checkpoints/best_model_phase9_swa.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Augmentation — Enhanced from Phase 4 v2 / Phase 8
# ════════════════════════════════════════════════════════════════════════════
def get_train_transforms():
    """Enhanced training augmentation with grid distortion + elastic + sharpen."""
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),

        # ── Geometric ────────────────────────────────────────────────────
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.08, 0.08),
                 rotate=(-8, 8), shear=(-5, 5), p=0.6, fill=128),
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        # NEW: simulate plate curvature / bending
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
        # NEW: simulate warping
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),

        # ── Color / Brightness ───────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                             val_shift_limit=30, p=0.4),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.Equalize(p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.3),
        # NEW: color invariance
        A.ChannelShuffle(p=0.15),

        # ── Blur / Noise ─────────────────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),

        # NEW: sharpen to improve character edge definition
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.25),

        # ── Occlusion / Compression ──────────────────────────────────────
        A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(4, 16),
                        hole_width_range=(6, 24), fill=128, p=0.4),
        A.ImageCompression(quality_range=(20, 60), p=0.3),

        # ── Normalize ────────────────────────────────────────────────────
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
    """Degradation for HR→LR simulation."""
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
# Dataset with Scenario Tracking
# ════════════════════════════════════════════════════════════════════════════
class Phase9Dataset(Dataset):
    """
    Like Phase8Dataset but tracks scenario per sample for balanced sampling.
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

        base = AdvancedMultiFrameDataset(root_dir, mode=mode,
                                         split_ratio=split_ratio)
        self.samples = base.samples

        # Extract scenario from track path
        self.scenarios = []
        for item in self.samples:
            scenario = self._extract_scenario(item['lr_paths'][0])
            self.scenarios.append(scenario)

        # Print scenario distribution
        dist = Counter(self.scenarios)
        print(f"  [{mode.upper()}] Scenario distribution:")
        for sc, cnt in sorted(dist.items()):
            print(f"    {sc}: {cnt} ({100*cnt/len(self.samples):.1f}%)")

    @staticmethod
    def _extract_scenario(path):
        """Extract scenario name from path like .../Scenario-A/Brazilian/track_xxxxx/..."""
        parts = path.replace('\\', '/').split('/')
        for i, part in enumerate(parts):
            if part.startswith('Scenario-'):
                scenario_letter = part.split('-')[1]  # 'A' or 'B'
                if i + 1 < len(parts):
                    country = parts[i + 1]  # 'Brazilian' or 'Mercosur'
                    return f"{scenario_letter}_{country}"
        return "unknown"

    def get_sample_weights(self):
        """
        Compute per-sample weights for WeightedRandomSampler.
        Inverse-frequency weighting → each scenario gets equal total weight.
        """
        dist = Counter(self.scenarios)
        total = len(self.scenarios)
        n_scenarios = len(dist)

        # Each scenario should have equal total probability
        # weight_i = total / (n_scenarios * count_i)
        weights = []
        for sc in self.scenarios:
            w = total / (n_scenarios * dist[sc])
            weights.append(w)

        return weights

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
    """
    Stochastic Weight Averaging.
    Maintains running average of model parameters.
    Call update() at the end of each LR cycle (when LR is lowest).
    Call apply() to copy averaged weights into the model.
    """

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
        """Add current model weights to running average."""
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


def update_bn(model, loader, device, num_batches=200):
    """Recompute BatchNorm running stats after SWA weight averaging."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()
            module.momentum = None  # cumulative moving average

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
                    optimizer, scaler, device, epoch, cfg):
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

            # SR loss
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

    # Per-scenario accuracy (for monitoring)
    scenario_stats = {}
    for gt, pred, target_text in zip(all_targets, all_preds, all_targets):
        correct = (pred == gt)
        # We don't have scenario info in val loader, so just track overall
        pass

    # Confusion stats
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
    log_dir = os.path.join(Config.LOG_DIR, f'phase9_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 9 — SWA + Scenario-Balanced + Enhanced Augmentation")
    print("=" * 80)
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = Phase9Dataset(Config.DATA_ROOT, mode='train', split_ratio=0.8,
                             sr_target_h=cfg['sr_target_h'],
                             sr_target_w=cfg['sr_target_w'])
    val_ds = Phase9Dataset(Config.DATA_ROOT, mode='val', split_ratio=0.8,
                           sr_target_h=cfg['sr_target_h'],
                           sr_target_w=cfg['sr_target_w'])

    # ── Scenario-Balanced Sampler ────────────────────────────────────────
    if cfg['balance_scenarios']:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),  # same epoch size
            replacement=True,
        )
        print(f"\n  ⚖️  Scenario-balanced sampling enabled")
        print(f"      Total train: {len(train_ds)} → sampled: {len(train_ds)} per epoch")
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                                  sampler=sampler,
                                  collate_fn=Phase9Dataset.collate_fn,
                                  num_workers=cfg['num_workers'], pin_memory=True,
                                  persistent_workers=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                                  shuffle=True,
                                  collate_fn=Phase9Dataset.collate_fn,
                                  num_workers=cfg['num_workers'], pin_memory=True,
                                  persistent_workers=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, collate_fn=Phase9Dataset.collate_fn,
                            num_workers=cfg['num_workers'], pin_memory=True,
                            persistent_workers=True)

    # Also create unweighted train loader for BN update (avoid distribution shift)
    bn_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                           shuffle=True, collate_fn=Phase9Dataset.collate_fn,
                           num_workers=cfg['num_workers'], pin_memory=True,
                           persistent_workers=False, drop_last=False)

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
    print("  Starting Phase 9 fine-tuning from Phase 4 v2...")
    print(f"  {cfg['epochs']} epochs | patience={cfg['patience']} "
          f"| SWA from epoch {cfg['swa_start_epoch']}")
    print(f"  LR: {cfg['lr_max']} → {cfg['lr_min']} "
          f"(cosine restart every {cfg['T_0']} epochs)")
    print(f"  Scenario-balanced: {cfg['balance_scenarios']}")
    print(f"  New augs: GridDistortion, ElasticTransform, Sharpen, ChannelShuffle")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        ctc_l, sr_l = train_one_epoch(
            model, train_loader, ctc_criterion, sr_criterion,
            optimizer, scaler, device, epoch, cfg)

        scheduler.step()

        val = validate(model, val_loader, ctc_criterion_val, device)

        # ── SWA snapshot at end of each LR cycle ────────────────────────
        is_cycle_end = ((epoch + 1) % cfg['T_0'] == 0)
        if epoch >= cfg['swa_start_epoch'] and is_cycle_end:
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
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{cfg['epochs']}:")
        print(f"  Train CTC: {ctc_l:.4f} | SR: {sr_l:.4f}")
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
                'phase9_config': cfg,
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

        # Early stopping
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

        best_ckpt = torch.load(cfg['model_save_name'], map_location=device,
                               weights_only=False)

        # Apply SWA weights
        swa.apply(model)

        # Recompute BN stats with UNWEIGHTED loader (natural distribution)
        print("  Recomputing BatchNorm statistics (unweighted loader)...")
        update_bn(model, bn_loader, device, num_batches=200)

        # Evaluate SWA model
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
                'phase9_config': cfg,
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
                'phase9_config': cfg,
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
        print(f"\n  ⚠️ Not enough SWA snapshots ({swa_count}). "
              f"Need at least 2.")

    writer.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Phase 9 Training Completed")
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
        print(f"\n  Phase 4 v2 -> Phase 9: {p4v2_acc:.2f}% -> {best_acc:.2f}% "
              f"({delta:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
