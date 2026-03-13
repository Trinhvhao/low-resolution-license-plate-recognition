"""
Phase 4 v2 — Retrain from scratch with 100 epochs + early stopping.

Same as original Phase 4 (78.08% at epoch 60/60, still improving) but:
  - 100 epochs instead of 60 (more room to converge)
  - patience=25 (was 12 — rarely triggered, now more generous)
  - Saves to best_model_phase4_v2.pth (preserves original best_model_phase4.pth)
  - batch=24 + GA=4 (shares GPU with Phase 5)

Architecture: Phase3Recognizer (BiLSTM + SpatialTemporalFusion + SR Branch)
From scratch (ImageNet backbone only).
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
    'epochs':                100,       # was 60 — more room to converge
    'batch_size':            24,        # lower for GPU sharing
    'gradient_accumulation': 4,         # eff batch = 96 (same as original)
    'learning_rate':         0.001,
    'weight_decay':          1e-4,
    'max_grad_norm':         1.0,

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

    # From scratch
    'from_scratch':          True,

    # Early stopping
    'patience':              25,        # was 12 — more generous

    # Regularization
    'label_smoothing':       0.05,

    # Focal CTC
    'focal_ctc_gamma':       2.0,

    # MixUp
    'mixup_alpha':           0.2,
    'mixup_prob':            0.3,

    # Workers
    'num_workers':           4,

    # Output — separate from original!
    'model_save_name':       'checkpoints/best_model_phase4_v2.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Augmentation (identical to Phase 4)
# ════════════════════════════════════════════════════════════════════════════
def get_strong_train_transforms():
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


def get_strong_val_transforms():
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_strong_degradation_transforms():
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
# Dataset (identical to Phase 4)
# ════════════════════════════════════════════════════════════════════════════
class Phase4Dataset(Dataset):
    def __init__(self, root_dir, mode='train', split_ratio=0.8,
                 sr_target_h=43, sr_target_w=120):
        self.mode = mode
        self.sr_h = sr_target_h
        self.sr_w = sr_target_w
        if mode == 'train':
            self.transform = get_strong_train_transforms()
            self.degrade = get_strong_degradation_transforms()
        else:
            self.transform = get_strong_val_transforms()
            self.degrade = None
        base = AdvancedMultiFrameDataset(root_dir, mode=mode, split_ratio=split_ratio)
        self.samples = base.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']
        use_hr = (self.mode == 'train' and len(item['hr_paths']) > 0
                  and random.random() < 0.5)
        if use_hr:
            images_list = self._load_frames(item['hr_paths'], apply_degradation=True)
        else:
            images_list = self._load_frames(item['lr_paths'], apply_degradation=False)
        images_tensor = torch.stack(images_list, dim=0)
        target = [Config.CHAR2IDX[c] for c in label if c in Config.CHAR2IDX]
        if len(target) == 0:
            target = [0]
        hr_tensor = self._load_hr(item.get('hr_paths', []))
        return images_tensor, torch.tensor(target, dtype=torch.long), len(target), label, hr_tensor

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
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction='none')

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        with torch.no_grad():
            p_easy = torch.exp(-ctc_loss.detach())
            focal_weight = (1.0 - p_easy) ** self.gamma
        return (focal_weight * ctc_loss).mean()


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, ctc_criterion, sr_criterion, optimizer,
                    scaler, scheduler, device, epoch, cfg):
    model.train()
    total_ctc, total_sr, total_loss_sum = 0.0, 0.0, 0.0
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
            input_lengths = torch.full((images.size(0),), T_out, dtype=torch.long)
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

        scaler_scale_before = scaler.get_scale()
        scaler.scale(loss).backward()

        if (bi + 1) % ga == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scaler_scale_before:
                scheduler.step()

        total_ctc += ctc_loss.item()
        total_sr += (sr_loss_val.item() if isinstance(sr_loss_val, torch.Tensor)
                     else sr_loss_val)
        total_loss_sum += loss.item() * ga
        n += 1

        pbar.set_postfix({
            'ctc': f"{ctc_loss.item():.4f}",
            'sr':  f"{(sr_loss_val.item() if isinstance(sr_loss_val, torch.Tensor) else sr_loss_val):.4f}",
            'lr':  f"{scheduler.get_last_lr()[0]:.2e}",
        })

    # Flush remaining
    if n % ga != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=cfg['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_ctc / n, total_sr / n, total_loss_sum / n


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

        decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)
        all_preds.extend(decoded)
        all_targets.extend(labels_text)
        all_confs.extend(get_prediction_confidence(preds).tolist())

    avg_loss = val_loss / len(loader)
    acc = calculate_accuracy(all_preds, all_targets) * 100
    cer = calculate_cer(all_preds, all_targets)
    is_c = [p == t for p, t in zip(all_preds, all_targets)]
    gap = calculate_confidence_gap(all_confs, is_c)
    return {'loss': avg_loss, 'accuracy': acc, 'cer': cer,
            'confidence_gap': gap, 'outputs/predictions/predictions': all_preds,
            'targets': all_targets}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG
    seed_everything(Config.SEED)
    device = Config.DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase4_v2_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 4 v2 — Retrain from scratch, 100 epochs, patience=25")
    print("=" * 80)
    print(f"  Image: {IMG_HEIGHT}×{IMG_WIDTH}")
    print(f"  Saves to: {cfg['model_save_name']}  (original best_model_phase4.pth preserved)")
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = Phase4Dataset(Config.DATA_ROOT, mode='train', split_ratio=0.8,
                             sr_target_h=cfg['sr_target_h'],
                             sr_target_w=cfg['sr_target_w'])
    val_ds = Phase4Dataset(Config.DATA_ROOT, mode='val', split_ratio=0.8,
                           sr_target_h=cfg['sr_target_h'],
                           sr_target_w=cfg['sr_target_w'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, collate_fn=Phase4Dataset.collate_fn,
                              num_workers=cfg['num_workers'], pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, collate_fn=Phase4Dataset.collate_fn,
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
    print(f"  Training from scratch (ImageNet backbone only)")

    # ── Loss ─────────────────────────────────────────────────────────────
    ctc_criterion = FocalCTCLoss(blank=0, gamma=cfg['focal_ctc_gamma'])
    ctc_criterion_val = nn.CTCLoss(blank=0, zero_infinity=True)
    sr_criterion = SRLoss(lambda_l1=1.0,
                          lambda_perceptual=cfg['lambda_perceptual']).to(device)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])

    spe = len(train_loader) // cfg['gradient_accumulation']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg['learning_rate'],
        steps_per_epoch=spe,
        epochs=cfg['epochs'],
        pct_start=0.3,
        anneal_strategy='cos',
    )
    scaler = GradScaler()

    # ── Training loop ────────────────────────────────────────────────────
    patience_counter = 0
    best_acc = 0.0
    best_cer = float('inf')
    best_gap = 0.0
    best_epoch = -1

    print(f"\n{'='*80}")
    print("  Starting training from scratch...")
    print(f"  100 epochs | patience=25 | saves to {cfg['model_save_name']}")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        ctc_l, sr_l, total_l = train_one_epoch(
            model, train_loader, ctc_criterion, sr_criterion,
            optimizer, scaler, scheduler, device, epoch, cfg)

        val = validate(model, val_loader, ctc_criterion_val, device)

        # Logging
        writer.add_scalar('Loss/train_ctc', ctc_l, epoch)
        writer.add_scalar('Loss/train_sr', sr_l, epoch)
        writer.add_scalar('Loss/train_total', total_l, epoch)
        writer.add_scalar('Loss/val', val['loss'], epoch)
        writer.add_scalar('Accuracy/val', val['accuracy'], epoch)
        writer.add_scalar('CER/val', val['cer'], epoch)
        writer.add_scalar('ConfGap/val', val['confidence_gap'], epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1}/{cfg['epochs']}:")
        print(f"  Train CTC: {ctc_l:.4f} | SR: {sr_l:.4f} | Total: {total_l:.4f}")
        print(f"  Val   Loss: {val['loss']:.4f} | Acc: {val['accuracy']:.2f}% | "
              f"CER: {val['cer']:.4f} | Gap: {val['confidence_gap']:.4f}")

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
                'phase4_config': cfg,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
            }, cfg['model_save_name'])
            print(f"  -> Saved {cfg['model_save_name']}  "
                  f"(Acc {val['accuracy']:.2f}%, CER {val['cer']:.4f})")
        else:
            patience_counter += 1

        # Samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n  Sample predictions (epoch {epoch+1}):")
            for i in range(min(5, len(val['outputs/predictions/predictions']))):
                p, t = val['outputs/predictions/predictions'][i], val['targets'][i]
                m = "+" if p == t else "x"
                print(f"    [{m}] GT: {t:8s}  Pred: {p:8s}")
            print()

        # Early stopping
        if patience_counter >= cfg['patience']:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(best epoch {best_epoch+1}, acc {best_acc:.2f}%)")
            break

    writer.close()

    print(f"\n{'='*80}")
    print("  Phase 4 v2 Training Completed")
    print(f"{'='*80}")
    print(f"  Best Accuracy:       {best_acc:.2f}%")
    print(f"  Best CER:            {best_cer:.4f}")
    print(f"  Best Confidence Gap: {best_gap:.4f}")
    print(f"  Best Epoch:          {best_epoch+1}/{cfg['epochs']}")
    print(f"  Model:               {cfg['model_save_name']}")
    print(f"  Original Phase 4:    best_model_phase4.pth (preserved)")
    print(f"  TensorBoard:         {log_dir}")

    # Compare with original Phase 4
    p4_path = 'checkpoints/best_model_phase4.pth'
    if os.path.exists(p4_path):
        c = torch.load(p4_path, map_location='cpu', weights_only=False)
        p4_acc = c.get('accuracy', 0)
        print(f"\n  Phase 4 original -> Phase 4 v2: {p4_acc:.2f}% -> {best_acc:.2f}% "
              f"({best_acc - p4_acc:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
