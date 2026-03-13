"""
Phase 2 Training Script — Full Architecture Upgrade.

Tasks:
  2.1 SpatialTemporalAttentionFusion
  2.2 Transformer Encoder (2L/4H)
  2.3 Auxiliary SR Branch (CTC + SR_Loss multi-loss)
  2.4 Multi-loss: L = CTC_loss + λ_sr × (L1 + λ_perc × Perceptual)

Transfer learning from Phase 1.
FIXES: weight_decay=1e-4, lr=0.001, OneCycleLR, no backbone freeze.
Saves best model as best_model_phase2.pth.
Exits cleanly so PM2 --no-autorestart does not restart.

Usage:
    python train_phase2.py
"""

import os
import sys
import glob
import json
import math
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import AdvancedMultiFrameDataset
from models import Phase2Recognizer, SRLoss
from utils import (
    seed_everything, decode_predictions, calculate_cer,
    calculate_accuracy, get_prediction_confidence, calculate_confidence_gap
)
from transforms import get_val_transforms

# ════════════════════════════════════════════════════════════════════════════
# Phase 2 Configuration
# ════════════════════════════════════════════════════════════════════════════
P2 = {
    # Training
    'epochs':                80,
    'batch_size':            64,
    'gradient_accumulation': 2,       # effective batch = 128
    'learning_rate':         0.001,   # FIX: was 5e-4, now same as Phase 1
    'min_lr':                1e-6,
    'warmup_epochs':         8,
    'weight_decay':          1e-4,    # FIX: was 0.05 (500x too high!)
    'max_grad_norm':         1.0,

    # Model
    'd_model':               512,
    'nhead':                 4,
    'num_transformer_layers':2,
    'dim_feedforward':       1024,
    'fusion_reduction':      16,
    'dropout':               0.1,

    # SR branch (2.3 / 2.4)
    'use_sr_branch':         True,
    'sr_target_h':           43,      # actual HR height
    'sr_target_w':           120,     # actual HR width
    'lambda_sr':             0.1,     # weight for SR loss in total loss
    'lambda_perceptual':     0.1,     # weight for perceptual inside SR loss

    # Phase 1 transfer
    'phase1_checkpoint':     'checkpoints/best_model_phase1.pth',
    'freeze_backbone_epochs':0,       # FIX: was 8 → no freeze, fine-tune from start

    # Early stopping
    'patience':              15,

    # Regularization extras
    'label_smoothing':       0.05,    # FIX: was 0.1 → reduced
    'backbone_warmup_after_unfreeze': 0,  # FIX: no warmup needed (no freeze)

    # Workers
    'num_workers':           4,
}


# ════════════════════════════════════════════════════════════════════════════
# Dataset wrapper that also returns HR images for SR loss
# ════════════════════════════════════════════════════════════════════════════
class Phase2Dataset(Dataset):
    """
    Wraps AdvancedMultiFrameDataset and also returns the first HR image
    (resized to sr_target_h × sr_target_w, normalized to [0,1]) for SR loss.
    """

    def __init__(self, base_dataset, sr_target_h=43, sr_target_w=120):
        self.base = base_dataset
        self.sr_h = sr_target_h
        self.sr_w = sr_target_w

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        images_tensor, target, target_len, label = self.base[idx]
        item = self.base.samples[idx]

        # Load HR image (mean of available HR frames → single reference)
        hr_tensor = self._load_hr(item.get('hr_paths', []))
        return images_tensor, target, target_len, label, hr_tensor

    def _load_hr(self, hr_paths):
        """Load first HR frame, resize, normalize to [0,1] tensor."""
        if hr_paths:
            img = cv2.imread(hr_paths[0])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.sr_w, self.sr_h))
                img = img.astype(np.float32) / 255.0
                return torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        # Fallback: zeros (will have near-zero gradient)
        return torch.zeros(3, self.sr_h, self.sr_w)

    @staticmethod
    def collate_fn(batch):
        images, targets, target_lengths, labels_text, hr_images = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        hr_images = torch.stack(hr_images, 0)
        return images, targets, target_lengths, labels_text, hr_images


# (CosineWarmup removed — using OneCycleLR instead)


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, ctc_criterion, sr_criterion, optimizer,
                    scaler, scheduler, device, epoch, cfg):
    model.train()
    total_ctc, total_sr, total_loss_sum = 0.0, 0.0, 0.0
    n = 0
    ga = cfg['gradient_accumulation']
    use_sr = cfg['use_sr_branch'] and model.sr_branch is not None
    label_smooth = cfg.get('label_smoothing', 0.0)

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
    for bi, batch in enumerate(pbar):
        if use_sr:
            images, targets, tgt_lens, _, hr_images = batch
            hr_images = hr_images.to(device, non_blocking=True)
        else:
            images, targets, tgt_lens, _ = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast('cuda'):
            if use_sr:
                preds, sr_out = model(images, return_sr=True)
            else:
                preds = model(images)

            T_out = preds.size(1)
            input_lengths = torch.full((images.size(0),), T_out, dtype=torch.long)
            ctc_loss = ctc_criterion(preds.permute(1,0,2), targets,
                                      input_lengths, tgt_lens)

            # Label smoothing: penalise overconfident predictions
            # Equivalent to KL(uniform || output) = -H(output) + const
            if label_smooth > 0:
                # preds is log_softmax → mean is ≈ -entropy
                confidence_penalty = -preds.mean()  # maximise entropy
                ctc_loss = ctc_loss + label_smooth * confidence_penalty

            if use_sr and sr_out is not None:
                sr_loss_val, (l1_v, perc_v) = sr_criterion(sr_out, hr_images)
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
            # OneCycleLR: only step if optimizer actually stepped
            if scaler.get_scale() >= scaler_scale_before:
                scheduler.step()

        total_ctc += ctc_loss.item()
        total_sr  += sr_loss_val.item()
        total_loss_sum += loss.item() * ga
        n += 1

        pbar.set_postfix({
            'ctc':  f"{ctc_loss.item():.4f}",
            'sr':   f"{sr_loss_val.item():.4f}",
            'lr':   f"{scheduler.get_last_lr()[0]:.2e}"
        })

    # Flush remaining gradients
    if n % ga != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=cfg['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_ctc / n, total_sr / n, total_loss_sum / n


# ════════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, ctc_criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds, all_targets, all_confs = [], [], []

    for batch in loader:
        # Val loader may have 4 or 5 items (Phase2Dataset also returns HR)
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        tgt_lens = batch[2]
        labels_text = batch[3]

        preds = model(images)   # no SR at validation

        T_out = preds.size(1)
        input_lengths = torch.full((images.size(0),), T_out, dtype=torch.long)
        loss = ctc_criterion(preds.permute(1,0,2), targets, input_lengths, tgt_lens)
        val_loss += loss.item()

        decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)
        all_preds.extend(decoded)
        all_targets.extend(labels_text)
        all_confs.extend(get_prediction_confidence(preds).tolist())

    avg_loss = val_loss / len(loader)
    acc  = calculate_accuracy(all_preds, all_targets) * 100
    cer  = calculate_cer(all_preds, all_targets)
    is_c = [p == t for p, t in zip(all_preds, all_targets)]
    gap  = calculate_confidence_gap(all_confs, is_c)

    return {
        'loss': avg_loss, 'accuracy': acc, 'cer': cer,
        'confidence_gap': gap, 'outputs/predictions/predictions': all_preds, 'targets': all_targets
    }


# ════════════════════════════════════════════════════════════════════════════
# Freeze / unfreeze helpers
# ════════════════════════════════════════════════════════════════════════════
def freeze_backbone(model, freeze=True):
    for name, param in model.named_parameters():
        if 'stn' in name or 'backbone' in name:
            param.requires_grad = not freeze
    tag = "FROZEN" if freeze else "UNFROZEN"
    print(f"   🧊 STN + Backbone: {tag}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = P2
    seed_everything(Config.SEED)
    device = Config.DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase2_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 2 TRAINING — Full Architecture Upgrade")
    print("=" * 80)
    for k, v in cfg.items():
        print(f"   {k}: {v}")
    print(f"\n   Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    base_train = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='train',
                                           split_ratio=0.8)
    base_val   = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='val',
                                           split_ratio=0.8)

    train_ds = Phase2Dataset(base_train,
                             sr_target_h=cfg['sr_target_h'],
                             sr_target_w=cfg['sr_target_w'])
    val_ds   = Phase2Dataset(base_val,
                             sr_target_h=cfg['sr_target_h'],
                             sr_target_w=cfg['sr_target_w'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,
                              collate_fn=Phase2Dataset.collate_fn,
                              num_workers=cfg['num_workers'],
                              pin_memory=True, persistent_workers=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False,
                            collate_fn=Phase2Dataset.collate_fn,
                            num_workers=cfg['num_workers'],
                            pin_memory=True, persistent_workers=True)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = Phase2Recognizer(
        num_classes=Config.NUM_CLASSES,
        use_stn=Config.USE_STN,
        use_resnet_backbone=Config.USE_RESNET_BACKBONE,
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_transformer_layers=cfg['num_transformer_layers'],
        dim_feedforward=cfg['dim_feedforward'],
        fusion_reduction=cfg['fusion_reduction'],
        dropout=cfg['dropout'],
        use_sr_branch=cfg['use_sr_branch'],
        sr_target_h=cfg['sr_target_h'],
        sr_target_w=cfg['sr_target_w'],
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"   Params: {total_p:,} ({total_p*4/1024/1024:.1f} MB)")
    print(f"   Info: {model.get_model_info()}")

    # ── Phase 1 transfer ─────────────────────────────────────────────────
    if os.path.exists(cfg['phase1_checkpoint']):
        ckpt = model.load_phase1_weights(cfg['phase1_checkpoint'], device=device)
        print(f"   Phase 1 acc: {ckpt.get('accuracy',0):.2f}%")
        # FIX: No backbone freeze — fine-tune all layers from start
        if cfg['freeze_backbone_epochs'] > 0:
            freeze_backbone(model, freeze=True)
    else:
        print(f"   Phase 1 checkpoint not found — training from scratch")

    # ── Loss ──────────────────────────────────────────────────────────────
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sr_criterion  = SRLoss(lambda_l1=1.0,
                           lambda_perceptual=cfg['lambda_perceptual']).to(device)

    # ── Optimizer (discriminative LR) ─────────────────────────────────────
    # STN + backbone = pretrained from Phase 1 → lower LR
    # fusion + transformer + fc + sr_branch = new/random → higher LR
    pretrained_prefixes = ('stn.', 'backbone.')
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in pretrained_prefixes):
            pretrained_params.append(param)
        else:
            new_params.append(param)
    print(f"   Discriminative LR: {len(pretrained_params)} pretrained, "
          f"{len(new_params)} new params")

    lr_pretrained = cfg['learning_rate'] * 0.1   # 1e-4 for pretrained
    lr_new        = cfg['learning_rate']          # 1e-3 for new

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': lr_pretrained},
        {'params': new_params,        'lr': lr_new},
    ], weight_decay=cfg['weight_decay'])

    # FIX: OneCycleLR with gentler warmup (30%)
    spe = len(train_loader) // cfg['gradient_accumulation']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr_pretrained, lr_new],
        steps_per_epoch=spe,
        epochs=cfg['epochs'],
        pct_start=0.3,
        anneal_strategy='cos',
    )
    scaler = GradScaler()

    # ── Early stopping ────────────────────────────────────────────────────
    patience_counter = 0
    best_acc = 0.0
    best_cer = float('inf')
    best_gap = 0.0
    best_epoch = -1

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("   Starting training...")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        # Unfreeze backbone after warmup (only if freeze_backbone_epochs > 0)
        if cfg['freeze_backbone_epochs'] > 0 and epoch == cfg['freeze_backbone_epochs']:
            print(f"\n   Epoch {epoch+1}: Unfreezing backbone")
            freeze_backbone(model, freeze=False)

        ctc_l, sr_l, total_l = train_one_epoch(
            model, train_loader, ctc_criterion, sr_criterion,
            optimizer, scaler, scheduler, device, epoch, cfg)

        val = validate(model, val_loader, ctc_criterion, device)

        # Logging
        writer.add_scalar('Loss/train_ctc', ctc_l, epoch)
        writer.add_scalar('Loss/train_sr',  sr_l,  epoch)
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
            best_acc  = val['accuracy']
            best_cer  = val['cer']
            best_gap  = val['confidence_gap']
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy':         val['accuracy'],
                'cer':              val['cer'],
                'confidence_gap':   val['confidence_gap'],
                'config':           model.get_model_info(),
                'phase2_config':    cfg,
            }, 'checkpoints/best_model_phase2_fixed.pth')
            print(f"  -> Saved best_model_phase2_fixed.pth  "
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
    print(" Phase 2 Training Completed")
    print(f"{'='*80}")
    print(f"   Best Accuracy:       {best_acc:.2f}%")
    print(f"   Best CER:            {best_cer:.4f}")
    print(f"   Best Confidence Gap: {best_gap:.4f}")
    print(f"   Best Epoch:          {best_epoch+1}")
    print(f"   Model:               best_model_phase2_fixed.pth")
    print(f"   TensorBoard:         {log_dir}")

    if os.path.exists(cfg['phase1_checkpoint']):
        c = torch.load(cfg['phase1_checkpoint'], map_location='cpu',
                        weights_only=False)
        p1 = c.get('accuracy', 0)
        print(f"\n   Phase 1 -> Phase 2: {p1:.2f}% -> {best_acc:.2f}% "
              f"({best_acc - p1:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
