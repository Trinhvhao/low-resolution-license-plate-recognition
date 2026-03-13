"""
Phase 3 Training — Best of Phase 1 + Phase 2.

Fixes applied vs Phase 2:
  1. BiLSTM replaces Transformer (proven better for seq_len=20)
  2. weight_decay = 1e-4 (was 0.05, 500x too high)
  3. OneCycleLR replaces CosineAnnealing (better exploration)
  4. lr = 0.001 (was 5e-4)
  5. No backbone freeze — fine-tune all from start
  6. Transfer rnn+fc from Phase 1 (same architecture)

Keeps from Phase 2:
  - SpatialTemporalAttentionFusion (better than simple AttentionFusion)
  - AuxSRBranch (training-only, forces backbone to learn HR features)
  - Gradient accumulation for effective batch=192
  - Gradient clipping
  - Label smoothing (0.05, reduced from 0.1)
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from dataset import AdvancedMultiFrameDataset
from utils import (decode_predictions, calculate_accuracy, calculate_cer,
                   get_prediction_confidence, calculate_confidence_gap,
                   seed_everything)
from transforms import get_val_transforms
from models.recognizer_v3 import Phase3Recognizer
from models.sr_branch import SRLoss

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════
P3 = {
    # Training
    'epochs':                80,
    'batch_size':            96,       # same as Phase 1
    'gradient_accumulation': 2,        # effective batch = 192
    'lr_pretrained':         1e-4,     # low lr for pre-trained (stn/backbone/rnn/fc)
    'lr_new':                5e-4,     # higher lr for new modules (fusion/sr_branch)
    'weight_decay':          1e-4,     # same as Phase 1 (was 0.05 in Phase 2!)
    'max_grad_norm':         1.0,

    # BiLSTM (same as Phase 1)
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

    # Phase 1 transfer
    'phase1_checkpoint':     'checkpoints/best_model_phase1.pth',

    # Early stopping
    'patience':              15,

    # Regularization
    'label_smoothing':       0.05,     # reduced from 0.1

    # Workers
    'num_workers':           4,

    # Output
    'model_save_name':       'checkpoints/best_model_phase3.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Dataset wrapper (adds HR images for SR loss)
# ════════════════════════════════════════════════════════════════════════════
class Phase3Dataset(Dataset):
    def __init__(self, base_dataset, sr_target_h=43, sr_target_w=120):
        self.base = base_dataset
        self.sr_h = sr_target_h
        self.sr_w = sr_target_w

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        images_tensor, target, target_len, label = self.base[idx]
        item = self.base.samples[idx]
        hr_tensor = self._load_hr(item.get('hr_paths', []))
        return images_tensor, target, target_len, label, hr_tensor

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
            ctc_loss = ctc_criterion(preds.permute(1, 0, 2), targets,
                                     input_lengths, tgt_lens)

            # Label smoothing
            if label_smooth > 0:
                confidence_penalty = -preds.mean()
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
            # OneCycleLR steps per batch (like Phase 1)
            if scaler.get_scale() >= scaler_scale_before:
                scheduler.step()

        total_ctc += ctc_loss.item()
        total_sr  += sr_loss_val.item() if isinstance(sr_loss_val, torch.Tensor) else sr_loss_val
        total_loss_sum += loss.item() * ga
        n += 1

        pbar.set_postfix({
            'ctc':  f"{ctc_loss.item():.4f}",
            'sr':   f"{sr_loss_val.item() if isinstance(sr_loss_val, torch.Tensor) else sr_loss_val:.4f}",
            'lr':   f"{scheduler.get_last_lr()[0]:.2e}"
        })

    # Flush remaining gradients
    if n % ga != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=cfg['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

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
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        tgt_lens = batch[2]
        labels_text = batch[3]

        preds = model(images)

        T_out = preds.size(1)
        input_lengths = torch.full((images.size(0),), T_out, dtype=torch.long)
        loss = ctc_criterion(preds.permute(1, 0, 2), targets, input_lengths, tgt_lens)
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
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = P3
    save_name = cfg['model_save_name']
    seed_everything(Config.SEED)
    device = Config.DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase3_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 3 TRAINING — BiLSTM + SpatialTemporalFusion + SR Branch")
    print("=" * 80)
    print("  Fixes vs Phase 2:")
    print("    1. BiLSTM replaces Transformer (better for short sequences)")
    print("    2. weight_decay = 1e-4 (was 0.05)")
    print("    3. OneCycleLR with discriminative LR (pretrained=1e-4, new=5e-4)")
    print("    4. No backbone freeze — fine-tune from start")
    print("    5. Transfer rnn+fc from Phase 1 (same BiLSTM)")
    print("    6. pct_start=0.3 (gentler warmup for fine-tuning)")
    print("=" * 80)
    for k, v in cfg.items():
        print(f"   {k}: {v}")
    print(f"\n   Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    base_train = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='train',
                                           split_ratio=0.8)
    base_val   = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='val',
                                           split_ratio=0.8)

    train_ds = Phase3Dataset(base_train,
                             sr_target_h=cfg['sr_target_h'],
                             sr_target_w=cfg['sr_target_w'])
    val_ds   = Phase3Dataset(base_val,
                             sr_target_h=cfg['sr_target_h'],
                             sr_target_w=cfg['sr_target_w'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,
                              collate_fn=Phase3Dataset.collate_fn,
                              num_workers=cfg['num_workers'],
                              pin_memory=True, persistent_workers=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False,
                            collate_fn=Phase3Dataset.collate_fn,
                            num_workers=cfg['num_workers'],
                            pin_memory=True, persistent_workers=True)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"   Steps/epoch: {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
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
    print(f"   Params: {total_p:,} ({total_p * 4 / 1024 / 1024:.1f} MB)")
    print(f"   Info: {model.get_model_info()}")

    # ── Phase 1 transfer (stn + backbone + rnn + fc) ─────────────────────
    if os.path.exists(cfg['phase1_checkpoint']):
        ckpt = model.load_phase1_weights(cfg['phase1_checkpoint'], device=device)
        print(f"   Phase 1 accuracy: {ckpt.get('accuracy', 0):.2f}%")
    else:
        print(f"   ⚠️ Phase 1 checkpoint not found — training from scratch")

    # ── Loss ──────────────────────────────────────────────────────────────
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sr_criterion  = SRLoss(lambda_l1=1.0,
                           lambda_perceptual=cfg['lambda_perceptual']).to(device)

    # ── Optimizer (discriminative LR: low for pre-trained, high for new) ─
    pretrained_names = {'stn', 'backbone', 'rnn', 'fc'}
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module_name = name.split('.')[0]
        if module_name in pretrained_names:
            pretrained_params.append(param)
        else:
            new_params.append(param)

    print(f"   Param groups: pretrained={len(pretrained_params)}, new={len(new_params)}")

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': cfg['lr_pretrained']},
        {'params': new_params,        'lr': cfg['lr_new']},
    ], weight_decay=cfg['weight_decay'])

    # OneCycleLR with discriminative max_lr
    steps_per_epoch_effective = len(train_loader) // cfg['gradient_accumulation']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg['lr_pretrained'], cfg['lr_new']],
        steps_per_epoch=steps_per_epoch_effective,
        epochs=cfg['epochs'],
        pct_start=0.3,          # 30% warmup (gentler for fine-tuning)
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
                'phase3_config':    cfg,
            }, save_name)
            print(f"  -> Saved {save_name}  "
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
    print(f" Phase 3 Training Completed")
    print(f"{'='*80}")
    print(f"   Best Accuracy:       {best_acc:.2f}%")
    print(f"   Best CER:            {best_cer:.4f}")
    print(f"   Best Confidence Gap: {best_gap:.4f}")
    print(f"   Best Epoch:          {best_epoch+1}")
    print(f"   Model:               {save_name}")
    print(f"   TensorBoard:         {log_dir}")

    if os.path.exists(cfg['phase1_checkpoint']):
        c = torch.load(cfg['phase1_checkpoint'], map_location='cpu',
                       weights_only=False)
        p1 = c.get('accuracy', 0)
        print(f"\n   Phase 1 → Phase 3: {p1:.2f}% → {best_acc:.2f}% "
              f"({best_acc - p1:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
