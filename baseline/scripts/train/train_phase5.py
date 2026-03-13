"""
Phase 5 Training — CTC + Attention Dual Decoder

Architecture: Phase5Recognizer
  Encoder (from Phase 4): STN → ResNet-34 → SpatialTemporalFusion → BiLSTM
  CTC Head:               FC → LogSoftmax → CTC Loss
  Attention Decoder (NEW): GRU + Bahdanau Attention → CE Loss
  SR Branch:              AuxSRBranch → SR Loss (training only)

Training strategy:
  - Load encoder weights from Phase 4 (78.08%)
  - Discriminative LR: encoder 1e-4, new attention decoder 1e-3
  - Dual loss: λ_ctc * CTC + λ_attn * Attention_CE + λ_sr * SR
  - Teacher forcing with linear decay (1.0 → 0.3)
  - OneCycleLR (proven best from Phase 4)
  - 100 epochs, patience=25

Saves: best_model_phase5.pth
"""

import os
import sys
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
from models.recognizer_v4 import Phase5Recognizer
from models.sr_branch import SRLoss

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════
IMG_HEIGHT = 64
IMG_WIDTH  = 224

MAX_LABEL_LEN = 10  # Max plate length + margin (Brazilian=7, padded to 10)

CFG = {
    'epochs':                100,
    'batch_size':            24,       # Lower — shares GPU with Phase 4
    'gradient_accumulation': 4,        # eff batch = 96

    # Discriminative LR
    'lr_encoder':            1e-4,     # Lower for pretrained encoder
    'lr_decoder':            1e-3,     # Higher for new attention decoder
    'weight_decay':          1e-4,
    'max_grad_norm':         5.0,

    # OneCycleLR (proven from Phase 4)
    'pct_start':             0.1,
    'div_factor':            10,
    'final_div':             100,

    # Architecture
    'hidden_size':           256,
    'num_lstm_layers':       2,
    'dropout':               0.25,
    'fusion_reduction':      16,

    # Attention decoder
    'attention_dim':         256,
    'decoder_dim':           256,
    'embed_dim':             64,
    'max_decode_len':        MAX_LABEL_LEN,
    'attn_dropout':          0.2,

    # Loss weights
    'lambda_ctc':            1.0,
    'lambda_attn':           0.5,      # Start lower — let CTC stabilize
    'lambda_sr':             0.1,
    'lambda_perceptual':     0.1,
    'focal_ctc_gamma':       2.0,

    # Teacher forcing schedule
    'tf_start':              1.0,      # 100% teacher forcing initially
    'tf_end':                0.3,      # Decay to 30%
    'tf_decay_epochs':       60,       # Linear decay over 60 epochs

    # SR branch
    'use_sr_branch':         True,
    'sr_target_h':           43,
    'sr_target_w':           120,

    # Training
    'patience':              25,
    'label_smoothing':       0.1,      # For attention CE loss

    'num_workers':           4,
    'load_from':             'checkpoints/best_model_phase4.pth',
    'model_save_name':       'checkpoints/best_model_phase5_attn.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Augmentation (same as Phase 4)
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
# Dataset — returns padded attention targets
# ════════════════════════════════════════════════════════════════════════════
class Phase5Dataset(Dataset):
    def __init__(self, root_dir, mode='train', split_ratio=0.8,
                 sr_target_h=43, sr_target_w=120, max_label_len=MAX_LABEL_LEN):
        self.mode = mode
        self.sr_h = sr_target_h
        self.sr_w = sr_target_w
        self.max_label_len = max_label_len
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

        # CTC target (concatenated, no padding)
        target = [Config.CHAR2IDX[c] for c in label if c in Config.CHAR2IDX]
        if len(target) == 0:
            target = [0]

        # Attention target (padded to max_label_len with 0)
        attn_target = target[:self.max_label_len]
        attn_target = attn_target + [0] * (self.max_label_len - len(attn_target))
        attn_target = torch.tensor(attn_target, dtype=torch.long)

        hr_tensor = self._load_hr(item.get('hr_paths', []))

        return (images_tensor, torch.tensor(target, dtype=torch.long),
                len(target), label, hr_tensor, attn_target)

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
        images, targets, target_lengths, labels_text, hr_images, attn_targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        hr_images = torch.stack(hr_images)
        attn_targets = torch.stack(attn_targets)
        return images, targets, target_lengths, list(labels_text), hr_images, attn_targets


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
# Attention CE Loss (with mask for padded positions)
# ════════════════════════════════════════════════════════════════════════════
class AttentionCELoss(nn.Module):
    """Cross-entropy loss for attention decoder with padding mask."""
    def __init__(self, num_classes, label_smoothing=0.1, pad_idx=0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction='mean',
        )

    def forward(self, logits, targets, target_lengths):
        """
        Args:
            logits:         [B, max_len, num_classes]
            targets:        [B, max_len] (padded with 0)
            target_lengths: [B] actual lengths
        Returns:
            loss: scalar
        """
        B, T, C = logits.size()

        # Create mask: only compute loss on valid (non-padded) positions
        mask = torch.zeros(B, T, dtype=torch.bool, device=logits.device)
        for i, l in enumerate(target_lengths):
            mask[i, :l] = True

        # Flatten and apply mask
        logits_flat = logits[mask]      # [N_valid, C]
        targets_flat = targets[mask]    # [N_valid]

        if logits_flat.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return self.ce(logits_flat, targets_flat)


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
def get_teacher_forcing_ratio(epoch, cfg):
    """Linear decay from tf_start to tf_end over tf_decay_epochs."""
    if epoch >= cfg['tf_decay_epochs']:
        return cfg['tf_end']
    ratio = cfg['tf_start'] - (cfg['tf_start'] - cfg['tf_end']) * (epoch / cfg['tf_decay_epochs'])
    return ratio


def train_one_epoch(model, loader, ctc_criterion, attn_criterion, sr_criterion,
                    optimizer, scheduler, scaler, device, epoch, cfg):
    model.train()
    total_ctc, total_attn, total_sr, total_loss_sum = 0.0, 0.0, 0.0, 0.0
    n = 0
    ga = cfg['gradient_accumulation']
    use_sr = cfg['use_sr_branch'] and model.sr_branch is not None
    tf_ratio = get_teacher_forcing_ratio(epoch, cfg)

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
    for bi, batch in enumerate(pbar):
        images, targets, tgt_lens, _, hr_images, attn_targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        hr_images = hr_images.to(device, non_blocking=True)
        attn_targets = attn_targets.to(device, non_blocking=True)

        with autocast('cuda'):
            # Forward with both heads
            result = model(images, targets=attn_targets, return_sr=use_sr,
                          return_attention=True, teacher_forcing_ratio=tf_ratio)

            if use_sr:
                ctc_preds, attn_logits, sr_out = result
            else:
                ctc_preds, attn_logits = result
                sr_out = None

            # CTC Loss
            T_out = ctc_preds.size(1)
            input_lengths = torch.full((images.size(0),), T_out, dtype=torch.long)
            ctc_loss = ctc_criterion(ctc_preds.permute(1, 0, 2), targets,
                                     input_lengths, tgt_lens)

            # Attention Loss
            attn_loss = attn_criterion(attn_logits, attn_targets, tgt_lens)

            # SR Loss
            if use_sr and sr_out is not None:
                sr_loss_val, _ = sr_criterion(sr_out, hr_images)
            else:
                sr_loss_val = torch.tensor(0.0)

            # Combined loss
            loss = (cfg['lambda_ctc'] * ctc_loss +
                    cfg['lambda_attn'] * attn_loss +
                    cfg['lambda_sr'] * sr_loss_val) / ga

        scaler.scale(loss).backward()

        if (bi + 1) % ga == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Step OneCycleLR per optimizer step
            scheduler.step()

        total_ctc += ctc_loss.item()
        total_attn += attn_loss.item()
        total_sr += (sr_loss_val.item() if isinstance(sr_loss_val, torch.Tensor) else 0.0)
        total_loss_sum += loss.item() * ga
        n += 1

        pbar.set_postfix({
            'ctc':  f"{ctc_loss.item():.3f}",
            'attn': f"{attn_loss.item():.3f}",
            'sr':   f"{(sr_loss_val.item() if isinstance(sr_loss_val, torch.Tensor) else 0.0):.3f}",
            'tf':   f"{tf_ratio:.2f}",
            'lr':   f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    # Flush remaining
    if n % ga != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    return total_ctc / n, total_attn / n, total_sr / n, total_loss_sum / n


@torch.no_grad()
def validate(model, loader, ctc_criterion_val, device):
    """Validate using CTC greedy decode (same as Phase 4 for fair comparison)."""
    model.eval()
    val_loss = 0.0
    all_preds, all_targets, all_confs = [], [], []

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        tgt_lens = batch[2]
        labels_text = batch[3]

        # Only need CTC output for validation
        ctc_preds = model(images)
        T_out = ctc_preds.size(1)
        input_lengths = torch.full((images.size(0),), T_out, dtype=torch.long)
        loss = ctc_criterion_val(ctc_preds.permute(1, 0, 2), targets,
                                 input_lengths, tgt_lens)
        val_loss += loss.item()

        decoded = decode_predictions(torch.argmax(ctc_preds, dim=2), Config.IDX2CHAR)
        all_preds.extend(decoded)
        all_targets.extend(labels_text)
        all_confs.extend(get_prediction_confidence(ctc_preds).tolist())

    avg_loss = val_loss / len(loader)
    acc = calculate_accuracy(all_preds, all_targets) * 100
    cer = calculate_cer(all_preds, all_targets)
    is_c = [p == t for p, t in zip(all_preds, all_targets)]
    gap = calculate_confidence_gap(all_confs, is_c)
    return {'loss': avg_loss, 'accuracy': acc, 'cer': cer,
            'confidence_gap': gap, 'outputs/predictions/predictions': all_preds,
            'targets': all_targets}


@torch.no_grad()
def validate_attention(model, loader, device):
    """Additionally validate using the attention decoder (for monitoring)."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        labels_text = batch[3]

        # Get attention decoder output (teacher_forcing_ratio=0 for pure autoregressive)
        ctc_preds, attn_logits = model(images, targets=None,
                                        return_attention=True,
                                        teacher_forcing_ratio=0.0)

        # Decode attention output
        attn_decoded = attn_logits.argmax(dim=2)  # [B, max_len]
        for i in range(attn_decoded.size(0)):
            chars = []
            for j in range(attn_decoded.size(1)):
                idx = attn_decoded[i, j].item()
                if idx == 0:  # blank/padding → stop
                    break
                if idx in Config.IDX2CHAR:
                    chars.append(Config.IDX2CHAR[idx])
            all_preds.append(''.join(chars))
        all_targets.extend(labels_text)

    acc = calculate_accuracy(all_preds, all_targets) * 100
    return acc


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG
    seed_everything(Config.SEED)
    device = Config.DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase5_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 5 — CTC + Attention Dual Decoder")
    print("=" * 80)
    print(f"  Load encoder from: {cfg['load_from']}")
    print(f"  LR encoder: {cfg['lr_encoder']} | LR decoder: {cfg['lr_decoder']}")
    print(f"  Loss: {cfg['lambda_ctc']}×CTC + {cfg['lambda_attn']}×AttnCE + {cfg['lambda_sr']}×SR")
    print(f"  Teacher forcing: {cfg['tf_start']} → {cfg['tf_end']} over {cfg['tf_decay_epochs']} epochs")
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = Phase5Dataset(Config.DATA_ROOT, mode='train', split_ratio=0.8,
                             sr_target_h=cfg['sr_target_h'], sr_target_w=cfg['sr_target_w'],
                             max_label_len=cfg['max_decode_len'])
    val_ds = Phase5Dataset(Config.DATA_ROOT, mode='val', split_ratio=0.8,
                           sr_target_h=cfg['sr_target_h'], sr_target_w=cfg['sr_target_w'],
                           max_label_len=cfg['max_decode_len'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              collate_fn=Phase5Dataset.collate_fn,
                              num_workers=cfg['num_workers'], pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            collate_fn=Phase5Dataset.collate_fn,
                            num_workers=cfg['num_workers'], pin_memory=True,
                            persistent_workers=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────
    model = Phase5Recognizer(
        num_classes=Config.NUM_CLASSES,
        use_stn=Config.USE_STN,
        use_resnet_backbone=Config.USE_RESNET_BACKBONE,
        hidden_size=cfg['hidden_size'],
        num_lstm_layers=cfg['num_lstm_layers'],
        dropout=cfg['dropout'],
        fusion_reduction=cfg['fusion_reduction'],
        attention_dim=cfg['attention_dim'],
        decoder_dim=cfg['decoder_dim'],
        embed_dim=cfg['embed_dim'],
        max_decode_len=cfg['max_decode_len'],
        attn_dropout=cfg['attn_dropout'],
        use_sr_branch=cfg['use_sr_branch'],
        sr_target_h=cfg['sr_target_h'],
        sr_target_w=cfg['sr_target_w'],
    ).to(device)

    # Load Phase 4 encoder weights
    ckpt = model.load_phase4_weights(cfg['load_from'], device=device)
    phase4_acc = ckpt.get('accuracy', 0.0)

    info = model.get_model_info()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model info: {info}")
    print(f"  Total params: {total_params:,}")

    # Validate loaded encoder
    ctc_criterion_val = nn.CTCLoss(blank=0, zero_infinity=True)
    print("  Validating loaded encoder...")
    val0 = validate(model, val_loader, ctc_criterion_val, device)
    print(f"  Encoder baseline: Acc={val0['accuracy']:.2f}% | CER={val0['cer']:.4f}")

    # ── Loss ─────────────────────────────────────────────────────────────
    ctc_criterion = FocalCTCLoss(blank=0, gamma=cfg['focal_ctc_gamma'])
    attn_criterion = AttentionCELoss(
        num_classes=Config.NUM_CLASSES,
        label_smoothing=cfg['label_smoothing'],
        pad_idx=0,
    )
    sr_criterion = SRLoss(lambda_l1=1.0, lambda_perceptual=cfg['lambda_perceptual']).to(device)

    # ── Optimizer: Discriminative LR ─────────────────────────────────────
    # Separate pretrained encoder params from new attention decoder params
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'attention_decoder' in name:
            decoder_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': cfg['lr_encoder'], 'weight_decay': cfg['weight_decay']},
        {'params': decoder_params, 'lr': cfg['lr_decoder'], 'weight_decay': cfg['weight_decay']},
    ])

    # OneCycleLR with max_lr per group
    steps_per_epoch = len(train_loader) // cfg['gradient_accumulation']
    total_steps = cfg['epochs'] * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg['lr_encoder'], cfg['lr_decoder']],
        total_steps=total_steps,
        pct_start=cfg['pct_start'],
        div_factor=cfg['div_factor'],
        final_div_factor=cfg['final_div'],
        anneal_strategy='cos',
    )

    scaler = GradScaler()

    # ── Training ─────────────────────────────────────────────────────────
    patience_counter = 0
    best_acc = val0['accuracy']
    best_cer = val0['cer']
    best_gap = val0['confidence_gap']
    best_epoch = -1

    print(f"\n{'='*80}")
    print(f"  Starting from {best_acc:.2f}% (Phase 4 encoder)")
    print(f"  New: Attention Decoder (GRU+Bahdanau)")
    print(f"  Dual loss: CTC + Attention CE")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        tf_ratio = get_teacher_forcing_ratio(epoch, cfg)

        ctc_l, attn_l, sr_l, total_l = train_one_epoch(
            model, train_loader, ctc_criterion, attn_criterion, sr_criterion,
            optimizer, scheduler, scaler, device, epoch, cfg)

        val = validate(model, val_loader, ctc_criterion_val, device)

        # Also check attention decoder accuracy every 10 epochs
        attn_acc = 0.0
        if (epoch + 1) % 10 == 0:
            attn_acc = validate_attention(model, val_loader, device)

        # Logging
        writer.add_scalar('Loss/train_ctc', ctc_l, epoch)
        writer.add_scalar('Loss/train_attn', attn_l, epoch)
        writer.add_scalar('Loss/train_sr', sr_l, epoch)
        writer.add_scalar('Loss/val', val['loss'], epoch)
        writer.add_scalar('Accuracy/val_ctc', val['accuracy'], epoch)
        writer.add_scalar('CER/val', val['cer'], epoch)
        writer.add_scalar('TeacherForcing', tf_ratio, epoch)
        writer.add_scalar('LR/encoder', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/decoder', optimizer.param_groups[1]['lr'], epoch)
        if attn_acc > 0:
            writer.add_scalar('Accuracy/val_attn', attn_acc, epoch)

        attn_str = f" | AttnAcc: {attn_acc:.2f}%" if attn_acc > 0 else ""
        print(f"Epoch {epoch+1}/{cfg['epochs']} [TF={tf_ratio:.2f}]:")
        print(f"  Train CTC: {ctc_l:.4f} | Attn: {attn_l:.4f} | SR: {sr_l:.4f} | Total: {total_l:.4f}")
        print(f"  Val   Loss: {val['loss']:.4f} | CTC Acc: {val['accuracy']:.2f}% | "
              f"CER: {val['cer']:.4f} | Gap: {val['confidence_gap']:.4f}{attn_str}")
        print(f"  LR enc: {optimizer.param_groups[0]['lr']:.2e} | "
              f"LR dec: {optimizer.param_groups[1]['lr']:.2e}")

        # Save best (based on CTC accuracy for fair comparison)
        if val['accuracy'] > best_acc:
            improvement = val['accuracy'] - best_acc
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
                'phase5_config': cfg,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
                'loaded_from': cfg['load_from'],
                'phase4_accuracy': phase4_acc,
            }, cfg['model_save_name'])
            print(f"  -> Saved {cfg['model_save_name']} (Acc {val['accuracy']:.2f}%, +{improvement:.2f}%)")
        else:
            patience_counter += 1

        # Samples
        if (epoch + 1) % 10 == 0:
            print(f"\n  Samples (epoch {epoch+1}):")
            for i in range(min(5, len(val['outputs/predictions/predictions']))):
                p, t = val['outputs/predictions/predictions'][i], val['targets'][i]
                m = "+" if p == t else "x"
                print(f"    [{m}] GT: {t:8s}  Pred: {p:8s}")
            print()

        # Early stopping
        if patience_counter >= cfg['patience']:
            print(f"\n  Early stopping at epoch {epoch+1} (best: ep {best_epoch+1}, {best_acc:.2f}%)")
            break

    writer.close()
    print(f"\n{'='*80}")
    print("  Phase 5 Completed — CTC + Attention Dual Decoder")
    print(f"{'='*80}")
    print(f"  Phase 4 baseline:    {phase4_acc:.2f}%")
    print(f"  Best CTC Accuracy:   {best_acc:.2f}%")
    print(f"  Improvement:         {best_acc - phase4_acc:+.2f}%")
    print(f"  Best CER:            {best_cer:.4f}")
    print(f"  Best Epoch:          {best_epoch+1}")
    print(f"  Model:               {cfg['model_save_name']}")
    print(f"  TensorBoard:         {log_dir}")
    sys.exit(0)


if __name__ == '__main__':
    main()
