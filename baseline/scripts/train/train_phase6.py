"""
Phase 6 — Format-Constrained Decoding + Scenario-Balanced Sampling

Key improvements over Phase 4 v2 (78.47%):
  1. Scenario-Balanced Sampling: WeightedRandomSampler oversamples B_Brazilian (52.12% → target 70%+)
  2. Format-Constrained Validation: mask invalid chars at each position (letter↔digit confusion fix)
  3. Plate-Layout-Aware Training: store plate_layout per sample, use for format constraint
  4. Position-Type Auxiliary Loss: small CE loss enforcing letter-vs-digit at correct positions

Saves to: best_model_phase6.pth (preserves all previous checkpoints)
"""

import os
import sys
import glob
import json
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
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
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
    'epochs':                100,
    'batch_size':            24,
    'gradient_accumulation': 4,         # eff batch = 96
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

    # From Phase 4 v2 checkpoint (fine-tune)
    'from_scratch':          False,
    'pretrained_path':       'checkpoints/best_model_phase4_v2.pth',

    # Early stopping
    'patience':              25,

    # Regularization
    'label_smoothing':       0.05,

    # Focal CTC
    'focal_ctc_gamma':       2.0,

    # MixUp
    'mixup_alpha':           0.2,
    'mixup_prob':            0.3,

    # Balanced sampling — oversample B_Brazilian 3×
    'balance_oversampling':  True,
    'b_brazilian_weight':    3.0,
    'b_mercosur_weight':     1.0,
    'a_brazilian_weight':    1.5,     # slight boost — 77.15% acc
    'a_mercosur_weight':     1.0,    # 84.25% — already good

    # Format-constrained decoding at validation
    'use_format_constraint': True,

    # Position-type auxiliary loss weight
    'lambda_pos_type':       0.1,

    # Workers
    'num_workers':           4,

    # Output — separate from all previous!
    'model_save_name':       'checkpoints/best_model_phase6.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# Format constraint tables
# ════════════════════════════════════════════════════════════════════════════
# idx 0 = CTC blank, 1-10 = digits 0-9, 11-36 = A-Z
DIGIT_INDICES = list(range(1, 11))     # chars 0-9
LETTER_INDICES = list(range(11, 37))   # chars A-Z

# Brazilian: LLL DDDD  (positions 0-2 = letter, 3-6 = digit)
# Mercosur: LLL D L DD (positions 0-2 = letter, 3 = digit, 4 = letter, 5-6 = digit)
BRAZILIAN_POS_TYPE = ['letter', 'letter', 'letter', 'digit', 'digit', 'digit', 'digit']
MERCOSUR_POS_TYPE  = ['letter', 'letter', 'letter', 'digit', 'letter', 'digit', 'digit']


def get_format_mask(plate_layout, num_classes=37):
    """
    Return a [7, num_classes] mask: 1.0 for allowed chars, 0.0 for forbidden.
    Position 0 always allows blank (CTC needs it).
    """
    if plate_layout == 'Brazilian':
        pos_types = BRAZILIAN_POS_TYPE
    elif plate_layout == 'Mercosur':
        pos_types = MERCOSUR_POS_TYPE
    else:
        # Unknown — allow everything
        return torch.ones(7, num_classes)

    mask = torch.zeros(7, num_classes)
    for pos, ptype in enumerate(pos_types):
        mask[pos, 0] = 1.0  # always allow CTC blank
        if ptype == 'letter':
            for idx in LETTER_INDICES:
                mask[pos, idx] = 1.0
        else:
            for idx in DIGIT_INDICES:
                mask[pos, idx] = 1.0
    return mask


# Pre-compute masks
BRAZILIAN_MASK = get_format_mask('Brazilian')
MERCOSUR_MASK = get_format_mask('Mercosur')
UNKNOWN_MASK = torch.ones(7, 37)


def get_pos_type_targets(plate_layout):
    """Return [7] tensor: 0=letter, 1=digit for each position."""
    if plate_layout == 'Brazilian':
        return torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    elif plate_layout == 'Mercosur':
        return torch.tensor([0, 0, 0, 1, 0, 1, 1], dtype=torch.long)
    else:
        return torch.tensor([-1, -1, -1, -1, -1, -1, -1], dtype=torch.long)


# ════════════════════════════════════════════════════════════════════════════
# Augmentation (same as Phase 4 v2)
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
# Dataset — now includes plate_layout + scenario info
# ════════════════════════════════════════════════════════════════════════════
class Phase6Dataset(Dataset):
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

        self.samples = []
        self._load_data(root_dir, mode, split_ratio)

    def _load_data(self, root_dir, mode, split_ratio):
        """Load data with plate_layout and scenario info."""
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))

        if not all_tracks:
            print("❌ No data found.")
            return

        # Load split
        val_split_file = os.path.join(os.path.dirname(abs_root), 'val_tracks.json')
        if not os.path.exists(val_split_file):
            val_split_file = Config.VAL_SPLIT_FILE

        train_tracks, val_tracks = [], []
        if os.path.exists(val_split_file):
            print(f"📂 Loading split from '{val_split_file}'...")
            with open(val_split_file, 'r') as f:
                val_ids = set(json.load(f))
            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
        else:
            rng = random.Random(Config.SEED)
            rng.shuffle(all_tracks)
            split_idx = int(len(all_tracks) * split_ratio)
            train_tracks = all_tracks[:split_idx]
            val_tracks = all_tracks[split_idx:]

        selected = train_tracks if mode == 'train' else val_tracks
        print(f"[{mode.upper()}] Loading {len(selected)} tracks...")

        for track_path in tqdm(selected, desc=f"Indexing {mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]

                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue

                plate_layout = data.get('plate_layout', 'unknown')

                # Detect scenario from path
                if 'Scenario-A' in track_path:
                    scenario = 'A_Brazilian' if 'Brazilian' in track_path else 'A_Mercosur'
                elif 'Scenario-B' in track_path:
                    scenario = 'B_Brazilian' if 'Brazilian' in track_path else 'B_Mercosur'
                else:
                    scenario = 'unknown'

                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )

                if len(lr_files) > 0:
                    self.samples.append({
                        'lr_paths': lr_files,
                        'hr_paths': hr_files,
                        'label': label,
                        'plate_layout': plate_layout,
                        'scenario': scenario,
                    })
            except Exception:
                pass

        # Print scenario distribution
        from collections import Counter
        sc_counts = Counter(s['scenario'] for s in self.samples)
        print(f"  [{mode.upper()}] {len(self.samples)} samples: {dict(sc_counts)}")

    def get_sample_weights(self, cfg):
        """Compute per-sample weights for WeightedRandomSampler."""
        weight_map = {
            'A_Brazilian': cfg['a_brazilian_weight'],
            'A_Mercosur':  cfg['a_mercosur_weight'],
            'B_Brazilian': cfg['b_brazilian_weight'],
            'B_Mercosur':  cfg['b_mercosur_weight'],
            'unknown':     1.0,
        }
        weights = [weight_map.get(s['scenario'], 1.0) for s in self.samples]
        return weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']
        plate_layout = item['plate_layout']
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

        # Layout index: 0=Brazilian, 1=Mercosur, 2=unknown
        layout_idx = 0 if plate_layout == 'Brazilian' else (1 if plate_layout == 'Mercosur' else 2)

        return (images_tensor, torch.tensor(target, dtype=torch.long),
                len(target), label, hr_tensor, layout_idx)

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
        images, targets, target_lengths, labels_text, hr_images, layout_indices = zip(*batch)
        images = torch.stack(images)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        hr_images = torch.stack(hr_images)
        layout_indices = torch.tensor(layout_indices, dtype=torch.long)
        return images, targets, target_lengths, list(labels_text), hr_images, layout_indices


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
# Position-Type Auxiliary Loss
# ════════════════════════════════════════════════════════════════════════════
class PositionTypeLoss(nn.Module):
    """
    Auxiliary loss that enforces letter/digit constraints at each position.
    Takes CTC logits [B, T, C], computes per-position probability of
    letter-vs-digit, and applies CE loss.
    """
    def __init__(self):
        super().__init__()
        # digit indices in vocab: 1-10 (0-9), letter indices: 11-36 (A-Z)
        self.register_buffer('digit_mask',
            torch.zeros(37).scatter_(0, torch.tensor(DIGIT_INDICES), 1.0))
        self.register_buffer('letter_mask',
            torch.zeros(37).scatter_(0, torch.tensor(LETTER_INDICES), 1.0))

    def forward(self, logits, layout_indices):
        """
        logits: [B, T, C] — raw log-probs from model
        layout_indices: [B] — 0=Brazilian, 1=Mercosur, 2=unknown

        Returns: scalar loss (averaged over valid positions and batch items)
        """
        B, T, C = logits.shape
        device = logits.device

        # Softmax over char dimension
        probs = torch.exp(logits)  # log_softmax → softmax

        # Sum probs for digit and letter groups
        p_digit = (probs * self.digit_mask.unsqueeze(0).unsqueeze(0)).sum(dim=2)   # [B, T]
        p_letter = (probs * self.letter_mask.unsqueeze(0).unsqueeze(0)).sum(dim=2) # [B, T]

        # Stack: [B, T, 2] where dim2: 0=letter, 1=digit
        p_type = torch.stack([p_letter, p_digit], dim=2)  # [B, T, 2]
        log_p_type = torch.log(p_type + 1e-8)

        # For CTC output T timesteps, we need to map to 7 plate positions.
        # Use evenly-spaced sampling: pick T/7 positions
        plate_len = 7
        if T < plate_len:
            return torch.tensor(0.0, device=device)

        # Sample positions from CTC timeline (evenly spaced)
        pos_indices = torch.linspace(0, T - 1, plate_len).long().to(device)

        # Get log-probs at sampled positions: [B, 7, 2]
        log_p_sampled = log_p_type[:, pos_indices, :]

        # Build targets: [B, 7] — 0=letter, 1=digit, -1=ignore
        targets = torch.full((B, plate_len), -1, dtype=torch.long, device=device)
        for i in range(B):
            layout = layout_indices[i].item()
            if layout == 0:  # Brazilian
                targets[i] = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)
            elif layout == 1:  # Mercosur
                targets[i] = torch.tensor([0, 0, 0, 1, 0, 1, 1], device=device)
            # else: unknown → -1 (ignored)

        # Reshape for CE
        log_p_flat = log_p_sampled.reshape(-1, 2)  # [B*7, 2]
        targets_flat = targets.reshape(-1)          # [B*7]

        # Only compute loss for valid positions (not -1)
        valid_mask = targets_flat >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = F.nll_loss(log_p_flat[valid_mask], targets_flat[valid_mask])
        return loss


# ════════════════════════════════════════════════════════════════════════════
# Format-Constrained Decoding
# ════════════════════════════════════════════════════════════════════════════
def format_constrained_decode(logits, layout_indices, idx2char):
    """
    Post-hoc format-constrained decoding.

    1. Standard CTC greedy decode to get character string
    2. Record which CTC timestep produced each output character
    3. For positions violating format, replace with best valid char from that timestep

    logits: [B, T, C] log-probs
    layout_indices: [B] — 0=Brazilian, 1=Mercosur, 2=unknown
    idx2char: dict

    Returns: list of decoded strings
    """
    B, T, C = logits.shape

    format_rules = {
        0: BRAZILIAN_POS_TYPE,   # ['letter','letter','letter','digit','digit','digit','digit']
        1: MERCOSUR_POS_TYPE,    # ['letter','letter','letter','digit','letter','digit','digit']
    }
    digit_set = set(DIGIT_INDICES)
    letter_set = set(LETTER_INDICES)

    results = []
    for b in range(B):
        layout = layout_indices[b].item()
        lp = logits[b]  # [T, C]
        pred_indices = torch.argmax(lp, dim=1).cpu().tolist()  # [T]

        # CTC collapse: record (char_idx, best_timestep) for each output character
        chars = []
        char_timesteps = []
        prev = -1
        for t, idx in enumerate(pred_indices):
            if idx != 0 and idx != prev:
                chars.append(idx)
                char_timesteps.append(t)
            prev = idx

        # If layout unknown or length != 7, return as-is
        pos_types = format_rules.get(layout, None)
        if pos_types is None or len(chars) != 7:
            results.append(''.join(idx2char.get(c, '') for c in chars))
            continue

        # Apply format constraints per position
        corrected = []
        for pos in range(7):
            char_idx = chars[pos]
            t = char_timesteps[pos]
            expected_type = pos_types[pos]

            is_digit = char_idx in digit_set
            is_letter = char_idx in letter_set

            if expected_type == 'digit' and is_digit:
                corrected.append(char_idx)
            elif expected_type == 'letter' and is_letter:
                corrected.append(char_idx)
            else:
                # Violation: pick best valid char from this timestep
                valid_indices = DIGIT_INDICES if expected_type == 'digit' else LETTER_INDICES
                valid_tensor = torch.tensor(valid_indices, device=lp.device)
                valid_logits = lp[t, valid_tensor]
                best_valid = valid_indices[valid_logits.argmax().item()]
                corrected.append(best_valid)

        results.append(''.join(idx2char.get(c, '') for c in corrected))

    return results


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, ctc_criterion, sr_criterion, pos_type_loss,
                    optimizer, scaler, scheduler, device, epoch, cfg):
    model.train()
    total_ctc, total_sr, total_pos, total_loss_sum = 0.0, 0.0, 0.0, 0.0
    n = 0
    ga = cfg['gradient_accumulation']
    use_sr = cfg['use_sr_branch'] and model.sr_branch is not None
    label_smooth = cfg.get('label_smoothing', 0.0)
    lambda_pos = cfg.get('lambda_pos_type', 0.1)

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
    for bi, batch in enumerate(pbar):
        images, targets, tgt_lens, _, hr_images, layout_indices = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        hr_images = hr_images.to(device, non_blocking=True)
        layout_indices = layout_indices.to(device, non_blocking=True)

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

            # Position-type auxiliary loss
            pos_loss = pos_type_loss(preds, layout_indices)

            if use_sr and sr_out is not None:
                sr_loss_val, _ = sr_criterion(sr_out, hr_images)
                loss = (ctc_loss + cfg['lambda_sr'] * sr_loss_val
                        + lambda_pos * pos_loss) / ga
            else:
                sr_loss_val = torch.tensor(0.0)
                loss = (ctc_loss + lambda_pos * pos_loss) / ga

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
        total_pos += pos_loss.item()
        total_loss_sum += loss.item() * ga
        n += 1

        pbar.set_postfix({
            'ctc': f"{ctc_loss.item():.4f}",
            'pos': f"{pos_loss.item():.4f}",
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

    return total_ctc / n, total_sr / n, total_pos / n, total_loss_sum / n


@torch.no_grad()
def validate(model, loader, ctc_criterion_val, device, use_format_constraint=True):
    model.eval()
    val_loss = 0.0
    all_preds_greedy, all_preds_constrained = [], []
    all_targets, all_confs = [], []

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        tgt_lens = batch[2]
        labels_text = batch[3]
        layout_indices = batch[5].to(device, non_blocking=True)

        preds = model(images)
        T_out = preds.size(1)
        input_lengths = torch.full((images.size(0),), T_out, dtype=torch.long)
        loss = ctc_criterion_val(preds.permute(1, 0, 2), targets,
                                 input_lengths, tgt_lens)
        val_loss += loss.item()

        # Greedy decode (standard)
        decoded_greedy = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)
        all_preds_greedy.extend(decoded_greedy)

        # Format-constrained decode
        if use_format_constraint:
            decoded_constrained = format_constrained_decode(
                preds, layout_indices, Config.IDX2CHAR)
            all_preds_constrained.extend(decoded_constrained)

        all_targets.extend(labels_text)
        all_confs.extend(get_prediction_confidence(preds).tolist())

    avg_loss = val_loss / len(loader)
    acc_greedy = calculate_accuracy(all_preds_greedy, all_targets) * 100
    cer_greedy = calculate_cer(all_preds_greedy, all_targets)

    result = {
        'loss': avg_loss,
        'accuracy_greedy': acc_greedy,
        'cer_greedy': cer_greedy,
        'outputs/predictions/predictions_greedy': all_preds_greedy,
        'targets': all_targets,
    }

    if use_format_constraint and all_preds_constrained:
        acc_constrained = calculate_accuracy(all_preds_constrained, all_targets) * 100
        cer_constrained = calculate_cer(all_preds_constrained, all_targets)
        result['accuracy_constrained'] = acc_constrained
        result['cer_constrained'] = cer_constrained
        result['outputs/predictions/predictions_constrained'] = all_preds_constrained

    # Use best accuracy for model selection
    result['accuracy'] = max(acc_greedy,
                             result.get('accuracy_constrained', 0))
    result['cer'] = min(cer_greedy,
                        result.get('cer_constrained', float('inf')))
    result['outputs/predictions/predictions'] = (all_preds_constrained
                             if result['accuracy'] == result.get('accuracy_constrained', 0)
                             else all_preds_greedy)

    is_c = [p == t for p, t in zip(result['outputs/predictions/predictions'], all_targets)]
    result['confidence_gap'] = calculate_confidence_gap(all_confs, is_c)

    return result


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG
    seed_everything(Config.SEED)
    device = Config.DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase6_{ts}')
    writer = SummaryWriter(log_dir)

    print("=" * 80)
    print("  PHASE 6 — Format Constraint + Balanced Sampling")
    print("=" * 80)
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = Phase6Dataset(Config.DATA_ROOT, mode='train', split_ratio=0.8,
                             sr_target_h=cfg['sr_target_h'],
                             sr_target_w=cfg['sr_target_w'])
    val_ds = Phase6Dataset(Config.DATA_ROOT, mode='val', split_ratio=0.8,
                           sr_target_h=cfg['sr_target_h'],
                           sr_target_w=cfg['sr_target_w'])

    # ── Balanced sampler ─────────────────────────────────────────────────
    if cfg['balance_oversampling']:
        sample_weights = train_ds.get_sample_weights(cfg)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                                  sampler=sampler,
                                  collate_fn=Phase6Dataset.collate_fn,
                                  num_workers=cfg['num_workers'],
                                  pin_memory=True,
                                  persistent_workers=True, drop_last=True)
        print(f"  Balanced sampling enabled: B_Br×{cfg['b_brazilian_weight']}, "
              f"A_Br×{cfg['a_brazilian_weight']}, A_Mc×{cfg['a_mercosur_weight']}, "
              f"B_Mc×{cfg['b_mercosur_weight']}")
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                                  shuffle=True,
                                  collate_fn=Phase6Dataset.collate_fn,
                                  num_workers=cfg['num_workers'],
                                  pin_memory=True,
                                  persistent_workers=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, collate_fn=Phase6Dataset.collate_fn,
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

    # Load Phase 4 v2 checkpoint
    if not cfg['from_scratch'] and os.path.exists(cfg['pretrained_path']):
        ckpt = torch.load(cfg['pretrained_path'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        prev_acc = ckpt.get('accuracy', 0)
        prev_epoch = ckpt.get('epoch', 0) + 1
        print(f"  Loaded {cfg['pretrained_path']}: acc={prev_acc:.2f}%, epoch={prev_epoch}")
    else:
        print("  Training from scratch (ImageNet backbone only)")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_p:,}")

    # ── Loss ─────────────────────────────────────────────────────────────
    ctc_criterion = FocalCTCLoss(blank=0, gamma=cfg['focal_ctc_gamma'])
    ctc_criterion_val = nn.CTCLoss(blank=0, zero_infinity=True)
    sr_criterion = SRLoss(lambda_l1=1.0,
                          lambda_perceptual=cfg['lambda_perceptual']).to(device)
    pos_type_loss = PositionTypeLoss().to(device)

    # ── Optimizer — lower LR for fine-tuning ─────────────────────────────
    if not cfg['from_scratch']:
        lr = cfg['learning_rate'] * 0.1   # 1e-4 for fine-tuning
    else:
        lr = cfg['learning_rate']

    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg['weight_decay'])

    spe = len(train_loader) // cfg['gradient_accumulation']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=spe,
        epochs=cfg['epochs'],
        pct_start=0.1,       # shorter warmup for fine-tuning
        anneal_strategy='cos',
    )
    scaler = GradScaler()

    # ── Training loop ────────────────────────────────────────────────────
    patience_counter = 0
    best_acc = 0.0
    best_cer = float('inf')
    best_gap = 0.0
    best_epoch = -1
    best_method = 'greedy'

    print(f"\n{'='*80}")
    print("  Starting Phase 6 training...")
    print(f"  100 epochs | patience=25 | saves to {cfg['model_save_name']}")
    print(f"{'='*80}\n")

    for epoch in range(cfg['epochs']):
        ctc_l, sr_l, pos_l, total_l = train_one_epoch(
            model, train_loader, ctc_criterion, sr_criterion, pos_type_loss,
            optimizer, scaler, scheduler, device, epoch, cfg)

        val = validate(model, val_loader, ctc_criterion_val, device,
                       use_format_constraint=cfg['use_format_constraint'])

        # Logging
        writer.add_scalar('Loss/train_ctc', ctc_l, epoch)
        writer.add_scalar('Loss/train_sr', sr_l, epoch)
        writer.add_scalar('Loss/train_pos_type', pos_l, epoch)
        writer.add_scalar('Loss/train_total', total_l, epoch)
        writer.add_scalar('Loss/val', val['loss'], epoch)
        writer.add_scalar('Accuracy/val_greedy', val['accuracy_greedy'], epoch)
        writer.add_scalar('CER/val_greedy', val['cer_greedy'], epoch)
        if 'accuracy_constrained' in val:
            writer.add_scalar('Accuracy/val_constrained', val['accuracy_constrained'], epoch)
            writer.add_scalar('CER/val_constrained', val['cer_constrained'], epoch)
        writer.add_scalar('Accuracy/val_best', val['accuracy'], epoch)
        writer.add_scalar('ConfGap/val', val['confidence_gap'], epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Print
        acc_str = f"Greedy: {val['accuracy_greedy']:.2f}%"
        if 'accuracy_constrained' in val:
            acc_str += f" | Constrained: {val['accuracy_constrained']:.2f}%"
        print(f"Epoch {epoch+1}/{cfg['epochs']}:")
        print(f"  Train CTC: {ctc_l:.4f} | SR: {sr_l:.4f} | PosType: {pos_l:.4f}")
        print(f"  Val   Loss: {val['loss']:.4f} | {acc_str}")
        print(f"  CER: {val['cer']:.4f} | Gap: {val['confidence_gap']:.4f}")

        # Determine which method is better
        curr_method = 'greedy'
        if val.get('accuracy_constrained', 0) > val['accuracy_greedy']:
            curr_method = 'constrained'

        # Save best
        if val['accuracy'] > best_acc:
            best_acc = val['accuracy']
            best_cer = val['cer']
            best_gap = val['confidence_gap']
            best_epoch = epoch
            best_method = curr_method
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val['accuracy'],
                'accuracy_greedy': val['accuracy_greedy'],
                'accuracy_constrained': val.get('accuracy_constrained', 0),
                'cer': val['cer'],
                'confidence_gap': val['confidence_gap'],
                'config': model.get_model_info(),
                'phase6_config': cfg,
                'best_method': best_method,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
                'continued_from': cfg['pretrained_path'],
            }, cfg['model_save_name'])
            print(f"  -> Saved {cfg['model_save_name']}  "
                  f"(Acc {val['accuracy']:.2f}% [{best_method}], CER {val['cer']:.4f})")
        else:
            patience_counter += 1

        # Samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n  Sample predictions (epoch {epoch+1}):")
            preds_to_show = val['outputs/predictions/predictions']
            targets_to_show = val['targets']
            for i in range(min(5, len(preds_to_show))):
                p, t = preds_to_show[i], targets_to_show[i]
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
    print("  Phase 6 Training Completed")
    print(f"{'='*80}")
    print(f"  Best Accuracy:       {best_acc:.2f}% [{best_method}]")
    print(f"  Best CER:            {best_cer:.4f}")
    print(f"  Best Confidence Gap: {best_gap:.4f}")
    print(f"  Best Epoch:          {best_epoch+1}/{cfg['epochs']}")
    print(f"  Model:               {cfg['model_save_name']}")
    print(f"  Based on:            {cfg['pretrained_path']}")
    print(f"  TensorBoard:         {log_dir}")

    # Compare with Phase 4 v2
    p4v2_path = cfg['pretrained_path']
    if os.path.exists(p4v2_path):
        c = torch.load(p4v2_path, map_location='cpu', weights_only=False)
        p4v2_acc = c.get('accuracy', 0)
        print(f"\n  Phase 4 v2 -> Phase 6: {p4v2_acc:.2f}% -> {best_acc:.2f}% "
              f"({best_acc - p4v2_acc:+.2f}%)")

    sys.exit(0)


if __name__ == '__main__':
    main()
