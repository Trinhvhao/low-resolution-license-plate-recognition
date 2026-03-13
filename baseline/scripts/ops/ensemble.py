"""
Ensemble Inference — Confidence-weighted voting of Phase 1/2/3 models.

Combines three models with different architectures for prediction diversity:
  - Phase 1: BiLSTM + AttentionFusion          (76.88%)
  - Phase 2: Transformer + SpatialTemporalFusion (76.25%)
  - Phase 3: BiLSTM + SpatialTemporalFusion     (76.55%)

Union accuracy on val = 80.2% → ensemble should capture most of this.

Methods supported:
  1. soft_vote  — average log-probs → beam search (default)
  2. hard_vote  — majority vote on decoded texts (confidence-weighted)

Usage:
    python ensemble.py                          # eval on val set
    python ensemble.py --mode test              # generate submission
    python ensemble.py --mode test --output predictions_ensemble.txt
"""

import os
import sys
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import AdvancedMultiFrameDataset
from models import Phase1Recognizer, Phase2Recognizer, Phase3Recognizer
from utils import decode_predictions, calculate_accuracy, calculate_cer
from transforms import get_val_transforms
from postprocess import beam_search_decode, validate_plate_format


# ── Model definitions ───────────────────────────────────────────────────────
MODEL_DEFS = [
    {
        'name': 'Phase1',
        'path': 'checkpoints/best_model_phase1.pth',
        'class': Phase1Recognizer,
        'kwargs': {
            'num_classes': Config.NUM_CLASSES,
            'use_stn': True,
            'use_resnet_backbone': True,
        },
    },
    {
        'name': 'Phase2Fixed',
        'path': 'checkpoints/best_model_phase2_fixed.pth',
        'class': Phase2Recognizer,
        'kwargs': {
            'num_classes': Config.NUM_CLASSES,
            'use_stn': True,
            'use_resnet_backbone': True,
            'd_model': 512,
            'nhead': 4,
            'num_transformer_layers': 2,
            'dim_feedforward': 1024,
            'fusion_reduction': 16,
            'dropout': 0.1,
            'use_sr_branch': False,  # no SR at inference
        },
    },
    {
        'name': 'Phase3',
        'path': 'checkpoints/best_model_phase3.pth',
        'class': Phase3Recognizer,
        'kwargs': {
            'num_classes': Config.NUM_CLASSES,
            'use_stn': True,
            'use_resnet_backbone': True,
        },
    },
    {
        'name': 'Phase3Scratch',
        'path': 'checkpoints/best_model_phase3_scratch.pth',
        'class': Phase3Recognizer,
        'kwargs': {
            'num_classes': Config.NUM_CLASSES,
            'use_stn': True,
            'use_resnet_backbone': True,
        },
    },
]


# ── Load models ─────────────────────────────────────────────────────────────
def load_models(device, model_defs=MODEL_DEFS):
    """Load all models and return list of (name, model, accuracy)."""
    models = []
    for md in model_defs:
        if not os.path.exists(md['path']):
            print(f"  ⚠️  {md['name']}: {md['path']} not found, skipping")
            continue
        ckpt = torch.load(md['path'], map_location=device, weights_only=False)
        model = md['class'](**md['kwargs']).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        acc = ckpt.get('accuracy', 0)
        print(f"  ✅ {md['name']}: acc={acc:.2f}%, epoch={ckpt.get('epoch',0)+1}")
        models.append((md['name'], model, acc))
    return models


# ── Soft voting (log-prob averaging) ────────────────────────────────────────
@torch.no_grad()
def ensemble_soft_vote(models, images, idx2char, beam_width=10):
    """
    Average log-probabilities from all models, then beam-search decode.
    Weighted by validation accuracy.
    """
    device = images.device
    all_log_probs = []
    weights = []

    for name, model, acc in models:
        log_probs = model(images)  # [B, T, C]
        all_log_probs.append(log_probs)
        weights.append(acc)

    # Normalize weights
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    w = w / w.sum()

    # Weighted average in log-prob space:
    #   log( sum_i w_i * exp(log_prob_i) )
    stacked = torch.stack(all_log_probs, dim=0)       # [M, B, T, C]
    # Convert log-prob to prob, weighted average, back to log-prob
    probs = stacked.exp()                               # [M, B, T, C]
    w_view = w.view(-1, 1, 1, 1)
    avg_probs = (probs * w_view).sum(dim=0)             # [B, T, C]
    avg_log_probs = avg_probs.log()                     # [B, T, C]

    # Beam search decode
    batch_size = avg_log_probs.size(0)
    decoded_texts = []
    confidences = []
    for i in range(batch_size):
        text, score, _ = beam_search_decode(
            avg_log_probs[i].cpu(), idx2char,
            beam_width=beam_width, blank=0,
            format_bonus_weight=1.5,
        )
        decoded_texts.append(text)
        confidences.append(score)

    return decoded_texts, confidences


# ── Hard voting (majority vote, confidence-weighted) ────────────────────────
@torch.no_grad()
def ensemble_hard_vote(models, images, idx2char, beam_width=10):
    """
    Each model decodes independently, then majority vote.
    Ties broken by model accuracy weight.
    """
    per_model_preds = []
    per_model_confs = []

    for name, model, acc in models:
        log_probs = model(images)
        batch_size = log_probs.size(0)

        texts = []
        confs = []
        for i in range(batch_size):
            text, score, _ = beam_search_decode(
                log_probs[i].cpu(), idx2char,
                beam_width=beam_width, blank=0,
                format_bonus_weight=1.5,
            )
            texts.append(text)
            confs.append(score)
        per_model_preds.append((texts, confs, acc))

    # Majority vote per sample
    decoded_texts = []
    confidences = []
    for i in range(batch_size):
        votes = defaultdict(float)
        for texts, confs, acc in per_model_preds:
            pred = texts[i]
            conf = confs[i]
            votes[pred] += acc * conf  # weight by accuracy × confidence
        best = max(votes, key=votes.get)
        decoded_texts.append(best)
        confidences.append(votes[best])

    return decoded_texts, confidences


# ── Evaluation on val set ───────────────────────────────────────────────────
def evaluate_ensemble(models, val_loader, device, method='soft_vote',
                      beam_width=10):
    """Evaluate ensemble on validation set."""
    idx2char = Config.IDX2CHAR
    all_preds, all_targets, all_confs = [], [], []

    pbar = tqdm(val_loader, desc=f"Ensemble ({method})")
    for batch in pbar:
        images = batch[0].to(device)
        labels_text = batch[3]

        if method == 'soft_vote':
            decoded, confs = ensemble_soft_vote(
                models, images, idx2char, beam_width)
        else:
            decoded, confs = ensemble_hard_vote(
                models, images, idx2char, beam_width)

        all_preds.extend(decoded)
        all_targets.extend(labels_text)
        all_confs.extend(confs)

    acc = calculate_accuracy(all_preds, all_targets) * 100
    cer = calculate_cer(all_preds, all_targets)

    # Error analysis
    errors = [(t, p) for p, t in zip(all_preds, all_targets) if p != t]
    len_err = sum(1 for t, p in errors if len(p) != len(t))

    print(f"\n{'='*60}")
    print(f"  Ensemble ({method}) Results")
    print(f"{'='*60}")
    print(f"  Accuracy:   {acc:.2f}%")
    print(f"  CER:        {cer:.4f}")
    print(f"  Errors:     {len(errors)} / {len(all_targets)}")
    print(f"  Length err: {len_err} ({len_err*100/max(len(errors),1):.1f}%)")
    print(f"  Beam width: {beam_width}")

    # Samples
    print(f"\n  Sample predictions:")
    shown = 0
    for p, t in zip(all_preds, all_targets):
        if shown >= 10:
            break
        mark = "✓" if p == t else "✗"
        print(f"    [{mark}] GT: {t:8s}  Pred: {p:8s}")
        shown += 1

    return acc, cer


# ── Test set inference ──────────────────────────────────────────────────────
def inference_ensemble(models, test_dir, device, method='soft_vote',
                       beam_width=10, batch_size=32):
    """Run ensemble inference on test set."""
    from inference import TestDataset

    transform = get_val_transforms()
    test_ds = TestDataset(test_dir=test_dir, transform=transform, num_frames=5)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    idx2char = Config.IDX2CHAR
    predictions = {}

    pbar = tqdm(test_loader, desc=f"Test ensemble ({method})")
    for images, track_ids in pbar:
        images = images.to(device)

        if method == 'soft_vote':
            decoded, confs = ensemble_soft_vote(
                models, images, idx2char, beam_width)
        else:
            decoded, confs = ensemble_hard_vote(
                models, images, idx2char, beam_width)

        for track_id, text, conf in zip(track_ids, decoded, confs):
            predictions[track_id] = {'text': text, 'confidence': conf}

    return predictions


def save_predictions(predictions, output_path):
    """Save predictions in ICPR submission format."""
    sorted_preds = sorted(predictions.items(),
                          key=lambda x: int(x[0].split('_')[1]))
    with open(output_path, 'w') as f:
        for track_id, pred in sorted_preds:
            f.write(f"{track_id},{pred['text']};{pred['confidence']:.4f}\n")
    print(f"  Saved {len(predictions)} predictions to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Ensemble Inference')
    parser.add_argument('--mode', default='val', choices=['val', 'test'],
                        help='val = evaluate on val set, test = generate submission')
    parser.add_argument('--method', default='soft_vote',
                        choices=['soft_vote', 'hard_vote'])
    parser.add_argument('--beam_width', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_dir', default='Pa7a3Hin-test-public')
    parser.add_argument('--output', default='outputs/predictions/predictions_ensemble.txt')
    args = parser.parse_args()

    device = Config.DEVICE
    print("=" * 60)
    print("  ENSEMBLE INFERENCE")
    print("=" * 60)

    # Load all models
    print("\nLoading models...")
    models = load_models(device)
    print(f"  Loaded {len(models)} models\n")

    if args.mode == 'val':
        # Evaluate on val set
        val_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='val',
                                           split_ratio=0.8)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, collate_fn=val_ds.collate_fn,
                                num_workers=4, pin_memory=True)

        # Try both methods
        for method in ['soft_vote', 'hard_vote']:
            evaluate_ensemble(models, val_loader, device,
                              method=method, beam_width=args.beam_width)

    else:
        # Test set inference
        predictions = inference_ensemble(
            models, args.test_dir, device,
            method=args.method, beam_width=args.beam_width,
            batch_size=args.batch_size)
        save_predictions(predictions, args.output)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    sys.exit(0)


if __name__ == '__main__':
    main()
