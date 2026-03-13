"""
Fast Enhanced Inference — ICPR 2026 LRLPR Challenge (Blind Test).

KEY DESIGN: All TTA views processed as a BATCH on GPU — not one-by-one in Python.
Each track returns [N_TTA, 5, 3, H, W]; entire batch → single model forward call.

INFERENCE IMPROVEMENTS (no retraining):
  1. Image pre-processing: CLAHE + bilateral denoise + unsharp mask  (CPU, lossless)
  2. TTA 6 views: averaged in batch on GPU
  3. Multi-model ensemble: Phase10 + Phase8SWA, accuracy-weighted soft vote
  4. Beam search: width=25, format bonus=2.5  (vs width=10, bonus=1.5 default)
  5. Format correction: position-specific O↔0, I↔1, B↔8, S↔5, G↔6 correction

Usage:
    python inference_enhanced.py
    python inference_enhanced.py --test_dir ../TKzFBtn7-test-blind
    python inference_enhanced.py --fast   # greedy, no TTA, no preprocess
"""

import os, sys, argparse, cv2, re, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models.recognizer_v3 import Phase3Recognizer
from postprocess import validate_plate_format

IMG_H, IMG_W = 64, 224

# ── 1. Image pre-processing ──────────────────────────────────────────────────
def preprocess(img):
    img = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.25, blurred, -0.25, 0)
    return np.clip(img, 0, 255).astype(np.uint8)

# ── 2. TTA transforms (6 views) ──────────────────────────────────────────────
_N = dict(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
TTA_TRANSFORMS = [
    A.Compose([A.Resize(IMG_H,IMG_W), A.Normalize(**_N), ToTensorV2()]),
    A.Compose([A.Resize(IMG_H,IMG_W), A.RandomBrightnessContrast(brightness_limit=(0.08,0.12), contrast_limit=(0.05,0.1), p=1), A.Normalize(**_N), ToTensorV2()]),
    A.Compose([A.Resize(IMG_H,IMG_W), A.RandomBrightnessContrast(brightness_limit=(-0.12,-0.08), contrast_limit=(0.05,0.1), p=1), A.Normalize(**_N), ToTensorV2()]),
    A.Compose([A.Resize(IMG_H,IMG_W), A.Sharpen(alpha=(0.2,0.4), lightness=(0.9,1.1), p=1), A.Normalize(**_N), ToTensorV2()]),
    A.Compose([A.Resize(IMG_H,IMG_W), A.CLAHE(clip_limit=2.0, tile_grid_size=(4,4), p=1), A.Normalize(**_N), ToTensorV2()]),
    A.Compose([A.Resize(IMG_H,IMG_W), A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.12,0.18), p=1), A.Normalize(**_N), ToTensorV2()]),
]

# ── 3. Dataset ────────────────────────────────────────────────────────────────
class BlindDataset(Dataset):
    """Returns [N_TTA, 5, 3, H, W] per track — all TTA views precomputed."""
    def __init__(self, test_dir, tta_transforms=TTA_TRANSFORMS, use_preprocess=True):
        self.tta_transforms = tta_transforms
        self.use_preprocess = use_preprocess
        self.tracks = []
        for name in sorted(os.listdir(test_dir)):
            path = os.path.join(test_dir, name)
            if os.path.isdir(path) and name.startswith('track_'):
                files = sorted([os.path.join(path, f) for f in os.listdir(path)
                                if f.startswith('lr-') and f.endswith(('.jpg','.png'))])
                if files:
                    self.tracks.append((name, files))
        print(f"  Found {len(self.tracks)} test tracks")

    def __len__(self): return len(self.tracks)

    def _load(self, path):
        img = cv2.imread(path)
        img = np.zeros((20,50,3), dtype=np.uint8) if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return preprocess(img) if self.use_preprocess else img

    def __getitem__(self, idx):
        track_id, files = self.tracks[idx]
        paths = (files[:5] + [files[-1]] * 5)[:5]
        frames = [self._load(p) for p in paths]
        views = torch.stack([torch.stack([tf(image=f)['image'] for f in frames])
                             for tf in self.tta_transforms])  # [N_TTA, 5, 3, H, W]
        return views, track_id

def collate_fn(batch):
    xs, ids = zip(*batch)
    return torch.stack(xs), list(ids)

# ── 4. Model loading ─────────────────────────────────────────────────────────
def load_model(path, device):
    if not os.path.exists(path):
        print(f"  ❌ Not found: {path}"); return None, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {}); p10 = ckpt.get('phase10_config', {})
    m = Phase3Recognizer(
        num_classes=Config.NUM_CLASSES, use_stn=cfg.get('stn', True),
        use_resnet_backbone=True,
        hidden_size=p10.get('hidden_size', 256), num_lstm_layers=p10.get('num_lstm_layers', 2),
        dropout=0.0, fusion_reduction=p10.get('fusion_reduction', 16), use_sr_branch=False,
    ).to(device)
    m.load_state_dict(ckpt['model_state_dict'], strict=False)
    m.eval()
    acc = ckpt.get('accuracy', 0)
    print(f"  ✅ {os.path.basename(path):40s}  acc={acc:.2f}%  epoch={ckpt.get('epoch',-1)+1}")
    return m, acc

# ── 5. Format correction ─────────────────────────────────────────────────────
L2D = {'O':'0','I':'1','Z':'2','S':'5','G':'6','B':'8','T':'7','A':'4','L':'1'}
D2L = {'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B','7':'T','4':'A'}
LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
DIGITS  = set('0123456789')

def _fix(chars, pos_l, pos_d):
    out = list(chars)
    for i in pos_l:
        if out[i] in DIGITS:
            if out[i] in D2L: out[i] = D2L[out[i]]
            else: return None
    for i in pos_d:
        if out[i] in LETTERS:
            if out[i] in L2D: out[i] = L2D[out[i]]
            else: return None
    return ''.join(out)

def correct_format(text):
    if len(text) != 7: return text, 'unchanged'
    if validate_plate_format(text)[0]: return text, validate_plate_format(text)[1]
    chars = list(text.upper())
    br = _fix(chars, [0,1,2], [3,4,5,6])
    mc = _fix(chars, [0,1,2,4], [3,5,6])
    br_v = br and validate_plate_format(br)[0]
    mc_v = mc and validate_plate_format(mc)[0]
    if br_v and mc_v:
        return (br,'corrected_br') if sum(a!=b for a,b in zip(chars,list(br))) <= sum(a!=b for a,b in zip(chars,list(mc))) else (mc,'corrected_mc')
    if br_v: return br, 'corrected_br'
    if mc_v: return mc, 'corrected_mc'
    return text, 'uncorrectable'

# ── 6. Beam search decode ────────────────────────────────────────────────────
def beam_decode(log_probs, beam_width=25, blank=0, format_bonus=2.5):
    NEG_INF = float('-inf')
    def la(a,b):
        if a==NEG_INF: return b
        if b==NEG_INF: return a
        mx=max(a,b); return mx+math.log1p(math.exp(-abs(a-b)))
    T, C = log_probs.shape
    beams = {'': (0.0, NEG_INF)}
    for t in range(T):
        nb = defaultdict(lambda: (NEG_INF, NEG_INF))
        lp = log_probs[t]
        for prefix, (pb, pnb) in beams.items():
            pt = la(pb, pnb)
            for c in range(C):
                lc = lp[c].item()
                if c == blank:
                    ob,onb = nb[prefix]; nb[prefix] = (la(ob, pt+lc), onb)
                else:
                    ch = Config.IDX2CHAR.get(c,'')
                    last = prefix[-1] if prefix else ''
                    if ch == last:
                        ob,onb = nb[prefix]; nb[prefix] = (ob, la(onb, pnb+lc))
                        np2 = prefix+ch; ob2,onb2 = nb[np2]; nb[np2] = (ob2, la(onb2, pb+lc))
                    else:
                        np2 = prefix+ch; ob2,onb2 = nb[np2]; nb[np2] = (ob2, la(onb2, pt+lc))
        beams = dict(sorted(nb.items(), key=lambda x: la(x[1][0],x[1][1]), reverse=True)[:beam_width])
    best, best_s = '', NEG_INF
    for p,(pb,pnb) in beams.items():
        raw=la(pb,pnb); s=raw+(format_bonus if validate_plate_format(p)[0] else 0.0)
        if s>best_s: best_s,best=s,p
    raw=la(beams.get(best,(NEG_INF,NEG_INF))[0], beams.get(best,(NEG_INF,NEG_INF))[1])
    return best, min(1.0, math.exp(raw/max(T,1)))

# ── 7. Batch inference ───────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(models_weights, loader, device, beam_width=25, format_bonus=2.5,
                  use_format_correction=True):
    total_w = sum(w for _,w in models_weights)
    predictions = {}; corr_stats = defaultdict(int)
    for x_batch, track_ids in tqdm(loader, desc="Inference"):
        B, N_TTA, T5, C3, H, W = x_batch.shape  # [B, N_TTA, 5, 3, H, W]
        x_flat = x_batch.view(B*N_TTA, T5, C3, H, W).to(device, non_blocking=True)
        avg_probs = None
        for model, weight in models_weights:
            probs = torch.softmax(model(x_flat), dim=-1)  # [B*N_TTA, T_seq, C]
            avg_probs = probs*(weight/total_w) if avg_probs is None else avg_probs + probs*(weight/total_w)
        T_seq, num_cls = avg_probs.shape[1], avg_probs.shape[2]
        avg_log = avg_probs.view(B, N_TTA, T_seq, num_cls).mean(dim=1).log()  # [B, T, C]
        for i, tid in enumerate(track_ids):
            text, conf = beam_decode(avg_log[i].cpu(), beam_width, format_bonus=format_bonus)
            if use_format_correction:
                text, corr = correct_format(text)
            else:
                corr = 'none'
            predictions[tid] = {'text': text, 'confidence': conf, 'correction': corr}
            corr_stats[corr] += 1
    return predictions, corr_stats

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir',   default='../TKzFBtn7-test-blind')
    ap.add_argument('--models',     nargs='+', default=['checkpoints/best_model_phase10.pth','checkpoints/best_model_phase8_swa.pth'])
    ap.add_argument('--output',     default='outputs/predictions/predictions_blind_enhanced.txt')
    ap.add_argument('--submission', default='outputs/submissions/submission_blind_enhanced.txt')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--beam_width', type=int, default=25)
    ap.add_argument('--fast', action='store_true', help='No TTA, no preprocess, greedy')
    ap.add_argument('--no_preprocess',        action='store_true')
    ap.add_argument('--no_format_correction', action='store_true')
    args = ap.parse_args()

    device = Config.DEVICE
    tta = TTA_TRANSFORMS[:1] if args.fast else TTA_TRANSFORMS
    use_prep = not (args.fast or args.no_preprocess)

    print("="*70)
    print("  FAST ENHANCED INFERENCE — ICPR 2026 LRLPR")
    print("="*70)
    print(f"  Device: {device}  |  TTA views: {len(tta)}  |  Preprocess: {use_prep}")
    print(f"  Beam width: {args.beam_width}  |  Batch: {args.batch_size}")

    print("\n  Loading models...")
    mw = [(m,acc) for mp in args.models for m,acc in [load_model(mp, device)] if m]
    if not mw: print("  No models!"); return

    print(f"\n  Loading test data...")
    ds     = BlindDataset(args.test_dir, tta_transforms=tta, use_preprocess=use_prep)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    preds, corr_stats = run_inference(mw, loader, device,
                                      beam_width=args.beam_width, format_bonus=2.5,
                                      use_format_correction=not args.no_format_correction)

    sorted_p = sorted(preds.items(), key=lambda x: int(x[0].split('_')[1]))

    with open(args.output, 'w') as f:
        for tid,d in sorted_p: f.write(f"{tid},{d['text']};{d['confidence']:.4f}\n")
    with open(args.submission, 'w') as f:
        for tid,d in sorted_p: f.write(f"{tid},{d['text']}\n")

    valid_n = sum(1 for d in preds.values() if validate_plate_format(d['text'])[0])
    avg_c   = float(np.mean([d['confidence'] for d in preds.values()]))
    low_c   = sum(1 for d in preds.values() if d['confidence'] < 0.5)

    print(f"\n{'='*70}")
    print(f"  Predictions:       {len(preds)}")
    print(f"  Valid format:      {valid_n}/{len(preds)} ({100*valid_n/len(preds):.1f}%)")
    print(f"  Avg confidence:    {avg_c:.4f}")
    print(f"  Low conf (<0.5):   {low_c}")
    print(f"  Corrections:       {dict(corr_stats)}")
    print(f"\n  Samples:")
    for tid,d in sorted_p[:15]:
        v,_ = validate_plate_format(d['text'])
        corr = f" [{d['correction']}]" if d['correction'] not in ('none','brazilian_old','mercosur') else ''
        print(f"    {tid}: {d['text']}  conf={d['confidence']:.4f}  {'✓' if v else '?'}{corr}")
    print(f"\n  Saved: {args.output}")
    print(f"  Saved: {args.submission}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
