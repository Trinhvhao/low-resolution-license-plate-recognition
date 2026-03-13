"""Deep analysis of Phase 8 SWA model — per-character, per-position, per-scenario."""
import torch, sys, os, json, cv2, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from utils import decode_predictions, seed_everything
from models.recognizer_v3 import Phase3Recognizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

seed_everything(42)
device = 'cuda'

# Load P8 SWA model
ckpt = torch.load('checkpoints/best_model_phase8_swa.pth', map_location='cpu', weights_only=False)
model = Phase3Recognizer(num_classes=37, use_stn=True, use_resnet_backbone=True,
    hidden_size=256, num_lstm_layers=2, dropout=0.25, fusion_reduction=16, use_sr_branch=True).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

with open('/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/val_tracks.json') as f:
    val_tracks = json.load(f)
val_set = set(val_tracks)

val_transform = A.Compose([
    A.Resize(height=64, width=224),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])

train_root = '/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/train'

# Storage
all_results = []  # (scenario, label, pred, confidence)

with torch.no_grad():
    for sc_dir in ['Scenario-A', 'Scenario-B']:
        for layout_dir in ['Brazilian', 'Mercosur']:
            scenario_name = f'{sc_dir[-1]}_{layout_dir}'
            base_path = os.path.join(train_root, sc_dir, layout_dir)
            if not os.path.isdir(base_path):
                continue
            for track_dir in sorted(os.listdir(base_path)):
                if track_dir not in val_set:
                    continue
                track_path = os.path.join(base_path, track_dir)
                ann_file = os.path.join(track_path, 'annotations.json')
                if not os.path.exists(ann_file):
                    continue
                with open(ann_file) as f:
                    ann = json.load(f)
                label = ann['plate_text']
                lr_paths = sorted([os.path.join(track_path, ff) for ff in os.listdir(track_path)
                                 if ff.startswith('lr-') and (ff.endswith('.jpg') or ff.endswith('.png'))])
                if not lr_paths:
                    continue
                if len(lr_paths) < 5:
                    lr_paths = lr_paths + [lr_paths[-1]] * (5 - len(lr_paths))
                else:
                    lr_paths = lr_paths[:5]
                frames = []
                for p in lr_paths:
                    img = cv2.imread(p)
                    if img is None:
                        img = np.zeros((64, 224, 3), dtype=np.uint8)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = val_transform(image=img)['image']
                    frames.append(img)
                x = torch.stack(frames).unsqueeze(0).to(device)
                preds = model(x)
                # Get confidence
                probs = torch.softmax(preds, dim=-1)
                max_probs = probs.max(dim=-1).values.mean().item()
                decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)[0]
                all_results.append((scenario_name, label, decoded, max_probs))

print(f"Total samples: {len(all_results)}")

# ═══════════════════════════════════════════════════════
# 1. Per-Scenario Accuracy
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  1. ACCURACY BY SCENARIO')
print('='*70)
sc_stats = defaultdict(lambda: [0, 0])
for sc, label, pred, conf in all_results:
    sc_stats[sc][1] += 1
    if label == pred:
        sc_stats[sc][0] += 1
tc, tt = 0, 0
for sc in sorted(sc_stats.keys()):
    c, t = sc_stats[sc]
    tc += c; tt += t
    print(f'  {sc:15s}: {c:4d}/{t:4d} = {c/t*100:.2f}%')
print(f'  {"TOTAL":15s}: {tc:4d}/{tt:4d} = {tc/tt*100:.2f}%')

# ═══════════════════════════════════════════════════════
# 2. Per-Character Accuracy (GT char → how often correct)
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  2. PER-CHARACTER ACCURACY')
print('='*70)
char_correct = Counter()
char_total = Counter()
for _, label, pred, _ in all_results:
    for pos in range(min(len(label), len(pred))):
        gc = label[pos]
        char_total[gc] += 1
        if pred[pos] == gc:
            char_correct[gc] += 1

print(f'  {"Char":>4s}  {"Correct":>7s} / {"Total":>5s}  {"Acc%":>6s}')
for c in sorted(char_total.keys()):
    acc = char_correct[c] / char_total[c] * 100
    print(f'  {c:>4s}  {char_correct[c]:7d} / {char_total[c]:5d}  {acc:6.2f}%')

# Worst characters
print('\n  Bottom 10 characters (worst accuracy):')
char_accs = [(c, char_correct[c]/char_total[c]*100, char_total[c]) for c in char_total]
char_accs.sort(key=lambda x: x[1])
for c, acc, total in char_accs[:10]:
    print(f'    {c}: {acc:.2f}% ({total} total)')

# ═══════════════════════════════════════════════════════
# 3. Per-Character PER-SCENARIO Accuracy (find hotspots)
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  3. WORST CHAR-SCENARIO COMBINATIONS')
print('='*70)
sc_char_correct = defaultdict(Counter)
sc_char_total = defaultdict(Counter)
for sc, label, pred, _ in all_results:
    for pos in range(min(len(label), len(pred))):
        gc = label[pos]
        sc_char_total[sc][gc] += 1
        if pred[pos] == gc:
            sc_char_correct[sc][gc] += 1

combos = []
for sc in sc_char_total:
    for c in sc_char_total[sc]:
        t = sc_char_total[sc][c]
        cr = sc_char_correct[sc][c]
        if t >= 10:  # minimum sample count
            acc = cr / t * 100
            combos.append((sc, c, acc, cr, t))
combos.sort(key=lambda x: x[2])
print(f'  {"Scenario":>15s}  {"Char":>4s}  {"Acc%":>6s}  {"Correct/Total":>13s}')
for sc, c, acc, cr, t in combos[:20]:
    print(f'  {sc:>15s}  {c:>4s}  {acc:6.2f}%  {cr:5d}/{t:5d}')

# ═══════════════════════════════════════════════════════
# 4. Per-Position Accuracy
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  4. PER-POSITION ACCURACY')
print('='*70)
pos_correct = Counter()
pos_total = Counter()
for _, label, pred, _ in all_results:
    for pos in range(max(len(label), len(pred))):
        gt = label[pos] if pos < len(label) else ''
        pr = pred[pos] if pos < len(pred) else ''
        pos_total[pos] += 1
        if gt == pr:
            pos_correct[pos] += 1
for pos in sorted(pos_total.keys()):
    acc = pos_correct[pos] / pos_total[pos] * 100
    print(f'  Position {pos}: {pos_correct[pos]}/{pos_total[pos]} = {acc:.2f}%')

# ═══════════════════════════════════════════════════════
# 5. Per-Position PER-SCENARIO Accuracy
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  5. PER-POSITION PER-SCENARIO ACCURACY')
print('='*70)
sc_pos_correct = defaultdict(Counter)
sc_pos_total = defaultdict(Counter)
for sc, label, pred, _ in all_results:
    for pos in range(max(len(label), len(pred))):
        gt = label[pos] if pos < len(label) else ''
        pr = pred[pos] if pos < len(pred) else ''
        sc_pos_total[sc][pos] += 1
        if gt == pr:
            sc_pos_correct[sc][pos] += 1
for sc in sorted(sc_pos_total.keys()):
    parts = []
    for pos in sorted(sc_pos_total[sc].keys()):
        acc = sc_pos_correct[sc][pos] / sc_pos_total[sc][pos] * 100
        parts.append(f'P{pos}:{acc:.1f}%')
    print(f'  {sc:15s}: {" | ".join(parts)}')

# ═══════════════════════════════════════════════════════
# 6. Top Confusions Per Scenario
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  6. TOP 10 CONFUSIONS PER SCENARIO')
print('='*70)
sc_confusions = defaultdict(Counter)
for sc, label, pred, _ in all_results:
    for pos in range(min(len(label), len(pred))):
        if label[pos] != pred[pos]:
            sc_confusions[sc][(label[pos], pred[pos])] += 1
for sc in sorted(sc_confusions.keys()):
    top = sc_confusions[sc].most_common(10)
    pairs = ', '.join(f'{g}→{p}:{n}' for (g, p), n in top)
    print(f'  {sc:15s}: {pairs}')

# ═══════════════════════════════════════════════════════
# 7. Confidence Analysis
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  7. CONFIDENCE ANALYSIS')
print('='*70)
correct_confs = [conf for _, l, p, conf in all_results if l == p]
wrong_confs = [conf for _, l, p, conf in all_results if l != p]
print(f'  Correct: mean={np.mean(correct_confs):.4f}, median={np.median(correct_confs):.4f}')
print(f'  Wrong:   mean={np.mean(wrong_confs):.4f}, median={np.median(wrong_confs):.4f}')

# Per-scenario confidence
for sc in sorted(sc_stats.keys()):
    sc_corr = [conf for s, l, p, conf in all_results if s == sc and l == p]
    sc_wrng = [conf for s, l, p, conf in all_results if s == sc and l != p]
    if sc_wrng:
        print(f'  {sc}: correct={np.mean(sc_corr):.4f}, wrong={np.mean(sc_wrng):.4f}, gap={np.mean(sc_corr)-np.mean(sc_wrng):.4f}')

# ═══════════════════════════════════════════════════════
# 8. Length mismatch errors
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  8. LENGTH MISMATCH ERRORS')
print('='*70)
len_mismatch = [(sc, l, p) for sc, l, p, _ in all_results if len(l) != len(p)]
print(f'  Total length mismatches: {len(len_mismatch)}/{len(all_results)} ({100*len(len_mismatch)/len(all_results):.2f}%)')
sc_len_err = Counter()
for sc, l, p in len_mismatch:
    sc_len_err[sc] += 1
for sc in sorted(sc_len_err.keys()):
    total_sc = sc_stats[sc][1]
    print(f'  {sc}: {sc_len_err[sc]}/{total_sc} ({100*sc_len_err[sc]/total_sc:.2f}%)')
if len_mismatch:
    print('  Sample mismatches:')
    for sc, l, p in len_mismatch[:10]:
        print(f'    [{sc}] GT({len(l)}): {l}  Pred({len(p)}): {p}')

# ═══════════════════════════════════════════════════════
# 9. Number of frames analysis
# ═══════════════════════════════════════════════════════
print('\n' + '='*70)
print('  9. SUMMARY — KEY INSIGHTS FOR PHASE 10')
print('='*70)
total_wrong = sum(1 for _, l, p, _ in all_results if l != p)
total_right = sum(1 for _, l, p, _ in all_results if l == p)
print(f'  Total: {total_right}/{len(all_results)} correct ({total_right/len(all_results)*100:.2f}%)')
print(f'  Wrong: {total_wrong}')
print()
print('  Biggest improvement opportunities:')
for sc, c, acc, cr, t in combos[:10]:
    potential = t - cr
    print(f'    {sc}/{c}: {acc:.1f}% → fixing could gain up to {potential} samples')
