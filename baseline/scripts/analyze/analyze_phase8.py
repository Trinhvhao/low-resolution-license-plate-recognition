"""Quick error analysis script for Phase 8 SWA model."""
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
scenario_acc = {}
char_confusions = Counter()
position_errors = Counter()
wrong_by_scenario = defaultdict(list)

with torch.no_grad():
    for sc_dir in ['Scenario-A', 'Scenario-B']:
        for layout_dir in ['Brazilian', 'Mercosur']:
            scenario_name = f'{sc_dir[-1]}_{layout_dir}'
            base_path = os.path.join(train_root, sc_dir, layout_dir)
            if not os.path.isdir(base_path):
                continue
            sc_c, sc_t = 0, 0
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
                decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)[0]
                sc_t += 1
                if decoded == label:
                    sc_c += 1
                else:
                    wrong_by_scenario[scenario_name].append((label, decoded))
                    for pos in range(max(len(label), len(decoded))):
                        gt_c = label[pos] if pos < len(label) else '_'
                        pr_c = decoded[pos] if pos < len(decoded) else '_'
                        if gt_c != pr_c:
                            char_confusions[(gt_c, pr_c)] += 1
                            position_errors[pos] += 1
            scenario_acc[scenario_name] = (sc_c, sc_t)
            print(f'  Done {scenario_name}: {sc_c}/{sc_t} = {sc_c/sc_t*100:.2f}%')

print()
print('=' * 60)
print('  Phase 8 SWA: Accuracy by Scenario')
print('=' * 60)
tc, tt = 0, 0
for sc in sorted(scenario_acc.keys()):
    c, t = scenario_acc[sc]
    acc = c/t*100 if t > 0 else 0
    tc += c
    tt += t
    print(f'  {sc:15s}: {c:4d}/{t:4d} = {acc:.2f}%')
print(f'  {"TOTAL":15s}: {tc:4d}/{tt:4d} = {tc/tt*100:.2f}%')

p4v2 = {'A_Brazilian': 83.97, 'A_Mercosur': 84.55, 'B_Brazilian': 52.12, 'B_Mercosur': 79.28}
print()
print('  Delta vs Phase 4 v2 (78.47%):')
for sc in sorted(scenario_acc.keys()):
    c, t = scenario_acc[sc]
    acc = c/t*100 if t > 0 else 0
    old = p4v2.get(sc, 0)
    print(f'    {sc}: {old:.2f}% -> {acc:.2f}% ({acc-old:+.2f}%)')

print()
print('  Top 20 Char Confusions (GT->Pred):')
for (gt, pr), cnt in char_confusions.most_common(20):
    print(f'    {gt}->{pr}: {cnt}')

print()
print('  Errors by Position:')
for pos in sorted(position_errors.keys()):
    print(f'    Pos {pos}: {position_errors[pos]}')

print()
print('  Wrong per Scenario:')
for sc in sorted(wrong_by_scenario.keys()):
    n = len(wrong_by_scenario[sc])
    print(f'    {sc}: {n} wrong')
    for gt, pr in wrong_by_scenario[sc][:5]:
        print(f'      GT: {gt}  Pred: {pr}')
