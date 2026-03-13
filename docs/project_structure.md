# Project Structure (Reorganized)

## Root

- `README.md`: Legacy README.
- `START_HERE.md`: Quick navigation and command entrypoints.
- `requirements.txt`: Root dependencies.
- `val_tracks.json`: Validation metadata.
- `baseline/`: Main OCR/LPR implementation and artifacts.
- `scripts/`: Operational and test scripts.
- `configs/`: Configuration files by subsystem.
- `assets/`: Archives and static assets.
- `docs/`: Documentation.
- `train/`: Training data.
- `logs/`: Runtime logs.

## scripts/

- `scripts/ops/`
  - `train.sh`
  - `start_training_phase1.sh`
  - `start_training_fast_a30.sh`
  - `monitor_phase1.sh`
- `scripts/test/`
  - `test_fast_a30_config.py`

## baseline/

- `baseline/checkpoints/`
  - `best_model*.pth`
- `baseline/logs/`
  - `lpr-phase*-{out,err}.log`
- `baseline/outputs/`
  - `baseline/outputs/predictions/`
  - `baseline/outputs/submissions/`
  - `baseline/outputs/evaluations/`
- `baseline/scripts/ops/`
  - `run_phase2.sh`
  - `run_phase2_fixed.sh`
  - `run_phase2_update.sh`
  - `run_phase3.sh`
  - `test.sh`
  - `ensemble.py`
- `baseline/scripts/test/`
  - `test.py`
  - `test_phase1.py`
  - `test_training_phase1.py`
- `baseline/scripts/analyze/`
  - `analyze_blind_differences.py`
  - `analyze_phase8.py`
  - `analyze_phase8_deep.py`
  - `analyze_results.py`
  - `check_predictions.py`
  - `compare_predictions.py`
- `baseline/scripts/train/`
  - `train.py`
  - `train_phase2.py` ... `train_phase13.py`
  - `train_fast_a30.py`

## configs/

- `configs/pm2/`
  - `ecosystem.config.js.bak`

## assets/

- `assets/archives/`
  - `Pa7a3Hin-test-public.zip`
  - `TKzFBtn7-test-blind.zip`
  - `wYe7pBJ7-train.zip`

## Compatibility Wrappers

Root-level wrappers were kept for operational continuity:

- `train.sh`
- `start_training_phase1.sh`
- `start_training_fast_a30.sh`
- `monitor_phase1.sh`
- `test_fast_a30_config.py`

Baseline root wrappers were also kept:

- `baseline/run_phase2.sh`
- `baseline/run_phase2_fixed.sh`
- `baseline/run_phase2_update.sh`
- `baseline/run_phase3.sh`
- `baseline/test.sh`

## baseline/ Core Files (Cleaned)

- `config.py`
- `dataset.py`
- `transforms.py`
- `utils.py`
- `inference.py`
- `inference_enhanced.py`
- `pipeline.py`
- `postprocess.py`
- `__init__.py`
