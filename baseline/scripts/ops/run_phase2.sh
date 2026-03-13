#!/bin/bash
# Phase 2 training runner — always exits 0 so PM2 does not autorestart on completion
set -e
cd /home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/baseline
export PATH="/home/nhannv/.conda/envs/ocr/bin:$PATH"
/home/nhannv/.conda/envs/ocr/bin/python train_phase2.py
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[run_phase2.sh] Training failed with exit code $EXIT_CODE" >&2
fi
# Always exit 0 — PM2 --no-autorestart only prevents restart on non-zero exit on some versions
exit 0
