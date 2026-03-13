#!/bin/bash
# Phase 2 Fixed Training — PM2 runner
# Fixes applied: discriminative LR, pct_start=0.3, weight_decay=1e-4, no freeze
# Architecture: Transformer (Phase2Recognizer)
# Saves to: best_model_phase2_fixed.pth

cd /home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/baseline

pm2 delete lpr-phase2-fixed 2>/dev/null

pm2 start \
  --name lpr-phase2-fixed \
  --interpreter /home/nhannv/.conda/envs/ocr/bin/python \
  --no-autorestart \
  -- train_phase2.py

echo "Phase 2 Fixed launched. Monitor with:"
echo "  pm2 logs lpr-phase2-fixed --lines 50"
echo "  pm2 list"
