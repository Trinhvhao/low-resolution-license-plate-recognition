#!/bin/bash
# Phase 3 Training — PM2 runner
cd "$(dirname "$0")"
source activate ocr 2>/dev/null || conda activate ocr 2>/dev/null
exec python train_phase3.py
