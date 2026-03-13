#!/bin/bash
set -e
cd /home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/baseline
export PATH="/home/nhannv/.conda/envs/ocr/bin:$PATH"
/home/nhannv/.conda/envs/ocr/bin/python train_phase2_update.py
exit 0
