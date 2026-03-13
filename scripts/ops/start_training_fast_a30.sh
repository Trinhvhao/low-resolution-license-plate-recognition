#!/bin/bash

# 🚀 Fast Training Script for A30 GPU
# Optimized for maximum throughput and VRAM utilization

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "🚀 FAST TRAINING - A30 GPU OPTIMIZED"
echo "=========================================="
echo ""

# Change to baseline directory
cd "$PROJECT_ROOT/baseline" || exit 1

echo "📍 Working directory: $(pwd)"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found! Make sure CUDA is installed."
    exit 1
fi

echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Check if previous fast training is running
if pm2 describe lpr-fast-a30 &> /dev/null; then
    echo "⚠️  Found existing 'lpr-fast-a30' process"
    read -p "   Stop and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping old process..."
        pm2 delete lpr-fast-a30
    else
        echo "❌ Aborted. Please stop manually: pm2 delete lpr-fast-a30"
        exit 1
    fi
fi

echo "🚀 Starting fast training with PM2..."
pm2 start ecosystem.fast_a30.config.js

echo ""
echo "✅ Training started!"
echo ""
echo "📊 Monitor with:"
echo "   pm2 status"
echo "   pm2 logs lpr-fast-a30"
echo "   pm2 monit"
echo ""
echo "📈 TensorBoard (in another terminal):"
echo "   tensorboard --logdir runs --port 6006 --bind_all"
echo ""
echo "🛑 To stop:"
echo "   pm2 stop lpr-fast-a30"
echo ""
echo "=========================================="
