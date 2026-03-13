#!/bin/bash
# Quick reference for monitoring Phase 1 training

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 PHASE 1 TRAINING - QUICK REFERENCE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if running
if ! pm2 describe lpr-phase1 > /dev/null 2>&1; then
    echo "❌ Training not running!"
    echo "   Start with: ./start_training_phase1.sh"
    exit 1
fi

echo "✅ Training Status:"
pm2 describe lpr-phase1 | grep -E "status|uptime|memory|cpu|restarts"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 MONITORING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Real-time logs:"
echo "  pm2 logs lpr-phase1"
echo ""
echo "Last 50 lines:"
echo "  pm2 logs lpr-phase1 --lines 50 --nostream"
echo ""
echo "Resource monitor:"
echo "  pm2 monit"
echo ""
echo "GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 TENSORBOARD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  tensorboard --logdir baseline/runs --port 6006"
echo "  # Then open: http://localhost:6006"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛑 CONTROL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  pm2 stop lpr-phase1      # Pause"
echo "  pm2 restart lpr-phase1   # Resume"
echo "  pm2 delete lpr-phase1    # Stop & remove"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 FILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f "baseline/best_model_phase1.pth" ]; then
    echo "✓ Model checkpoint:"
    ls -lh baseline/best_model_phase1.pth
    echo ""
fi

if [ -d "baseline/runs" ]; then
    echo "✓ TensorBoard logs:"
    ls -lh baseline/runs/ | tail -5
    echo ""
fi

echo "Log files:"
ls -lh logs/phase1-*.log 2>/dev/null || echo "  (logs will appear after first epoch)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  ESTIMATED TIME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  • Dataset: 16,000 train + 4,000 val"
echo "  • Batch size: 96"
echo "  • Batches/epoch: ~167 train + ~42 val"
echo "  • Time/epoch: ~5-7 minutes (estimated)"
echo "  • Total time: 2.5-3.5 hours (50 epochs max)"
echo "  • Early stopping: patience=7 epochs"
echo ""
