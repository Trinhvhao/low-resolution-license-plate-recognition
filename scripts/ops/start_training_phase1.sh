#!/bin/bash
# Phase 1 Training Script with PM2
# Usage: ./start_training_phase1.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
BASELINE_DIR="$PROJECT_ROOT/baseline"
LOGS_DIR="$PROJECT_ROOT/logs"

echo "🚀 Starting Phase 1 Training with PM2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$BASELINE_DIR"

# Check if already running
if pm2 describe lpr-phase1 > /dev/null 2>&1; then
    echo "⚠️  Phase 1 training already running!"
    echo ""
    pm2 describe lpr-phase1
    echo ""
    echo "Options:"
    echo "  • View logs:    pm2 logs lpr-phase1"
    echo "  • Stop:         pm2 stop lpr-phase1"
    echo "  • Restart:      pm2 restart lpr-phase1"
    echo "  • Delete:       pm2 delete lpr-phase1"
    exit 1
fi

# Create PM2 ecosystem file for Phase 1
cat > ecosystem.phase1.config.js << EOF
module.exports = {
  apps: [{
    name: 'lpr-phase1',
    script: 'python',
    args: 'train.py',
    cwd: '$BASELINE_DIR',
    interpreter: 'none',
    instances: 1,
    autorestart: false,
    watch: false,
    max_memory_restart: '20G',
    env: {
      CUDA_VISIBLE_DEVICES: '0',
      PYTHONUNBUFFERED: '1'
    },
    error_file: '$LOGS_DIR/phase1-error.log',
    out_file: '$LOGS_DIR/phase1-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    merge_logs: true,
    time: true
  }]
};
EOF

# Create logs directory
mkdir -p "$LOGS_DIR"

echo ""
echo "📋 Configuration:"
echo "  • Name:         lpr-phase1"
echo "  • Script:       train.py"
echo "  • Working dir:  baseline/"
echo "  • GPU:          CUDA:0"
echo "  • Auto-restart: No"
echo "  • Logs:         logs/phase1-*.log"
echo ""

# Start with PM2
pm2 start ecosystem.phase1.config.js

echo ""
echo "✅ Phase 1 training started!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Monitoring Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  pm2 list                    # List all processes"
echo "  pm2 logs lpr-phase1         # View real-time logs"
echo "  pm2 logs lpr-phase1 --lines 100"
echo "  pm2 monit                   # Resource monitor"
echo ""
echo "🎯 TensorBoard:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  tensorboard --logdir runs --port 6006"
echo ""
echo "🛑 Control:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  pm2 stop lpr-phase1         # Pause training"
echo "  pm2 restart lpr-phase1      # Resume training"
echo "  pm2 delete lpr-phase1       # Remove from PM2"
echo ""
echo "📂 Outputs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  • Model:    baseline/best_model_phase1.pth"
echo "  • Logs:     logs/phase1-*.log"
echo "  • TBoard:   baseline/runs/"
echo ""
