#!/bin/bash
# Main script để quản lý training với pm2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================================${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null; then
    print_error "pm2 is not installed!"
    echo "Install with: npm install -g pm2"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Parse command
case "$1" in
    start)
        print_header "Starting Training with PM2"
        
        # Check if already running
        if pm2 list | grep -q "lpr-training.*online"; then
            print_error "Training is already running!"
            echo "Use './train.sh stop' to stop it first"
            exit 1
        fi
        
        # Make scripts executable
        chmod +x run_training.sh
        chmod +x setup_env.sh
        
        # Start with pm2
        pm2 start ecosystem.config.js
        
        print_success "Training started!"
        print_info "Monitor with: ./train.sh logs"
        print_info "Check status: ./train.sh status"
        ;;
        
    stop)
        print_header "Stopping Training"
        pm2 stop lpr-training
        print_success "Training stopped!"
        ;;
        
    restart)
        print_header "Restarting Training"
        pm2 restart lpr-training
        print_success "Training restarted!"
        ;;
        
    delete)
        print_header "Deleting Training Process"
        pm2 delete lpr-training
        print_success "Training process deleted!"
        ;;
        
    logs)
        print_header "Showing Training Logs (Ctrl+C to exit)"
        pm2 logs lpr-training --lines 100
        ;;
        
    status)
        print_header "Training Status"
        pm2 list | grep -E "lpr-training|App name"
        echo ""
        pm2 describe lpr-training 2>/dev/null || echo "Process not found"
        ;;
        
    monitor)
        print_header "PM2 Monitor (Ctrl+C to exit)"
        pm2 monit
        ;;
        
    info)
        print_header "Training Information"
        echo "📁 Working Directory: $SCRIPT_DIR"
        echo "📊 Dataset: train/"
        echo "🖥️  GPU: A30 (24GB)"
        echo "🐍 Conda Env: ocr"
        echo "📝 Logs: logs/"
        echo ""
        echo "Commands:"
        echo "  ./train.sh start    - Start training"
        echo "  ./train.sh stop     - Stop training"
        echo "  ./train.sh restart  - Restart training"
        echo "  ./train.sh logs     - View logs"
        echo "  ./train.sh status   - Check status"
        echo "  ./train.sh monitor  - PM2 monitor"
        echo "  ./train.sh delete   - Delete process"
        ;;
        
    *)
        print_header "License Plate Recognition Training Manager"
        echo ""
        echo "Usage: ./train.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start    - Start training with pm2"
        echo "  stop     - Stop training"
        echo "  restart  - Restart training"
        echo "  delete   - Delete pm2 process"
        echo "  logs     - View training logs (live)"
        echo "  status   - Check training status"
        echo "  monitor  - Open pm2 monitor"
        echo "  info     - Show training info"
        echo ""
        echo "Examples:"
        echo "  ./train.sh start    # Start training"
        echo "  ./train.sh logs     # Watch logs"
        echo "  ./train.sh status   # Check if running"
        ;;
esac
