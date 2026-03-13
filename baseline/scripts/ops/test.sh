#!/bin/bash

# License Plate Recognition - Test/Evaluation Script
# This script evaluates the trained model on validation or test dataset

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASELINE_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$BASELINE_DIR")"

# Default values
MODEL_PATH="$BASELINE_DIR/best_model.pth"
MODE="val"
BATCH_SIZE=""
SAVE_RESULTS=false
SHOW_ERRORS=10

# Function to print usage
usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  ./test.sh [OPTIONS]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  --mode {val|test}           Evaluation mode (default: val)"
    echo "  --model_path PATH           Path to model checkpoint (default: best_model.pth)"
    echo "  --batch_size SIZE           Batch size (default: from config)"
    echo "  --save_results              Save results to JSON file"
    echo "  --show_errors N             Show N error cases (default: 10)"
    echo "  --help                      Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  ./test.sh                                    # Evaluate on val set"
    echo "  ./test.sh --mode test --save_results         # Evaluate on test set and save results"
    echo "  ./test.sh --batch_size 256 --show_errors 20  # Custom batch size and errors"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --save_results)
            SAVE_RESULTS=true
            shift
            ;;
        --show_errors)
            SHOW_ERRORS="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "val" && "$MODE" != "test" ]]; then
    echo -e "${RED}Error: Invalid mode. Must be 'val' or 'test'${NC}"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Error: Model file not found at $MODEL_PATH${NC}"
    echo "Please train the model first or provide correct path with --model_path"
    exit 1
fi

# Print configuration
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       License Plate Recognition - Evaluation Script         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Model Path:     $MODEL_PATH"
echo "  Mode:           $MODE"
echo "  Batch Size:     ${BATCH_SIZE:-default}"
echo "  Save Results:   $SAVE_RESULTS"
echo "  Show Errors:    $SHOW_ERRORS"
echo ""

# Build Python command
PYTHON_CMD="cd '$BASELINE_DIR' && python test.py"
PYTHON_CMD="$PYTHON_CMD --mode $MODE"
PYTHON_CMD="$PYTHON_CMD --model_path '$MODEL_PATH'"
PYTHON_CMD="$PYTHON_CMD --show_errors $SHOW_ERRORS"

if [ -n "$BATCH_SIZE" ]; then
    PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
fi

if [ "$SAVE_RESULTS" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --save_results"
fi

# Run evaluation
echo -e "${GREEN}Starting evaluation...${NC}"
echo ""

eval "$PYTHON_CMD"

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Evaluation completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Evaluation failed with error code $RESULT${NC}"
    exit $RESULT
fi
