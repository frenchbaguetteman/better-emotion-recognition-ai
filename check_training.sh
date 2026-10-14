#!/usr/bin/env bash
# TRAINING MONITOR - Check training status anytime
# Usage: bash check_training.sh [fold]

FOLD=${1:-0}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/training_logs"
TRAINING_LOG="${LOG_DIR}/training_fold_${FOLD}.log"
GPU_LOG="${LOG_DIR}/gpu_status_fold_${FOLD}.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC} TRAINING STATUS - Fold $FOLD"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}\n"

# Check if logs exist
if [ ! -f "${TRAINING_LOG}" ]; then
    echo -e "${RED}✗ No training log found${NC}"
    echo "  Expected: ${TRAINING_LOG}"
    exit 1
fi

# Check process status
PID_FILE="${LOG_DIR}/.training_pid_${FOLD}"
if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${GREEN}✓ Training Running${NC}"
        echo "  PID: $PID"
        echo "  Status: Active"
    else
        echo -e "${YELLOW}⚠ Training Not Running${NC}"
        echo "  PID: $PID (process ended)"
        
        # Check if completed
        if grep -q "Training finished" "${TRAINING_LOG}" 2>/dev/null; then
            echo -e "${GREEN}✓ Training Completed Successfully${NC}"
        else
            echo -e "${RED}✗ Training Failed${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ No PID file found${NC}"
fi

echo ""
echo -e "${BLUE}Last Training Update:${NC}"
tail -10 "${TRAINING_LOG}" | sed 's/^/  /'

echo ""
echo -e "${BLUE}Latest GPU Status:${NC}"
if [ -f "${GPU_LOG}" ]; then
    tail -5 "${GPU_LOG}" | sed 's/^/  /'
else
    echo "  (GPU monitoring not started)"
fi

echo ""
echo -e "${BLUE}Commands to Check Progress:${NC}"
echo "  Monitor training:  tail -f ${TRAINING_LOG}"
echo "  Monitor GPU:       tail -f ${GPU_LOG}"
echo "  Watch both:        watch -n 1 'tail -20 ${GPU_LOG}'"
echo "  Search for epoch:  grep 'Epoch' ${TRAINING_LOG} | tail"
echo "  Search for errors: grep -i error ${TRAINING_LOG}"

echo ""
