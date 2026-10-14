#!/usr/bin/env bash
# MASTER TRAINING LAUNCHER - Run this on your VM
# This script handles everything: setup, training, monitoring, and background execution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FOLD=${1:-0}
BATCH_SIZE=${2:-16}
SEED=${3:-42}
EPOCHS=${4:-10}
LR=${5:-1e-5}

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/training_logs"
GPU_LOG="${LOG_DIR}/gpu_status_fold_${FOLD}.log"
TRAINING_LOG="${LOG_DIR}/training_fold_${FOLD}.log"
MONITOR_PID_FILE="${LOG_DIR}/.monitor_pid_${FOLD}"

# Create log directory
mkdir -p "${LOG_DIR}"

# KILL OLD TRAINING PROCESSES
TRAINING_PID_FILE="${LOG_DIR}/.training_pid_${FOLD}"
if [ -f "${TRAINING_PID_FILE}" ]; then
    OLD_TRAIN_PID=$(cat "${TRAINING_PID_FILE}")
    if kill -0 "$OLD_TRAIN_PID" 2>/dev/null; then
        echo "Stopping previous training run (PID: $OLD_TRAIN_PID)..."
        kill "$OLD_TRAIN_PID" || kill -9 "$OLD_TRAIN_PID" || true
    fi
fi
# Also kill any zombies matching the script name just in case
pkill -f "python.*train_optimized.py --fold $FOLD" || true
sleep 1

print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Step 1: Verification
print_header "STEP 1: Verifying Setup"

if ! command -v python &> /dev/null; then
    print_error "Python not found"
    exit 1
fi
print_success "Python found: $(python --version)"

if ! python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" &> /dev/null; then
    print_error "PyTorch/CUDA setup issue"
    exit 1
fi
print_success "PyTorch with CUDA verified"

if [ ! -f "${PROJECT_DIR}/src/train_optimized.py" ]; then
    print_error "train_optimized.py not found at ${PROJECT_DIR}/src/"
    exit 1
fi
print_success "Training script found"

# Step 2: GPU Diagnostics
print_header "STEP 2: GPU Diagnostics"

python "${PROJECT_DIR}/src/gpu_optimize.py" 2>&1 | head -30

# Step 3: Start GPU Monitoring
print_header "STEP 3: Starting GPU Monitoring"

# Kill any previous monitoring
if [ -f "${MONITOR_PID_FILE}" ]; then
    OLD_PID=$(cat "${MONITOR_PID_FILE}")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        kill "$OLD_PID" || true
        print_info "Stopped previous monitor (PID: $OLD_PID)"
    fi
fi

# Start new monitoring in background
python "${PROJECT_DIR}/src/gpu_monitor.py" --monitor > "${GPU_LOG}" 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > "${MONITOR_PID_FILE}"
print_success "GPU monitoring started (PID: $MONITOR_PID)"
print_info "GPU metrics will be logged to: ${GPU_LOG}"

# Step 4: Set Environment Variables
print_header "STEP 4: Setting Environment Variables"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4294967296
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

print_success "Environment configured"

# Step 5: Start Training
print_header "STEP 5: Starting Training"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Fold: $FOLD"
echo "  Batch Size: $BATCH_SIZE"
echo "  Seed: $SEED"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Project: ${PROJECT_DIR}"
echo "  GPU Log: ${GPU_LOG}"
echo "  Training Log: ${TRAINING_LOG}"

# Start training in background
print_info "Starting training in background..."
nohup python "${PROJECT_DIR}/src/train_optimized.py" \
    --fold "$FOLD" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    > "${TRAINING_LOG}" 2>&1 &

TRAINING_PID=$!
echo $TRAINING_PID > "${LOG_DIR}/.training_pid_${FOLD}"

print_success "Training started (PID: $TRAINING_PID)"
print_info "Training output: ${TRAINING_LOG}"

# Step 6: Initial Status Check
print_header "STEP 6: Training Status"

sleep 5

if kill -0 $TRAINING_PID 2>/dev/null; then
    print_success "Training process running"
    
    # Show first few lines of training output
    print_info "Initial training output:"
    head -20 "${TRAINING_LOG}" | sed 's/^/  /'
else
    print_error "Training process failed to start"
    print_error "Check log: ${TRAINING_LOG}"
    exit 1
fi

# Step 7: Final Instructions
print_header "✨ TRAINING RUNNING IN BACKGROUND"

echo -e "${GREEN}✓ Everything started successfully!${NC}\n"

echo -e "${YELLOW}To monitor training progress:${NC}"
echo "  tail -f ${TRAINING_LOG}"
echo ""
echo -e "${YELLOW}To monitor GPU in real-time:${NC}"
echo "  tail -f ${GPU_LOG}"
echo ""
echo -e "${YELLOW}To check both simultaneously:${NC}"
echo "  watch -n 1 'tail -20 ${GPU_LOG}'"
echo ""
echo -e "${YELLOW}To see process status:${NC}"
echo "  ps aux | grep train_optimized.py"
echo "  ps aux | grep gpu_monitor.py"
echo ""
echo -e "${YELLOW}To stop training manually:${NC}"
echo "  kill $TRAINING_PID"
echo ""
echo -e "${YELLOW}To stop monitoring:${NC}"
echo "  kill $MONITOR_PID"
echo ""
echo -e "${BLUE}Processes:${NC}"
echo "  Training PID:  $TRAINING_PID"
echo "  Monitor PID:   $MONITOR_PID"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Training will continue running even after you disconnect!${NC}"
echo -e "${GREEN}You can reconnect anytime to check progress.${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Keep script alive to show real-time GPU status
print_header "REAL-TIME GPU STATUS (Updating...)"

for i in {1..30}; do
    echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} - Iteration $i/30"
    
    if ! kill -0 $TRAINING_PID 2>/dev/null; then
        print_error "Training process stopped"
        
        # Check if it finished successfully
        if grep -q "Training finished" "${TRAINING_LOG}" 2>/dev/null; then
            print_success "Training completed successfully!"
        else
            print_error "Training failed. Check log: ${TRAINING_LOG}"
        fi
        break
    fi
    
    if [ -f "${GPU_LOG}" ]; then
        tail -3 "${GPU_LOG}" | sed 's/^/  /'
    fi
    
    sleep 2
done

print_header "SUMMARY"

echo -e "${BLUE}Training Status:${NC}"
if kill -0 $TRAINING_PID 2>/dev/null; then
    print_success "Training running (PID: $TRAINING_PID)"
    print_info "To monitor: tail -f ${TRAINING_LOG}"
else
    print_info "Training session completed"
fi

echo ""
echo -e "${BLUE}Log Files:${NC}"
echo "  Training: ${TRAINING_LOG}"
echo "  GPU:      ${GPU_LOG}"
echo ""
print_success "Setup complete!"
