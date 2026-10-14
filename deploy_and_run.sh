#!/usr/bin/env bash
# INSTANT DEPLOYMENT - Deploy and run training in one command
# Usage: bash deploy_and_run.sh [fold] [batch_size] [seed]

# Override any of these via environment variables, e.g.
#   JARVISLABS_PORT=12345 bash deploy_and_run.sh 0 16 42
JARVISLABS_HOST="${JARVISLABS_HOST:-sshd.jarvislabs.ai}"
JARVISLABS_PORT="${JARVISLABS_PORT:?Set JARVISLABS_PORT to your instance SSH port}"
JARVISLABS_USER="${JARVISLABS_USER:-root}"
PROJECT_PATH="${PROJECT_PATH:-/home/better-emotion-recognition-ai}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/jarvislabs_key}"

FOLD=${1:-0}
BATCH_SIZE=${2:-16}
SEED=${3:-42}

HOST="${JARVISLABS_USER}@${JARVISLABS_HOST}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              INSTANT TRAINING DEPLOYMENT (RSYNC)              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Host: $HOST:$JARVISLABS_PORT"
echo "  Key: $SSH_KEY"
echo "  Remote Path: $PROJECT_PATH"
echo "  Fold: $FOLD"
echo "  Batch Size: $BATCH_SIZE"
echo "  Seed: $SEED"
echo ""

# Check SSH connection
echo "Testing SSH connection..."
ssh -q -o ConnectTimeout=10 -i "$SSH_KEY" -p "$JARVISLABS_PORT" "$HOST" "exit 0"
if [ $? -ne 0 ]; then
    echo "⚠ SSH check failed. Retrying..."
    sleep 2
    ssh -q -o ConnectTimeout=20 -i "$SSH_KEY" -p "$JARVISLABS_PORT" "$HOST" "exit 0"
    if [ $? -ne 0 ]; then
         echo "✗ SSH connection failed."
         exit 1
    fi
else
    echo "✓ SSH connection OK"
fi

echo ""
echo "Step 1: Syncing code via rsync..."
# Sync everything EXCEPT dataset and checkpoints
# We want to sync: src/, data/, *.sh, *.py, *.txt
rsync -avz -e "ssh -p $JARVISLABS_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
    --exclude 'Emotion_Balanced/' \
    --exclude 'checkpoints/' \
    --exclude 'checkpoints_local/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '.venv/' \
    --exclude '*.pyc' \
    --exclude '*.tar.gz' \
    ./ "$HOST:$PROJECT_PATH/"

if [ $? -eq 0 ]; then
    echo "✓ Code synced successfully."
else
    echo "✗ Rsync failed. Trying fallback to SCP..."
    tar --exclude='checkpoints' --exclude='Emotion_Balanced' --exclude='.git' -czf deploy_fallback.tar.gz .
    scp -P "$JARVISLABS_PORT" -i "$SSH_KEY" deploy_fallback.tar.gz "$HOST:$PROJECT_PATH/"
    ssh -p "$JARVISLABS_PORT" -i "$SSH_KEY" "$HOST" "cd $PROJECT_PATH && tar -xzf deploy_fallback.tar.gz && rm deploy_fallback.tar.gz"
fi

echo ""
echo "Step 2: Starting training on VM..."
echo ""

# Execute on remote
ssh -p "$JARVISLABS_PORT" -i "$SSH_KEY" "$HOST" "
    cd $PROJECT_PATH || { echo 'Directory not found'; exit 1; }
    
    echo 'Installing dependencies...'
    pip install -q -r requirements.txt
    
    chmod +x run_training.sh check_training.sh
    
    # Run!
    bash run_training.sh $FOLD $BATCH_SIZE $SEED
"
