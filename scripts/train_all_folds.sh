#!/bin/bash

# Train one model per speaker-independent fold (5-fold cross-validation).

cd "$(dirname "$0")/.."
echo "Starting 5-fold training..."

for fold in {0..4}
do
    echo "=================================================="
    echo "Training Fold $fold"
    echo "=================================================="
    
    python src/train.py \
        --fold $fold \
        --epochs 15 \
        --batch_size 16 \
        --accumulation_steps 2 \
        --lr 2e-5 \
        --vad_lambda 1.0
        
    echo "Finished Fold $fold"
done

echo "All folds complete. Run src/calibrate.py, src/evaluate.py and src/report.py next."
