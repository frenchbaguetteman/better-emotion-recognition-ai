#!/bin/bash

# Run 5-Fold Cross Validation Ensemble Training
# This will train 5 models, one for each fold.

echo "Starting Ensemble Training..."
export PYTHONPATH=$PYTHONPATH:.

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

echo "Ensemble training complete. Data extraction and ensembling of predictions usually happens in inference step."
