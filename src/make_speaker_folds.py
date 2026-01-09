import pandas as pd
import json
import os
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from config import Config
import numpy as np

def make_folds(n_folds=5):
    print(f"Creating {n_folds} speaker-independent folds...")
    manifest_df = pd.read_csv(Config.MANIFEST_PATH)
    
    # We need an integer ID for grouping if speakers are strings
    # But GroupKFold accepts strings usually.
    
    groups = manifest_df['speaker_id']
    X = manifest_df['path'] # dummy
    y = manifest_df['emotion_label'] # dummy
    
    gkf = GroupKFold(n_splits=n_folds)
    
    splits = []
    
    # Fold indices
    for fold_idx, (train_idx, val_full_idx) in enumerate(gkf.split(X, y, groups)):
        # train_idx are indices of the main training set for this fold
        
        # Now we need to split val_full_idx into 'val_optim' (for early stopping) and 'calibration'
        # We must respect speaker independence here too.
        
        val_full_df = manifest_df.iloc[val_full_idx]
        val_groups = val_full_df['speaker_id']
        
        # Use GroupShuffleSplit to split validation speakers 50/50
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        
        # We need relative indices within val_full_df to map back to global indices
        val_inner_idx, calib_inner_idx = next(gss.split(val_full_df, groups=val_groups))
        
        val_optim_global_idx = val_full_df.iloc[val_inner_idx].index.tolist()
        calib_global_idx = val_full_df.iloc[calib_inner_idx].index.tolist()
        train_global_idx = train_idx.tolist()
        
        fold_data = {
            "fold": fold_idx,
            "train": train_global_idx,
            "val": val_optim_global_idx,
            "calibration": calib_global_idx
        }
        splits.append(fold_data)
        
        print(f"Fold {fold_idx}: Train={len(train_global_idx)}, Val={len(val_optim_global_idx)}, Calib={len(calib_global_idx)}")
        
    with open(Config.SPLITS_PATH, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Saved splits to {Config.SPLITS_PATH}")

if __name__ == "__main__":
    make_folds()
