import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
from dataset import EmotionDataset
from model import EmotionModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()

def compute_calibration_metrics(probs, labels, n_bins=10):
    # ECE
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Splits
    with open(Config.SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    fold_data = splits[args.fold]
    
    # Report on all held-out speakers for this fold (val + calibration). Note that val was
    # used for early stopping and calibration for temperature scaling; there is no untouched test set.
    
    manifest_df = pd.read_csv(Config.MANIFEST_PATH)
    test_indices = fold_data['val'] + fold_data['calibration']
    test_df = manifest_df.loc[test_indices].reset_index(drop=True)
    
    print(f"Evaluating on {len(test_df)} samples (Fold {args.fold} Validation Speakers)")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    ds = EmotionDataset(test_df, tokenizer=tokenizer, mode='val')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # Load Model
    model_path = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"fold_{args.fold}_seed_{args.seed}", "best_model.pt")
    model = EmotionModel(Config.MODEL_NAME, num_classes=len(Config.EMOTIONS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Calibration
    calib_path = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"fold_{args.fold}_seed_{args.seed}", "calibration.pkl")
    if os.path.exists(calib_path):
        with open(calib_path, 'rb') as f:
            calib = pickle.load(f)
        T = calib['temperature']
    else:
        T = 1.0
        
    all_logits = []
    all_labels = []
    all_vad_preds = []
    all_vad_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            vad_targets = batch['vad'].to(device)
            
            input_ids = batch['input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            
            logits, vad = model(
                input_values=input_values,
                attention_mask=attention_mask,
                input_ids=input_ids,
                text_attention_mask=text_attention_mask
            )
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_vad_preds.append(vad.cpu())
            all_vad_targets.append(vad_targets.cpu())
            
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_vad_preds = torch.cat(all_vad_preds)
    all_vad_targets = torch.cat(all_vad_targets)
    
    # Classification Metrics
    probs = torch.softmax(all_logits / T, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average='macro')
    cm = confusion_matrix(all_labels, preds)
    
    # Regression Metrics
    # Output: valence, arousal, dominance
    vad_rmse = np.sqrt(mean_squared_error(all_vad_targets, all_vad_preds, multioutput='raw_values'))
    vad_mae = mean_absolute_error(all_vad_targets, all_vad_preds, multioutput='raw_values')
    
    pearsons = []
    for i in range(3):
        p, _ = pearsonr(all_vad_targets[:, i], all_vad_preds[:, i])
        pearsons.append(p)
        
    # Calibration Metrics
    ece = compute_calibration_metrics(probs, all_labels)
    nll = nn.CrossEntropyLoss()(all_logits / T, all_labels).item()
    
    # Saving Results
    results = {
        "fold": args.fold,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "vad_rmse": vad_rmse.tolist(),
        "vad_mae": vad_mae.tolist(),
        "vad_pearson": pearsons,
        "ece": ece,
        "nll": nll,
        "temperature": T
    }
    
    print(json.dumps(results, indent=4))
    
    out_file = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"fold_{args.fold}_seed_{args.seed}", "results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    import pandas as pd
    args = parse_args()
    evaluate(args)
