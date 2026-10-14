import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
    parser.add_argument("--target_error", type=float, default=0.10, help="Target error rate for abstain policy (e.g. 0.10)")
    return parser.parse_args()

def nll_loss(temperature, logits, labels):
    T = temperature[0]
    loss_fct = nn.CrossEntropyLoss()
    # Apply temperature
    scaled_logits = logits / T
    return loss_fct(scaled_logits, labels).item()

def calibrate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Splits
    with open(Config.SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    fold_data = splits[args.fold]
    
    # Validation/Calibration Set
    manifest_df = pd.read_csv(Config.MANIFEST_PATH)
    calib_df = manifest_df.loc[fold_data['calibration']].reset_index(drop=True)
    
    # Dataset
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    calib_ds = EmotionDataset(calib_df, tokenizer=tokenizer, mode='val')
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False)
    
    # Load Model
    model_path = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"fold_{args.fold}_seed_{args.seed}", "best_model.pt")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}")
    model = EmotionModel(Config.MODEL_NAME, num_classes=len(Config.EMOTIONS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Inference
    all_logits = []
    all_labels = []
    
    print("Running inference on calibration set...")
    with torch.no_grad():
        for batch in tqdm(calib_loader):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            
            logits, _ = model(
                input_values=input_values,
                attention_mask=attention_mask,
                input_ids=input_ids,
                text_attention_mask=text_attention_mask
            )
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Temperature Scaling
    # Find T that minimizes NLL
    print("Optimizing Temperature...")
    init_T = np.array([1.5])
    res = minimize(nll_loss, init_T, args=(all_logits, all_labels), method='L-BFGS-B', bounds=[(0.01, 10.0)])
    T_opt = res.x[0]
    print(f"Optimal Temperature: {T_opt:.4f}")
    
    # Apply Temperature
    scaled_logits = all_logits / T_opt
    probs = torch.softmax(scaled_logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    
    # Accuracy Check
    correct = (predictions == all_labels).float()
    acc = correct.mean().item()
    print(f"Calibration Set Accuracy: {acc:.4f}")
    
    # Find Threshold for Abstention
    # We want error rate < target_error among accepted samples.
    # Error rate = 1 - Accuracy
    
    # Sort by confidence
    sorted_indices = torch.argsort(confidences) # ascending
    
    best_tau = 0.0
    best_coverage = 0.0
    
    # Sweep thresholds
    thresholds = torch.linspace(0, 1, 100)
    
    for tau in thresholds:
        accepted_mask = confidences >= tau
        if accepted_mask.sum() == 0:
            continue
            
        accepted_acc = correct[accepted_mask].mean().item()
        accepted_error = 1.0 - accepted_acc
        coverage = accepted_mask.float().mean().item()
        
        if accepted_error <= args.target_error:
            if coverage > best_coverage:
                best_coverage = coverage
                best_tau = tau.item()
                
    print(f"Selected Threshold (tau): {best_tau:.4f}")
    print(f"Expected Coverage: {best_coverage:.4f} at Error <= {args.target_error}")
    
    # Save Calibration Params
    calib_data = {
        "temperature": T_opt,
        "threshold": best_tau,
        "fold": args.fold,
        "seed": args.seed
    }
    
    out_file = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"fold_{args.fold}_seed_{args.seed}", "calibration.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(calib_data, f)
        
    print(f"Saved calibration data to {out_file}")

if __name__ == "__main__":
    args = parse_args()
    calibrate(args)
