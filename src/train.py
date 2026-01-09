import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from config import Config
from dataset import EmotionDataset
from model import EmotionModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold index to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8) # Small BS for large models
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--vad_lambda", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    return parser.parse_args()

def train(args):
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Splits
    with open(Config.SPLITS_PATH, 'r') as f:
        splits = json.load(f)
        
    fold_data = splits[args.fold]
    
    # Load Manifest
    manifest_df = pd.read_csv(Config.MANIFEST_PATH)
    
    # Create DataFrames
    train_df = manifest_df.loc[fold_data['train']].reset_index(drop=True)
    val_df = manifest_df.loc[fold_data['val']].reset_index(drop=True)
    # Calib set is ignored during training loop
    
    print(f"Fold {args.fold}: Train size {len(train_df)}, Val size {len(val_df)}")
    
    # Datasets
    train_ds = EmotionDataset(train_df, mode='train')
    val_ds = EmotionDataset(val_df, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    print(f"Loading model {Config.MODEL_NAME}...")
    model = EmotionModel(Config.MODEL_NAME, num_classes=len(Config.EMOTIONS))
    model.to(device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    num_training_steps = len(train_loader) * args.epochs // args.accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    
    best_macro_f1 = 0.0
    output_dir = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"fold_{args.fold}_seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            vad_targets = batch['vad'].to(device)
            
            logits, vad_preds = model(input_values)
            
            loss_cls = class_criterion(logits, labels)
            loss_vad = reg_criterion(vad_preds, vad_targets)
            
            loss = loss_cls + args.vad_lambda * loss_vad
            loss = loss / args.accumulation_steps
            loss.backward()
            
            if (step + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * args.accumulation_steps})
            
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_vad_preds = []
        all_vad_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch['input_values'].to(device)
                labels = batch['labels'].to(device)
                vad_targets = batch['vad'].to(device)
                
                logits, vad_preds = model(input_values)
                
                # Metrics
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_vad_preds.extend(vad_preds.cpu().numpy())
                all_vad_targets.extend(vad_targets.cpu().numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        vad_rmse = np.sqrt(mean_squared_error(all_vad_targets, all_vad_preds))
        
        print(f"Epoch {epoch+1}: Val Macro-F1 = {macro_f1:.4f}, VAD RMSE = {vad_rmse:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print("Saved Best Model")

        # Save latest model refreshing every epoch
        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pt"))
        print(f"Refreshed checkpoint at epoch {epoch+1}")
            
            # Save logits for Ensembling / Analysis?
            # Actually we need logits on calibration set, not val_optim set.
            # But saving metrics is good.
    
    print("Training finished.")

if __name__ == "__main__":
    args = parse_args()
    train(args)
