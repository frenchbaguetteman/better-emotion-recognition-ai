import torch
import numpy as np
import os
import argparse
import pandas as pd
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, mean_squared_error
from tqdm import tqdm

from config import Config
from model import EmotionModel
from dataset import EmotionDataset

def train(args):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Metal Performance Shaders
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load manifest and splits
    manifest_df = pd.read_csv(Config.MANIFEST_PATH)
    with open(Config.SPLITS_PATH, 'r') as f:
        splits_list = json.load(f)
    
    # splits_list is a list of dicts, one per fold
    fold_split = splits_list[args.fold]
    train_indices = fold_split['train']
    val_indices = fold_split['val']
    
    train_df = manifest_df.iloc[train_indices].reset_index(drop=True)
    val_df = manifest_df.iloc[val_indices].reset_index(drop=True)
    
    # Tokenizer (not used in audio-only, but keep for dataset compatibility)
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    
    # Create datasets - audio only
    train_dataset = EmotionDataset(train_df, tokenizer=tokenizer, mode='train')
    val_dataset = EmotionDataset(val_df, tokenizer=tokenizer, mode='val')
    
    print(f"Fold {args.fold}: Train size {len(train_dataset)}, Val size {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Model (audio only)
    print("Loading model microsoft/wavlm-large (audio only)...")
    model = EmotionModel(audio_model_name="microsoft/wavlm-large", use_text=False)
    model = model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    class_weights = None
    if not args.no_class_weights:
        label_ids = train_df['emotion_label'].map(Config.EMOTION2ID).to_numpy()
        counts = np.bincount(label_ids, minlength=len(Config.EMOTIONS))
        weights = np.zeros(len(Config.EMOTIONS), dtype=np.float32)
        non_zero = counts > 0
        weights[non_zero] = counts.sum() / (counts[non_zero] * len(Config.EMOTIONS))
        if weights[non_zero].size > 0:
            weights[non_zero] = weights[non_zero] / weights[non_zero].mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Class weights: {weights}")
    
    class_criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    reg_criterion = torch.nn.MSELoss()
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_macro_f1 = 0.0
    patience = args.patience
    patience_counter = 0
    
    output_dir = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"fold_{args.fold}_audio_only")
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            vad_targets = batch['vad'].to(device)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, vad_preds = model(input_values=input_values, attention_mask=attention_mask)
                
                loss_cls = class_criterion(logits, labels)
                loss_vad = reg_criterion(vad_preds, vad_targets)
                loss = loss_cls + args.vad_lambda * loss_vad
                loss = loss / args.accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * args.accumulation_steps})
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        all_vad_preds = []
        all_vad_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch['input_values'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                vad_targets = batch['vad'].to(device)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits, vad_preds = model(input_values=input_values, attention_mask=attention_mask)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_vad_preds.extend(vad_preds.cpu().numpy())
                all_vad_targets.extend(vad_targets.cpu().numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        vad_rmse = np.sqrt(mean_squared_error(all_vad_targets, all_vad_preds))
        
        print(f"Epoch {epoch+1}: Val Macro-F1 = {macro_f1:.4f}, VAD RMSE = {vad_rmse:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model_audio_only.pt"))
            print("Saved Best Model")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")
        
        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pt"))
        
        if patience_counter >= patience:
            print("Early Stopping Triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--vad_lambda", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class-weighted loss")
    
    args = parser.parse_args()
    train(args)
