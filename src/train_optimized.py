#!/usr/bin/env python3
"""
Optimized Training Script with GPU Utilization Fixes
Run this instead of train.py to get max GPU usage
"""

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import f1_score, mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import time

from config import Config
from dataset import EmotionDataset
from model import EmotionModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold index to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size - increase for better GPU utilization")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--vad_lambda", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class-weighted loss")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--gradient_checkpoint", action="store_true", help="Enable gradient checkpointing to save memory")
    return parser.parse_args()

def setup_gpu_optimizations():
    """Apply all GPU optimizations"""
    if torch.cuda.is_available():
        print("🚀 Setting up GPU optimizations...")
        
        # TF32 acceleration
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Memory management
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4294967296'
        
        # NCCL settings
        os.environ['NCCL_DEBUG'] = 'INFO'
        
        print("✓ TF32 enabled")
        print("✓ cuDNN benchmark enabled")
        print("✓ GPU memory optimization enabled")
        
        # Print GPU info
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"✓ GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")

def train(args):
    # Setup GPU
    setup_gpu_optimizations()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n📍 Using device: {device}\n")
    
    # Load Splits
    with open(Config.SPLITS_PATH, 'r') as f:
        splits = json.load(f)
        
    fold_data = splits[args.fold]
    
    # Load Manifest
    manifest_df = pd.read_csv(Config.MANIFEST_PATH)
    
    # NEW: Fix paths for VM environment
    def fix_vm_path(path):
        if not isinstance(path, str): return path
        if "Emotion_Balanced" in path:
            rel_path = path.split("Emotion_Balanced", 1)[1].lstrip("/")
            # Priority 1: /home/jl_fs/Emotion_Balanced (Fast Volume)
            p1 = os.path.join("/home/jl_fs/Emotion_Balanced", rel_path)
            if os.path.exists(p1): return p1
            # Priority 2: Local ./Emotion_Balanced
            p2 = os.path.join(os.getcwd(), "Emotion_Balanced", rel_path)
            if os.path.exists(p2): return p2
            # Priority 3: Absolute app path
            p3 = os.path.join("/home/better-emotion-recognition-ai/Emotion_Balanced", rel_path)
            if os.path.exists(p3): return p3
        return path

    print("🔧 Adjusting dataset paths for VM...")
    manifest_df['path'] = manifest_df['path'].apply(fix_vm_path)
    
    # Create DataFrames
    train_df = manifest_df.loc[fold_data['train']].reset_index(drop=True)
    val_df = manifest_df.loc[fold_data['val']].reset_index(drop=True)
    
    print(f"Fold {args.fold}: Train size {len(train_df)}, Val size {len(val_df)}")
    
    # Load Tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    
    # Datasets with caching enabled
    print("📦 Creating datasets with audio caching...")
    train_ds = EmotionDataset(train_df, tokenizer=tokenizer, mode='train', cache_audio=True)
    val_ds = EmotionDataset(val_df, tokenizer=tokenizer, mode='val', cache_audio=True)
    
    # Optimized DataLoaders
    print("📦 Creating dataloaders with optimized settings...")
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced from 8
        pin_memory=True,  # Speed up GPU transfer
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=4  # Prefetch batches
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # Model
    print(f"📦 Loading model {Config.MODEL_NAME}...")
    model = EmotionModel(Config.MODEL_NAME, num_classes=len(Config.EMOTIONS))
    
    # Optional: Gradient checkpointing to save memory
    if args.gradient_checkpoint and hasattr(model.backbone, 'gradient_checkpointing'):
        model.backbone.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    model.to(device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    num_training_steps = len(train_loader) * args.epochs // args.accumulation_steps
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    # Loss functions
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
    
    class_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    reg_criterion = nn.MSELoss()
    
    # Mixed Precision Setup
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_macro_f1 = 0.0
    patience = args.patience
    patience_counter = 0
    
    output_dir = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"fold_{args.fold}_seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Batch Size: {args.batch_size}")
    print(f"Accumulation Steps: {args.accumulation_steps}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"Mixed Precision: Enabled")
    print(f"Gradient Checkpointing: {'Enabled' if args.gradient_checkpoint else 'Disabled'}")
    print("=" * 70 + "\n")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            step_start = time.time()
            
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            vad_targets = batch['vad'].to(device)
            input_ids = batch['input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, vad_preds = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    input_ids=input_ids,
                    text_attention_mask=text_attention_mask
                )
                
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
            
            step_time = time.time() - step_start
            progress_bar.set_postfix({
                'loss': f"{loss.item() * args.accumulation_steps:.4f}",
                'time': f"{step_time:.2f}s"
            })
            
            # GPU memory monitoring (every 10 steps)
            if (step + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                # print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        
        # Validation
        print("🔍 Running validation...")
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_vad_preds = []
        all_vad_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_values = batch['input_values'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                vad_targets = batch['vad'].to(device)
                input_ids = batch['input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits, vad_preds = model(
                        input_values=input_values,
                        attention_mask=attention_mask,
                        input_ids=input_ids,
                        text_attention_mask=text_attention_mask
                    )
                
                # Metrics
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_vad_preds.extend(vad_preds.cpu().numpy())
                all_vad_targets.extend(vad_targets.cpu().numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        vad_rmse = np.sqrt(mean_squared_error(all_vad_targets, all_vad_preds))
        
        print(f"\n✓ Epoch {epoch+1}: Val Macro-F1 = {macro_f1:.4f}, VAD RMSE = {vad_rmse:.4f}\n")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print("💾 Saved Best Model")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience {patience_counter}/{patience}")

        # Save latest model
        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pt"))
        
        if patience_counter >= patience:
            print("🛑 Early Stopping Triggered.")
            break
    
    print("\n" + "=" * 70)
    print("✅ Training finished!")
    print("=" * 70)

if __name__ == "__main__":
    args = parse_args()
    train(args)
