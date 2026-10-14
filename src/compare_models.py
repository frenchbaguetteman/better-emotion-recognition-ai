"""
Compare audio-only vs audio+text models across emotions.
Filters results to only include the 4 emotions: Angry, Happy, Neutral, Sad
"""

import torch
import pandas as pd
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import argparse
from tqdm import tqdm

from config import Config
from model import EmotionModel
from dataset import EmotionDataset

def load_model(checkpoint_path, use_text=True, device="cuda"):
    """Load a trained model from checkpoint."""
    model = EmotionModel(audio_model_name="microsoft/wavlm-large", use_text=use_text)
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️  Model not found at {checkpoint_path}")
        return None
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device, use_text=False):
    """Evaluate model and return per-emotion metrics."""
    all_preds = []
    all_labels = []
    all_emotions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating ({'text' if use_text else 'audio'})"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Prepare text inputs if using text model
            if use_text and 'input_ids' in batch:
                input_ids = batch['input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                logits, _ = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    input_ids=input_ids,
                    text_attention_mask=text_attention_mask
                )
            else:
                logits, _ = model(input_values=input_values, attention_mask=attention_mask)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get emotion names for context
            if 'emotion_label' in batch:
                all_emotions.extend(batch['emotion_label'])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds, all_labels, all_emotions

def compute_metrics(preds, labels, emotion_names):
    """Compute overall and per-emotion metrics."""
    # Overall metrics
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # Per-emotion metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    emotion_metrics = {}
    for i, emotion in enumerate(Config.EMOTIONS):
        emotion_metrics[emotion] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        }
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'per_emotion': emotion_metrics,
        'confusion_matrix': confusion_matrix(labels, preds).tolist()
    }

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test data
    print("\nLoading test dataset...")
    manifest_df = pd.read_csv(Config.MANIFEST_PATH)
    
    # Filter to only 4 emotions
    manifest_df = manifest_df[manifest_df['emotion_label'].isin(Config.EMOTIONS)]
    print(f"Filtered to {len(manifest_df)} samples with 4 emotions: {Config.EMOTIONS}")
    
    # Shuffle manifest to get mixed emotions
    manifest_df = manifest_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # Load splits to find dedicated test/val indices if possible
    with open(Config.SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    
    # Actually just take a random sample of 500 from the whole filtered manifest 
    # since we want a quick but balanced comparison
    test_df = manifest_df.head(500).reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    test_dataset = EmotionDataset(test_df, tokenizer=tokenizer, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Test set: {len(test_df)} samples")
    print(f"Emotion distribution:\n{test_df['emotion_label'].value_counts()}\n")
    
    # Load models
    audio_only_checkpoint = os.path.join(
        Config.PROJECT_ROOT, "final_models", "best_model_audio_only.pt"
    )
    audio_text_checkpoint = os.path.join(
        Config.PROJECT_ROOT, "final_models", "best_model_wavlm_A100.pt"
    )
    
    audio_model = load_model(audio_only_checkpoint, use_text=False, device=device)
    text_model = load_model(audio_text_checkpoint, use_text=True, device=device)
    
    if audio_model is None or text_model is None:
        print("❌ Failed to load models")
        return
    
    # Evaluate both models
    print("\n" + "="*60)
    print("AUDIO-ONLY MODEL")
    print("="*60)
    audio_preds, labels, emotions = evaluate_model(audio_model, test_loader, device, use_text=False)
    audio_metrics = compute_metrics(audio_preds, labels, emotions)
    
    print("\n" + "="*60)
    print("AUDIO+TEXT MODEL")
    print("="*60)
    text_preds, _, _ = evaluate_model(text_model, test_loader, device, use_text=True)
    text_metrics = compute_metrics(text_preds, labels, emotions)
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON: TEXT-BASED vs AUDIO-ONLY")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Audio-Only':<15} {'Audio+Text':<15} {'Improvement':<15}")
    print("-" * 65)
    
    audio_macro = audio_metrics['macro_f1']
    text_macro = text_metrics['macro_f1']
    improvement = text_macro - audio_macro
    improvement_pct = (improvement / audio_macro * 100) if audio_macro > 0 else 0
    
    print(f"{'Macro F1':<20} {audio_macro:<15.4f} {text_macro:<15.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
    print(f"{'Micro F1':<20} {audio_metrics['micro_f1']:<15.4f} {text_metrics['micro_f1']:<15.4f}")
    print(f"{'Weighted F1':<20} {audio_metrics['weighted_f1']:<15.4f} {text_metrics['weighted_f1']:<15.4f}")
    
    # Per-emotion comparison
    print("\n" + "-"*65)
    print("PER-EMOTION BREAKDOWN")
    print("-"*65)
    print(f"{'Emotion':<15} {'Audio F1':<15} {'Text F1':<15} {'Improvement':<15}")
    print("-"*65)
    
    for emotion in Config.EMOTIONS:
        audio_f1 = audio_metrics['per_emotion'][emotion]['f1']
        text_f1 = text_metrics['per_emotion'][emotion]['f1']
        improvement = text_f1 - audio_f1
        
        print(f"{emotion:<15} {audio_f1:<15.4f} {text_f1:<15.4f} {improvement:+.4f}")
    
    # Save detailed results
    results = {
        'audio_only': audio_metrics,
        'audio_text': text_metrics,
        'comparison': {
            'macro_f1_improvement': text_macro - audio_macro,
            'macro_f1_improvement_pct': improvement_pct
        }
    }
    
    output_file = os.path.join(Config.OUTPUT_DIR, "model_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)
