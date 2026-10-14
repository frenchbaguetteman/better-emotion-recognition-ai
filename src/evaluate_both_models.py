"""
Evaluate Both Audio-Only and Audio+Text Models
"""

import torch
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from model import EmotionModel

# Core 4 emotions
CORE_EMOTIONS = ["Angry", "Happy", "Neutral", "Sad"]

# Dataset root: folder containing Ravdess/, Tess/, Crema/, Savee/ (override with --datasets_root)
DATASETS_ROOT = Path(os.environ.get("EMOTION_DATASETS_ROOT", os.path.join(Config.PROJECT_ROOT, "external_datasets")))

# Emotion mappings
EMOTION_MAPPINGS = {
    'RAVDESS': {
        '01': 'Neutral', '02': 'Calm', '03': 'Happy', '04': 'Sad',
        '05': 'Angry', '06': 'Fearful', '07': 'Disgusted', '08': 'Surprised'
    },
    'TESS': {
        'angry': 'Angry', 'disgust': 'Disgusted', 'fear': 'Fearful',
        'happy': 'Happy', 'neutral': 'Neutral', 'ps': 'Pleasant_surprise', 'sad': 'Sad'
    },
    'Crema': {
        'ANG': 'Angry', 'DIS': 'Disgusted', 'FEA': 'Fearful',
        'HAP': 'Happy', 'NEU': 'Neutral', 'SAD': 'Sad'
    },
    'Savee': {
        'a': 'Angry', 'd': 'Disgusted', 'f': 'Fearful',
        'h': 'Happy', 'n': 'Neutral', 'sa': 'Sad', 'su': 'Surprised'
    }
}

EMOTION_TO_4CORE = {
    'Angry': 'Angry', 'Happy': 'Happy', 'Neutral': 'Neutral', 'Sad': 'Sad',
    'Calm': 'Neutral', 'Fearful': 'Sad', 'Disgusted': 'Angry', 'Surprised': 'Happy'
}


class AudioTextDataset(Dataset):
    """Load audio files with Whisper transcriptions."""
    
    def __init__(self, file_list, tokenizer, whisper_model=None, sr=16000, duration=8.0):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.whisper_model = whisper_model
        self.sr = sr
        self.duration = duration
        self.target_len = int(sr * duration)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if isinstance(self.file_list, dict):
            # Handle dict-based access
            keys = list(self.file_list.keys())
            item_key = keys[idx]
            item = self.file_list[item_key]
        else:
            # Handle list-based access
            item = self.file_list[idx]
        
        path = item.get('path', item.get(0)) if isinstance(item, dict) else item['path']
        emotion = item.get('emotion', item.get(1)) if isinstance(item, dict) else item['emotion']
        dataset = item.get('dataset', item.get(2)) if isinstance(item, dict) else item['dataset']
        
        # Load audio
        try:
            audio_data, sr = sf.read(path)
            audio_data = audio_data.astype(np.float32)
            
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            if sr != self.sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sr)
            
            if len(audio_data) < self.target_len:
                audio_data = np.pad(audio_data, (0, self.target_len - len(audio_data)))
            else:
                start = (len(audio_data) - self.target_len) // 2
                audio_data = audio_data[start:start + self.target_len]
            
            waveform = torch.from_numpy(audio_data).float()
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            waveform = torch.zeros(self.target_len)
        
        # Transcribe with Whisper if available
        text = ""
        if self.whisper_model:
            try:
                result = self.whisper_model.transcribe(path, verbose=False, language='en', fp16=False)
                text = result.get('text', '').strip()
            except Exception as e:
                print(f"Error transcribing {path}: {e}")
                text = ""
        
        # Tokenize text
        if text:
            tokens = self.tokenizer(text, max_length=64, padding='max_length', 
                                   truncation=True, return_tensors='pt')
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(64, dtype=torch.long)
            attention_mask = torch.zeros(64, dtype=torch.long)
        
        return {
            'audio': waveform,
            'input_ids': input_ids,
            'text_attention_mask': attention_mask,
            'text': text,
            'emotion': emotion,
            'dataset': dataset,
            'path': path
        }


class DatasetLoader:
    """Load files from all datasets."""
    
    def get_ravdess_files(self):
        files = []
        root = DATASETS_ROOT / "Ravdess"
        if not root.exists():
            return files
        
        actors_dir = root / "audio_speech_actors_01-24"
        if not actors_dir.exists():
            return files
        
        for actor_dir in actors_dir.glob("Actor_*"):
            for wav_file in actor_dir.glob("*.wav"):
                parts = wav_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]  # Emotion is at index 2 in RAVDESS format
                    emotion_label = EMOTION_MAPPINGS['RAVDESS'].get(emotion_code)
                    if emotion_label and emotion_label in EMOTION_TO_4CORE:
                        files.append({
                            'path': str(wav_file),
                            'emotion': EMOTION_TO_4CORE[emotion_label],
                            'dataset': 'RAVDESS'
                        })
        return files
    
    def get_tess_files(self):
        files = []
        root = DATASETS_ROOT / "Tess"
        if not root.exists():
            return files
        
        for item in root.iterdir():
            if not item.is_dir():
                continue
            parts = item.name.split('_')
            if len(parts) >= 2:
                emotion_name = parts[-1].lower()
                emotion_label = EMOTION_MAPPINGS['TESS'].get(emotion_name)
                if emotion_label and emotion_label in EMOTION_TO_4CORE:
                    for wav_file in item.glob("*.wav"):
                        files.append({
                            'path': str(wav_file),
                            'emotion': EMOTION_TO_4CORE[emotion_label],
                            'dataset': 'TESS'
                        })
        return files
    
    def get_crema_files(self):
        files = []
        root = DATASETS_ROOT / "Crema"
        if not root.exists():
            return files
        
        for wav_file in root.glob("*.wav"):
            parts = wav_file.stem.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion_label = EMOTION_MAPPINGS['Crema'].get(emotion_code)
                if emotion_label and emotion_label in EMOTION_TO_4CORE:
                    files.append({
                        'path': str(wav_file),
                        'emotion': EMOTION_TO_4CORE[emotion_label],
                        'dataset': 'CREMA-D'
                    })
        return files
    
    def get_savee_files(self):
        files = []
        root = DATASETS_ROOT / "Savee"
        if not root.exists():
            return files
        
        for wav_file in root.glob("*.wav"):
            # SAVEE format: DC_EMOTION## where EMOTION can be a, d, f, h, n, sa, su
            stem = wav_file.stem  # e.g., "DC_sa01"
            emotion_part = stem.split('_')[1]  # e.g., "sa01"
            
            # Extract emotion code (remove trailing digits)
            emotion_code = emotion_part.rstrip('0123456789')  # e.g., "sa"
            
            emotion_label = EMOTION_MAPPINGS['Savee'].get(emotion_code)
            if emotion_label and emotion_label in EMOTION_TO_4CORE:
                files.append({
                    'path': str(wav_file),
                    'emotion': EMOTION_TO_4CORE[emotion_label],
                    'dataset': 'SAVEE'
                })
        return files
    
    def load_all(self, limit_per_emotion=None):
        all_files = []
        
        # Load by dataset
        ravdess_files = self.get_ravdess_files()
        tess_files = self.get_tess_files()
        crema_files = self.get_crema_files()
        savee_files = self.get_savee_files()
        
        print(f"Raw counts - RAVDESS: {len(ravdess_files)}, TESS: {len(tess_files)}, CREMA-D: {len(crema_files)}, SAVEE: {len(savee_files)}")
        
        if limit_per_emotion:
            # Balance PER DATASET to avoid one dataset dominating
            for dataset_files, name in [
                (ravdess_files, 'RAVDESS'),
                (tess_files, 'TESS'),
                (crema_files, 'CREMA-D'),
                (savee_files, 'SAVEE')
            ]:
                if dataset_files:
                    df = pd.DataFrame(dataset_files)
                    balanced_list = []
                    for emotion in df['emotion'].unique():
                        emotion_df = df[df['emotion'] == emotion]
                        sample_size = min(limit_per_emotion, len(emotion_df))
                        sampled = emotion_df.sample(sample_size, random_state=42)
                        balanced_list.append(sampled)
                    
                    balanced = pd.concat(balanced_list, ignore_index=True)
                    all_files.extend(balanced.to_dict('records'))
                    print(f"  {name}: {len(balanced)} balanced samples ({limit_per_emotion} per emotion)")
            
            total = len(all_files)
            print(f"Total balanced: {total}")
        else:
            all_files.extend(ravdess_files)
            all_files.extend(tess_files)
            all_files.extend(crema_files)
            all_files.extend(savee_files)
            print(f"Total files loaded: {len(all_files)}")
        
        return all_files


def load_models(device):
    models = {}
    
    model_path = os.path.join(Config.PROJECT_ROOT, 'final_models', 'best_model_audio_only.pt')
    if os.path.exists(model_path):
        print(f"✓ Loading audio-only model")
        model = EmotionModel(use_text=False)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        models['audio_only'] = model
    
    model_path = os.path.join(Config.PROJECT_ROOT, 'final_models', 'best_model_audio_text.pt')
    if os.path.exists(model_path):
        print(f"✓ Loading audio+text model")
        model = EmotionModel(use_text=True)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        models['audio_text'] = model
    
    return models


def evaluate_audio_only(model, dataloader, device):
    predictions = []
    labels = []
    datasets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Audio-only"):
            audio = batch['audio'].to(device)
            logits, _ = model(input_values=audio)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            emotion_labels = [Config.EMOTION2ID[e] for e in batch['emotion']]
            labels.extend(emotion_labels)
            datasets.extend(batch['dataset'])
    
    return np.array(predictions), np.array(labels), datasets


def evaluate_audio_text(model, dataloader, device):
    """Evaluate audio+text model with real transcriptions."""
    predictions = []
    labels = []
    datasets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Audio+text"):
            audio = batch['audio'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            
            try:
                logits, _ = model(input_values=audio, input_ids=input_ids, 
                                text_attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                emotion_labels = [Config.EMOTION2ID[e] for e in batch['emotion']]
                labels.extend(emotion_labels)
                datasets.extend(batch['dataset'])
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
    
    return np.array(predictions), np.array(labels), datasets


def compute_metrics(preds, labels, datasets):
    results = {}
    
    for dataset in set(datasets):
        mask = np.array(datasets) == dataset
        dataset_preds = preds[mask]
        dataset_labels = labels[mask]
        
        if len(dataset_preds) == 0:
            continue
        
        macro_f1 = f1_score(dataset_labels, dataset_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(dataset_labels, dataset_preds, average='micro', zero_division=0)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            dataset_labels, dataset_preds, average=None, zero_division=0,
            labels=list(range(len(CORE_EMOTIONS)))
        )
        
        per_emotion = {}
        for i, emotion in enumerate(CORE_EMOTIONS):
            per_emotion[emotion] = {
                'f1': float(f1[i]),
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'support': int(support[i])
            }
        
        results[dataset] = {
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'per_emotion': per_emotion
        }
    
    return results


def main(args):
    global DATASETS_ROOT
    if args.datasets_root:
        DATASETS_ROOT = Path(args.datasets_root)
    
    device = "cpu"
    print(f"Using device: {device}\n")
    
    # Load Whisper for transcription
    print("Loading Whisper 'tiny' model for transcription...")
    try:
        import whisper
        whisper_model = whisper.load_model("tiny")
        print("✓ Whisper loaded\n")
    except Exception as e:
        print(f"Error loading Whisper: {e}")
        print("Cannot continue without transcriptions for fair audio+text evaluation")
        return
    
    # Load datasets
    print("Loading datasets...")
    loader = DatasetLoader()
    files = loader.load_all(limit_per_emotion=args.limit_per_emotion)
    
    if not files:
        print("❌ No files found!")
        return
    
    # Load models
    print("Loading models...")
    models = load_models(device)
    
    if not models:
        print("❌ No models found!")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    
    # Create dataset and dataloader
    print("Preparing dataset with Whisper transcriptions...")
    dataset = AudioTextDataset(files, tokenizer, whisper_model)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda batch: {
            'audio': torch.stack([b['audio'] for b in batch]),
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'text_attention_mask': torch.stack([b['text_attention_mask'] for b in batch]),
            'emotion': [b['emotion'] for b in batch],
            'dataset': [b['dataset'] for b in batch],
            'text': [b['text'] for b in batch]
        }
    )
    
    # Evaluate models
    results = {}
    
    if 'audio_only' in models:
        print(f"\n{'='*70}")
        print(f"EVALUATING: AUDIO-ONLY")
        print(f"{'='*70}")
        
        preds, labels, datasets = evaluate_audio_only(models['audio_only'], dataloader, device)
        metrics = compute_metrics(preds, labels, datasets)
        results['audio_only'] = metrics
        
        print(f"\n{'Dataset':<15} {'Macro F1':<15} {'Micro F1':<15}")
        print("-" * 45)
        for dataset in sorted(metrics.keys()):
            m = metrics[dataset]
            print(f"{dataset:<15} {m['macro_f1']:<15.4f} {m['micro_f1']:<15.4f}")
        
        if metrics:
            avg_macro = np.mean([m['macro_f1'] for m in metrics.values()])
            print("-" * 45)
            print(f"{'AVERAGE':<15} {avg_macro:<15.4f}")
    
    if 'audio_text' in models:
        print(f"\n{'='*70}")
        print(f"EVALUATING: AUDIO+TEXT (with Whisper transcriptions)")
        print(f"{'='*70}")
        
        preds, labels, datasets = evaluate_audio_text(models['audio_text'], dataloader, device)
        metrics = compute_metrics(preds, labels, datasets)
        results['audio_text'] = metrics
        
        print(f"\n{'Dataset':<15} {'Macro F1':<15} {'Micro F1':<15}")
        print("-" * 45)
        for dataset in sorted(metrics.keys()):
            m = metrics[dataset]
            print(f"{dataset:<15} {m['macro_f1']:<15.4f} {m['micro_f1']:<15.4f}")
        
        if metrics:
            avg_macro = np.mean([m['macro_f1'] for m in metrics.values()])
            print("-" * 45)
            print(f"{'AVERAGE':<15} {avg_macro:<15.4f}")
    
    # Comparison
    if len(results) == 2 and all(results.values()):
        print(f"\n{'='*70}")
        print("COMPARISON: AUDIO-ONLY vs AUDIO+TEXT (fair comparison with Whisper)")
        print(f"{'='*70}\n")
        
        print(f"{'Dataset':<15} {'Audio-Only':<15} {'Audio+Text':<15} {'Improvement':<15}")
        print("-" * 65)
        
        for dataset in set(results['audio_only'].keys()) & set(results['audio_text'].keys()):
            audio_f1 = results['audio_only'][dataset]['macro_f1']
            text_f1 = results['audio_text'][dataset]['macro_f1']
            improvement = text_f1 - audio_f1
            print(f"{dataset:<15} {audio_f1:<15.4f} {text_f1:<15.4f} {improvement:+.4f}")
    
    # Save results
    output_file = os.path.join(Config.PROJECT_ROOT, "comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit_per_emotion", type=int, default=None)
    parser.add_argument("--datasets_root", type=str, default=None, help="Folder containing Ravdess/, Tess/, Crema/, Savee/")
    args = parser.parse_args()
    
    main(args)
