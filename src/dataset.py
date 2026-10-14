import torch
import torch.nn as nn
import soundfile as sf
import pandas as pd
import numpy as np
import random
import os
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from config import Config

class EmotionDataset(Dataset):
    def __init__(self, manifest_df, processor=None, tokenizer=None, mode='train', target_sr=16000, cache_audio=True):
        self.df = manifest_df.reset_index(drop=True)
        self.mode = mode
        self.target_sr = target_sr
        self.processor = processor # HuggingFace feature extractor if needed, or raw audio
        self.tokenizer = tokenizer
        self.cache_audio = cache_audio
        self.audio_cache = {} if cache_audio else None
        
        # Precompute text inputs (tokenization)
        self.text_inputs_cache = {}
        
        # Load Transcripts
        transcript_path = os.path.join(Config.OUTPUT_DIR, "transcripts.csv")
        self.path2text = {}
        if os.path.exists(transcript_path):
            print(f"Loading transcripts from {transcript_path}...")
            try:
                t_df = pd.read_csv(transcript_path)
                # Create a lookup dictionary. Check if duplicates exist?
                t_df = t_df.dropna(subset=['path'])
                self.path2text = dict(zip(t_df['path'], t_df['transcript']))
            except Exception as e:
                print(f"Warning: Could not load transcripts: {e}")
        else:
            print(f"Warning: Transcripts file not found at {transcript_path}. Text inputs will be empty.")
        
        # Precomputed label indices
        self.df['label_idx'] = self.df['emotion_label'].map(Config.EMOTION2ID)
        
        # Precompute text encodings during init
        print("Precomputing text encodings...")
        self._precompute_text_encodings()
        
    def _precompute_text_encodings(self):
        """Tokenize all texts once during initialization"""
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            path = row['path']
            
            # Try to get text from manifest first, then from separate lookup
            text = row.get('text', self.path2text.get(path, ""))
            
            if not isinstance(text, str):
                text = ""
            text = text.strip()
            
            if self.tokenizer is not None and text:
                text_inputs = self.tokenizer(
                    text, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=Config.MAX_TEXT_LEN, 
                    return_tensors="pt"
                )
                self.text_inputs_cache[path] = {
                    'input_ids': text_inputs.input_ids.squeeze(0),
                    'text_mask': text_inputs.attention_mask.squeeze(0)
                }
            else:
                self.text_inputs_cache[path] = {
                    'input_ids': torch.zeros(Config.MAX_TEXT_LEN, dtype=torch.long),
                    'text_mask': torch.zeros(Config.MAX_TEXT_LEN, dtype=torch.long)
                }
        
    def __len__(self):
        return len(self.df)
    
    def _load_audio(self, path):
        """Load and resample audio efficiently"""
        if self.cache_audio and path in self.audio_cache:
            return self.audio_cache[path]
        
        try:
            audio_data, sr = sf.read(path)
        except Exception as e:
            # Add context to error
            if not os.path.exists(path):
                print(f"Error: File not found: {path} (Check if dataset volume is mounted at /home/jl_fs/)")
            raise e
            
        # audio_data is (T, C) or (T,)
        
        # Resample if needed using scipy directly to avoid librosa dependency issues
        if sr != self.target_sr:
            # Resample with scipy resample_poly
            # Librosa's resample(res_type='scipy') just calls this anyway
            from scipy.signal import resample_poly
            
            # GCD for resampling factors
            from math import gcd
            g = gcd(self.target_sr, sr)
            up = self.target_sr // g
            down = sr // g
            
            # Handle multidimensional
            if audio_data.ndim > 1:
                # Resample along axis 0 (time)
                audio_data = resample_poly(audio_data, up, down, axis=0)
                audio_data = audio_data.T # (C, T)
            else:
                audio_data = resample_poly(audio_data, up, down)
                audio_data = audio_data[np.newaxis, :] # (1, T)
        else:
            # No resample
            if audio_data.ndim == 1:
                audio_data = audio_data[np.newaxis, :]
            else:
                 audio_data = audio_data.T # (C, T)

        waveform = torch.from_numpy(audio_data.astype(np.float32)).float()
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Cache if enabled
        if self.cache_audio:
            self.audio_cache[path] = waveform
            
        return waveform

    def _augment(self, waveform):
        # 1. Time Shift
        if random.random() < 0.5:
            shift = int(random.uniform(-0.1, 0.1) * waveform.shape[1])
            waveform = torch.roll(waveform, shift, dims=1)
            
        # 2. Add Gaussian Noise
        if random.random() < 0.5:
            noise = torch.randn_like(waveform) * random.uniform(0, 0.01)
            waveform = waveform + noise
            
        # 3. Speed Perturbation
        if random.random() < 0.5:
            # Skip speed perturbation for now to remove torchaudio dependency complexity
            pass 

        return waveform

    def _crop(self, waveform):
        # Random crop to Config.DURATION
        target_len = int(Config.DURATION * self.target_sr)
        current_len = waveform.shape[1]
        
        if current_len > target_len:
            if self.mode == 'train':
                start = random.randint(0, current_len - target_len)
            else:
                # Center crop for val/test
                start = (current_len - target_len) // 2
            
            waveform = waveform[:, start:start+target_len]
            attention_mask = torch.ones(target_len, dtype=torch.long)
        else:
            # Pad
            padding = target_len - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            attention_mask = torch.cat(
                [
                    torch.ones(current_len, dtype=torch.long),
                    torch.zeros(padding, dtype=torch.long),
                ],
                dim=0,
            )
            
        return waveform, attention_mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        
        try:
            waveform = self._load_audio(path)
        except Exception as e:
            # Return a silence tensor or handle error
            print(f"Error loading {path}: {e}")
            target_len = int(Config.DURATION * self.target_sr)
            waveform = torch.zeros(1, target_len)
            attention_mask = torch.zeros(target_len, dtype=torch.long)
        else:
            if self.mode == 'train':
                waveform = self._augment(waveform)
            
            waveform, attention_mask = self._crop(waveform)

        # Squeeze channel dim for WavLM: (seq_len,)
        input_values = waveform.squeeze(0)
        
        label = row['label_idx']
        vad = torch.tensor([row['valence'], row['arousal'], row['dominance']], dtype=torch.float32)
        
        # Get cached text inputs
        text_cached = self.text_inputs_cache.get(path)
        if text_cached is not None:
            input_ids = text_cached['input_ids'].clone()
            text_mask = text_cached['text_mask'].clone()
        else:
            # Fallback
            input_ids = torch.zeros(Config.MAX_TEXT_LEN, dtype=torch.long)
            text_mask = torch.zeros(Config.MAX_TEXT_LEN, dtype=torch.long)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "text_attention_mask": text_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "vad": vad
        }
