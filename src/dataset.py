import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
import random
import os
from torch.utils.data import Dataset
from config import Config

class EmotionDataset(Dataset):
    def __init__(self, manifest_df, processor=None, mode='train', target_sr=16000):
        self.df = manifest_df
        self.mode = mode
        self.target_sr = target_sr
        self.processor = processor # HuggingFace feature extractor if needed, or raw audio
        
        # Precomputed label indices
        self.df['label_idx'] = self.df['emotion_label'].map(Config.EMOTION2ID)
        
    def __len__(self):
        return len(self.df)
    
    def _load_audio(self, path):
        # Use soundfile to avoid backend issues with torchaudio/TorchCodec
        try:
            audio_data, sr = sf.read(path)
        except Exception as e:
            # Fallback or re-raise
            raise e
            
        waveform = torch.from_numpy(audio_data).float()
        
        # soundfile returns (time, channels) or (time,)
        # torchaudio expects (channels, time)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # (1, time)
        else:
            waveform = waveform.t() # (channels, time)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
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
            
        # 3. Speed Perturbation (Resample)
        if random.random() < 0.5:
            speed_factor = random.uniform(0.95, 1.05)
            # Resample is basically speed change if we don't fix pitch.
            # But standard torch resample keeps pitch? No, resample changes speed and pitch together if we just re-interpret SR?
            # Actually torchaudio.transforms.Resample changes sampling rate.
            # To change speed:
            new_sr = int(self.target_sr * speed_factor)
            resampler = torchaudio.transforms.Resample(self.target_sr, new_sr)
            # This effectively changes speed if we play it back at target_sr? 
            # No, this just changes logical SR.
            # Speed perturbation usually involves resampling signal to new length but keeping SR constant, effectively stretching it.
            # e.g. signal is 16000 samples (1s). Resample to 16800. Play at 16000 -> 1.05s (slower).
            # To keep it simple, let's skip complex speed pertub or use torchaudio.transforms.TimeStretch (requires STFT).
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
        else:
            # Pad
            padding = target_len - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform

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

        if self.mode == 'train':
            waveform = self._augment(waveform)
            
        waveform = self._crop(waveform)
        
        # Squeeze channel dim for WavLM: (seq_len,)
        input_values = waveform.squeeze(0)
        
        label = row['label_idx']
        vad = torch.tensor([row['valence'], row['arousal'], row['dominance']], dtype=torch.float32)
        
        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
            "vad": vad
        }
