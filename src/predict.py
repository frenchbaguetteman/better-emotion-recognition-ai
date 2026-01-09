import argparse
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import numpy as np
import os
import json
import pickle
from transformers import WavLMModel

from config import Config
from model import EmotionModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory containing best_model.pt and calibration.pkl")
    return parser.parse_args()

def preprocess_audio(path, chunk_duration=8.0, stride=4.0, target_sr=16000):
    # waveform, sr = torchaudio.load(path)
    audio_data, sr = sf.read(path)
    waveform = torch.from_numpy(audio_data).float()
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Sliding window
    # waveform: (1, L)
    total_len = waveform.shape[1]
    chunk_len = int(chunk_duration * target_sr)
    stride_len = int(stride * target_sr)
    
    chunks = []
    
    if total_len < chunk_len:
        # Pad
        padding = chunk_len - total_len
        padded = torch.nn.functional.pad(waveform, (0, padding))
        chunks.append(padded)
    else:
        for i in range(0, total_len - chunk_len + 1, stride_len):
            chunk = waveform[:, i:i+chunk_len]
            chunks.append(chunk)
            
        # Handle last chunk if needed?
        # Typically we just take what fits.
        
    return torch.stack(chunks) # (N, 1, L)

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Calibration
    calib_path = os.path.join(args.checkpoint_dir, "calibration.pkl")
    if not os.path.exists(calib_path):
        print("Calibration file not found. Using defaults (T=1.0, tau=0.0)")
        calib = {"temperature": 1.0, "threshold": 0.0}
    else:
        with open(calib_path, 'rb') as f:
            calib = pickle.load(f)
            
    # Load Model
    model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    model = EmotionModel(Config.MODEL_NAME, num_classes=len(Config.EMOTIONS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Process Audio
    patches = preprocess_audio(args.input_file)
    patches = patches.squeeze(1).to(device) # (N, L)
    
    # Inference
    all_logits = []
    all_vads = []
    
    with torch.no_grad():
        # Batch processing if patches is large
        batch_size = 8
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            logits, vad = model(batch)
            all_logits.append(logits)
            all_vads.append(vad)
            
    all_logits = torch.cat(all_logits, dim=0)
    all_vads = torch.cat(all_vads, dim=0)
    
    # Average
    avg_logits = torch.mean(all_logits, dim=0) # (4,)
    avg_vad = torch.mean(all_vads, dim=0) # (3,)
    
    # Calibrate
    T = calib['temperature']
    scaled_logits = avg_logits / T
    probs = torch.softmax(scaled_logits, dim=0)
    
    conf, pred_idx = torch.max(probs, dim=0)
    pred_idx = pred_idx.item()
    conf = conf.item()
    
    # Decide
    result = {
        "emotion": Config.ID2EMOTION[pred_idx],
        "confidence": conf,
        "vad": {
            "valence": avg_vad[0].item(),
            "arousal": avg_vad[1].item(),
            "dominance": avg_vad[2].item()
        },
        "is_uncertain": False,
        "raw_logits": avg_logits.tolist() # For analysis
    }
    
    if conf < calib['threshold']:
        result['is_uncertain'] = True
        result['decision'] = "ABSTAIN"
    else:
        result['decision'] = result['emotion']
        
    # Output
    print(json.dumps(result, indent=4))
    return result

if __name__ == "__main__":
    args = parse_args()
    predict(args)
