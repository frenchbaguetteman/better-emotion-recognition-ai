import argparse
import torch
import librosa
import numpy as np
import whisper
import os
from transformers import AutoTokenizer
from config import Config
from model import EmotionModel

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_audio(path, target_sr=16000, duration=8.0):
    try:
        # Use soundfile to match training exactly
        import soundfile as sf
        
        # Read file (T, C) usually
        audio_data, sr = sf.read(path)
        
        # Convert to float32
        audio_data = audio_data.astype(np.float32)

        # Check dimensions (sf read gives (Time, Channels) for stereo, (Time,) for mono)
        if audio_data.ndim == 1:
            # (T,) -> (1, T)
            audio_data = audio_data[np.newaxis, :]
        else:
            # (T, C) -> (C, T)
            audio_data = audio_data.T
        
        # Resample if needed
        if sr != target_sr:
            # Use librosa for high quality resampling
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        
        waveform = torch.from_numpy(audio_data)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad/Crop
        target_len = int(duration * target_sr)
        if waveform.shape[1] > target_len:
            start = (waveform.shape[1] - target_len) // 2
            waveform = waveform[:, start:start+target_len]
            attention_mask = torch.ones(target_len, dtype=torch.long)
        else:
            padding = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            attention_mask = torch.cat(
                [
                    torch.ones(waveform.shape[1] - padding, dtype=torch.long),
                    torch.zeros(padding, dtype=torch.long),
                ],
                dim=0,
            )
            
        return waveform, attention_mask # (1, T), (T,)

    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def transcribe(audio_path, model):
    # Depending on whisper version, it might return a dict or object
    result = model.transcribe(audio_path)
    # result is a dict with 'text' key
    return result["text"].strip()

def predict(audio_path, model_path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if not torch.backends.mps.is_available() and torch.cuda.is_available():
        device = "cuda"
        
    print(f"Using device: {device}")
    
    # Load Main Model
    print("Loading Emotion Model...")
    model = EmotionModel(Config.MODEL_NAME, Config.TEXT_MODEL_NAME)
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    # Load to CPU first to avoid MPS unaligned blit errors
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    
    # Load Whisper
    print("Loading Whisper for transcription...")
    try:
        stt_model = whisper.load_model("tiny", device=device)
    except Exception as e:
        print(f"Warning: Failed to load Whisper on {device}, falling back to CPU. Error: {e}")
        stt_model = whisper.load_model("tiny", device="cpu")
    
    # Process Input
    print(f"Processing {audio_path}...")
    audio_tensor, attention_mask = load_audio(audio_path)
    if audio_tensor is None:
        return
    
    audio_tensor = audio_tensor.to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    # Transcribe
    print("Transcribing audio content...")
    text = transcribe(audio_path, stt_model)
    print(f"Transcript: '{text}'")
    
    # Tokenize
    if text:
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=Config.MAX_TEXT_LEN)
        input_ids = inputs.input_ids.to(device)
        text_attention_mask = inputs.attention_mask.to(device)
    else:
        input_ids = torch.zeros((1, Config.MAX_TEXT_LEN), dtype=torch.long, device=device)
        text_attention_mask = torch.zeros((1, Config.MAX_TEXT_LEN), dtype=torch.long, device=device)
    
    # Inference
    print("Running Multi-Modal Inference...")
    with torch.no_grad():
        logits, vad_output = model(
            input_values=audio_tensor,
            attention_mask=attention_mask,
            input_ids=input_ids, 
            text_attention_mask=text_attention_mask
        )
        
        # Softmax
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    emotion = Config.ID2EMOTION[pred_idx]
    confidence = probs[0][pred_idx].item()
    
    print("\n" + "=" * 40)
    print(f"PREDICTION: {emotion.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print(f"VAD (Val/Aro/Dom): {vad_output[0].cpu().numpy()}")
    print("=" * 40 + "\n")
    
    return emotion, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an audio file")
    parser.add_argument("audio_path", help="Path to audio file (wav, mp3, etc.)")
    parser.add_argument("--model_path", default="checkpoints/best_model.pt", help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    predict(args.audio_path, args.model_path)
