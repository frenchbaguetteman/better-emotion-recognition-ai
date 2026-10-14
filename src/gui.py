import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import random
import glob
import torch
import whisper
from transformers import AutoTokenizer
import librosa
import time
import sys

# Ensure src is in path so we can import config and model
sys.path.append(os.path.join(os.path.dirname(__file__)))
from config import Config
from model import EmotionModel

class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Better Emotion Recognition AI")
        self.root.geometry("600x600")
        
        # State
        self.model = None
        self.tokenizer = None
        self.stt_model = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.is_recording = False
        self.recording_fs = 16000
        self.recording_duration = 5 # seconds
        self.temp_filename = "temp_recording.wav"
        self.model_path = "checkpoints/fold_0_seed_42/best_model_with_transcription.pt"
        
        # Styling
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
        
        # Main Container
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(main_frame, text="Emotion Recognition System", style="Header.TLabel").pack(pady=(0, 20))
        
        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(pady=(0, 20))
        
        # --- Section 1: Live Prediction ---
        live_frame = ttk.LabelFrame(main_frame, text="Live Prediction", padding="15")
        live_frame.pack(fill=tk.X, pady=10)
        
        self.record_btn = ttk.Button(live_frame, text=f"Record ({self.recording_duration}s)", command=self.start_recording)
        self.record_btn.pack(pady=10)
        self.record_btn.state(['disabled']) # Disabled until model loads
        
        # --- Section 2: Random File ---
        random_frame = ttk.LabelFrame(main_frame, text="Random Test File", padding="15")
        random_frame.pack(fill=tk.X, pady=10)
        
        self.random_btn = ttk.Button(random_frame, text="Pick Random File & Predict", command=self.pick_random)
        self.random_btn.pack(pady=10)
        self.random_btn.state(['disabled'])
        
        self.file_info_var = tk.StringVar(value="")
        ttk.Label(random_frame, textvariable=self.file_info_var, wraplength=500).pack(pady=5)
        
        # --- Section 3: Results ---
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="15")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_var = tk.StringVar(value="Ready")
        self.result_label = ttk.Label(result_frame, textvariable=self.result_var, font=("Helvetica", 14))
        self.result_label.pack(pady=10)
        
        self.vad_var = tk.StringVar(value="")
        ttk.Label(result_frame, textvariable=self.vad_var, font=("Courier", 10)).pack(pady=5)
        
        # Checkbox for Text
        self.use_text_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(random_frame, text="Include Text Features (Experimental)", variable=self.use_text_var).pack(pady=5)
        
        # Start Loading
        threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        try:
            self.status_var.set(f"Loading models on {self.device}...")
            
            # 1. Emotion Model
            if not os.path.exists(self.model_path):
                self.fail_init(f"Checkpoint not found at {self.model_path}")
                return
                
            self.model = EmotionModel(Config.MODEL_NAME, Config.TEXT_MODEL_NAME)
            # Load to CPU first to avoid MPS unaligned blit errors
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # 2. Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
            
            # 3. Whisper
            # Use CPU for whisper if GPU memory is tight, but ideally device
            try:
                self.stt_model = whisper.load_model("tiny", device=self.device)
            except Exception as e:
                print(f"Whisper load failed on {self.device}, trying CPU: {e}")
                self.stt_model = whisper.load_model("tiny", device="cpu")
                
            self.root.after(0, self.enable_ui)
            
        except Exception as e:
            self.fail_init(str(e))

    def fail_init(self, error_msg):
        self.root.after(0, lambda: self.status_var.set(f"Error: {error_msg}"))
        print(f"Init Error: {error_msg}")

    def enable_ui(self):
        self.status_var.set("Models Loaded. Ready.")
        self.record_btn.state(['!disabled'])
        self.random_btn.state(['!disabled'])

    def start_recording(self):
        self.run_async(self._record_and_predict)

    def _record_and_predict(self):
        self.update_status("Recording...", "red")
        
        # Record
        try:
            myrecording = sd.rec(int(self.recording_duration * self.recording_fs), 
                               samplerate=self.recording_fs, channels=1)
            sd.wait()  # Wait until recording is finished
            
            # Save to temp file (robustness)
            sf.write(self.temp_filename, myrecording, self.recording_fs)
            
            self.update_status("Processing...", "blue")
            self._predict_file(self.temp_filename, is_live=True)
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")

    def pick_random(self):
        self.run_async(self._pick_random_task)

    def _pick_random_task(self):
        self.update_status("Selecting random file...", "blue")
        
        # Find files
        # Assuming structure: Emotion_Balanced/emotion_balanced/EmotionName/PodcastID/*.wav or .flac
        search_path = os.path.join("Emotion_Balanced", "emotion_balanced", "**", "*.flac")
        # Try flac first
        files = glob.glob(search_path, recursive=True)
        if not files:
             search_path = os.path.join("Emotion_Balanced", "emotion_balanced", "**", "*.wav")
             files = glob.glob(search_path, recursive=True)
             
        if not files:
            self.update_status("No audio files found!", "red")
            return
            
        target_file = random.choice(files)
        
        # Get Ground Truth from parent folder
        true_label = os.path.basename(os.path.dirname(os.path.dirname(target_file)))
        # Note: Structure is Emotion/PodcastID/File. So parent of parent.
        # Check if true label is in emotions
        if true_label not in Config.EMOTIONS:
             # Maybe structure is slightly different, try parent
             possible_label = os.path.basename(os.path.dirname(target_file))
             if possible_label in Config.EMOTIONS:
                 true_label = possible_label
        
        self.root.after(0, lambda: self.file_info_var.set(f"File: {os.path.basename(target_file)}\nTrue Label: {true_label}"))
        
        # Play audio
        try:
            data, fs = sf.read(target_file)
            sd.play(data, fs)
            # Don't wait, let it play while processing (or wait? wait is safer for UI)
            # sd.wait() 
        except Exception as e:
            print(f"Playback error: {e}")

        self.update_status("Processing...", "blue")
        self._predict_file(target_file, is_live=False)

    def _predict_file(self, filepath, is_live=False):
        try:
            # 1. Common Audio Load
            audio_tensor, attention_mask = self.load_audio_tensor(filepath)
            if audio_tensor is None:
                raise Exception("Failed to load audio")
            audio_tensor = audio_tensor.to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)

            # 2. Transcribe
            # self.stt_model.transcribe works with path or array. Best to use path if available.
            text = self.stt_model.transcribe(filepath)["text"].strip()
            
            # 3. Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=Config.MAX_TEXT_LEN)
            input_ids = inputs.input_ids.to(self.device)
            text_mask = inputs.attention_mask.to(self.device)
            
            # 4. Inference
            with torch.no_grad():
                # Decide inputs based on checkbox
                if self.use_text_var.get():
                     # Use real text
                     final_ids = input_ids
                     final_mask = text_mask
                     mode_str = "Text+Audio"
                else:
                     # Use empty text (Audio Only mode)
                     final_ids = torch.zeros((1, Config.MAX_TEXT_LEN), dtype=torch.long, device=self.device)
                     final_mask = torch.zeros((1, Config.MAX_TEXT_LEN), dtype=torch.long, device=self.device)
                     mode_str = "Audio Only"

                logits, vad_output = self.model(
                    input_values=audio_tensor,
                    attention_mask=attention_mask,
                    input_ids=final_ids,
                    text_attention_mask=final_mask
                )
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                
            pred_emotion = Config.ID2EMOTION[pred_idx]
            confidence = probs[0][pred_idx].item()
            vad = vad_output[0].cpu().numpy()
            
            # Print Raw Logits for debugging
            print("Inference Debug:")
            print(f"  Mode: {mode_str}")
            print(f"  Logits: {logits.cpu().numpy()}")
            print(f"  Pred: {pred_emotion}")
            
            # Update UI
            res_text = f"Pred: {pred_emotion.upper()} ({confidence:.1%})\nMode: {mode_str}\nTranscript: \"{text[:40]}...\""
            vad_text = f"Val: {vad[0]:.2f} | Aro: {vad[1]:.2f} | Dom: {vad[2]:.2f}"
            
            self.root.after(0, lambda: self.display_result(res_text, vad_text, pred_emotion))
            self.update_status("Ready", "black")
            
        except Exception as e:
            self.update_status(f"Prediction Error: {e}", "red")
            print(e)
            
    def load_audio_tensor(self, path):
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
            if sr != Config.SAMPLE_RATE:
                # Use librosa for high quality resampling
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
            
            waveform = torch.from_numpy(audio_data)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Pad/Crop
            target_len = int(Config.DURATION * Config.SAMPLE_RATE)
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
            
            # Debug Stats
            print(f"Audio Loaded: {path}")
            print(f"  Shape: {waveform.shape}")
            print(f"  Mean: {waveform.mean():.4f}, Std: {waveform.std():.4f}, Max: {waveform.max():.4f}")
            
            return waveform, attention_mask # (1, T), (T,)

        except Exception as e:
            print(f"Audio Load Error: {e}")
            return None, None

    def display_result(self, res_text, vad_text, emotion):
        self.result_var.set(res_text)
        self.vad_var.set(vad_text)
        
        # Color code
        color_map = {
            "Angry": "#ffcccc",
            "Happy": "#ccffcc",
            "Sad": "#ccccff",
            "Neutral": "#f0f0f0"
        }
        bg = color_map.get(emotion, "#ffffff")
        self.result_label.configure(background=bg)

    def run_async(self, func):
        threading.Thread(target=func, daemon=True).start()

    def update_status(self, text, color="black"):
        self.root.after(0, lambda: self._set_status(text, color))

    def _set_status(self, text, color):
        self.status_var.set(text)
        self.status_label.configure(foreground=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionGUI(root)
    root.mainloop()
