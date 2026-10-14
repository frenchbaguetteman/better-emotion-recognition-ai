import os
import pandas as pd
from faster_whisper import WhisperModel
from config import Config
import torch
import sys
import json

def transcribe_fast():
    print("Initializing Faster-Whisper model...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FP16 is faster on A100 and maintains good speech recognition accuracy
    # cpu_threads=16 for parallel audio decoding
    model = WhisperModel("tiny", device=device, compute_type="float16", cpu_threads=16)
    
    # Load manifest
    df = pd.read_csv(Config.MANIFEST_PATH)
    print(f"Loaded manifest with {len(df)} samples.", flush=True)
    
    # Check if we can resume
    output_path = os.path.join(Config.OUTPUT_DIR, "transcripts.csv")
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_paths = set(existing_df['path'].values)
        print(f"Found {len(existing_df)} existing transcripts. Resuming...", flush=True)
    else:
        processed_paths = set()
        existing_df = pd.DataFrame(columns=['path', 'transcript'])
        existing_df.to_csv(output_path, index=False)
        
    # Filter only unproc
    to_process = df[~df['path'].isin(processed_paths)].copy()
    print(f"Remaining to process: {len(to_process)}", flush=True)
    
    if len(to_process) == 0:
        print("Nothing left to process.", flush=True)
        return

    paths = to_process['path'].tolist()
    total = len(paths)
    status_path = os.path.join(Config.OUTPUT_DIR, "transcribe_status.json")

    for i, path in enumerate(paths, 1):
        if not os.path.exists(path):
            continue
            
        try:
            # Basic transcription - I/O is the real bottleneck, not GPU
            segments, info = model.transcribe(path, beam_size=1)
            
            # Combine segments
            text = " ".join([seg.text for seg in segments]).strip()
            
            # Append to file
            batch_df = pd.DataFrame([{'path': path, 'transcript': text}])
            batch_df.to_csv(output_path, mode='a', header=False, index=False)
            
            # Update status every 5 files
            if i % 5 == 0 or i == total:
                percent = (i / total) * 100
                print(f"Progress: {i}/{total} ({percent:.2f}%) - Last file: {os.path.basename(path)}", flush=True)
                
                with open(status_path, 'w') as f:
                    json.dump({
                        "current": i, 
                        "total": total, 
                        "percent": round(percent, 2), 
                        "last_file": os.path.basename(path)
                    }, f)
            
        except Exception as e:
            print(f"Error processing {path}: {e}", flush=True)
            
    print("Transcription complete.", flush=True)

if __name__ == "__main__":
    transcribe_fast()
