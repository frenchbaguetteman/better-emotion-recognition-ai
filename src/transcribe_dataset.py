import argparse
import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from config import Config
import torch

def transcribe_dataset():
    print("Initializing Whisper ASR pipeline...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Using small model for speed, can be upgraded to 'medium' or 'large-v2'
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
    
    # Load manifest
    df = pd.read_csv(Config.MANIFEST_PATH)
    print(f"Loaded manifest with {len(df)} samples.")
    
    # Check if we can resume
    output_path = os.path.join(Config.OUTPUT_DIR, "transcripts.csv")
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_paths = set(existing_df['path'].values)
        print(f"Found {len(existing_df)} existing transcripts. Resuming...")
    else:
        processed_paths = set()
        existing_df = pd.DataFrame(columns=['path', 'transcript'])
        # write header
        existing_df.to_csv(output_path, index=False)
        
    # Filter only unproc
    to_process = df[~df['path'].isin(processed_paths)].copy()
    print(f"Remaining to process: {len(to_process)}")
    
    # Process
    results = []
    # Reduced batch size to 128 to save VRAM for concurrent training
    batch_size = 128 
    paths = to_process['path'].tolist()
    
    # We use the pipeline's built-in batching if possible, but manual batching gives more control
    # for saving intermediate results
    
    for i in tqdm(range(0, len(to_process), batch_size)):
        batch_paths = paths[i : i+batch_size]
        
        # Whisper pipeline usually takes inputs as list of files or numpy arrays
        # Ensure paths exist
        valid_batch = [p for p in batch_paths if os.path.exists(p)]
        
        if not valid_batch:
            continue
            
        try:
            # Run inference
            batch_out = transcriber(valid_batch, batch_size=len(valid_batch))
            
            # Prepare rows
            rows = []
            for path, out in zip(valid_batch, batch_out):
                text = out['text'].strip()
                rows.append({'path': path, 'transcript': text})
            
            # Append to file immediately
            batch_df = pd.DataFrame(rows)
            batch_df.to_csv(output_path, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            
    print("Transcription complete.")

if __name__ == "__main__":
    transcribe_dataset()
