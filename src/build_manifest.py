import os
import pandas as pd
import glob
from tqdm import tqdm
from config import Config
import soundfile as sf

def build_manifest():
    print("Building manifest...")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Metadata
    print(f"Loading metadata from {Config.METADATA_CSV}")
    meta_df = pd.read_csv(Config.METADATA_CSV)
    
    # Create a key for fast lookup: Document_PartNumber (assuming parts align)
    # The filename format seen is MSP-PODCAST_0002_230.flac -> Document: MSP-PODCAST_0002, Part: 230
    meta_df['key'] = meta_df['Document'] + '_' + meta_df['Part Number'].astype(str)
    meta_df = meta_df.set_index('key')
    
    # 2. Walk the directory
    records = []
    
    audio_files = glob.glob(os.path.join(Config.RAW_AUDIO_DIR, "**", "*.flac"), recursive=True)
    # Also check for wav if flac not found mostly
    audio_files += glob.glob(os.path.join(Config.RAW_AUDIO_DIR, "**", "*.wav"), recursive=True)
    
    print(f"Found {len(audio_files)} audio files. Matching with metadata...")
    
    missing_meta_count = 0
    
    for file_path in tqdm(audio_files):
        filename = os.path.basename(file_path)
        # remove extension
        base = os.path.splitext(filename)[0]
        # format: MSP-PODCAST_xxxx_yyyy
        # We need to construct the key. 
        # The observed format was MSP-PODCAST_0002_230
        # Wait, if document is MSP-PODCAST_0002 and part is 230.
        # Let's hope the join works.
        
        try:
            # Reconstruct key from filename
            # Assume filename matches exactly the key logic: Document + '_' + Part
            # But the part number in filename might have padding?
            # In CSV 230 is integer. In filename it is '230'.
            # MSP-PODCAST_0002_230 -> Document=MSP-PODCAST_0002, Part=230
            # Let's split by last underscore
            parts = base.rsplit('_', 1)
            if len(parts) == 2:
                doc = parts[0]
                part_num = parts[1]
                key = f"{doc}_{part_num}"
                
                if key in meta_df.index:
                    row = meta_df.loc[key]
                    # Handle duplicate indices if any (shouldn't be for valid keys)
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                        
                    # Get duration
                    # We can read file or trust metadata 'total time' or start/end
                    # Let's read file to be safe and accurate for training
                    # info = sf.info(file_path)
                    # duration = info.duration
                    # Optimization: Use metadata duration if available to skip IO
                    duration = row['end'] - row['start']
                    
                    speaker_id = row['global_speakers']
                    if pd.isna(speaker_id):
                        missing_meta_count += 1
                        continue

                    records.append({
                        'path': file_path,
                        'duration': duration,
                        'speaker_id': speaker_id,
                        'emotion_label': row['Emotion'],
                        'valence': row['Valence'],
                        'arousal': row['Arousal'],
                        'dominance': row['Dominance'],
                        'split': 'train' # Default, will split later
                    })
                else:
                    missing_meta_count += 1
            else:
                missing_meta_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"Matched {len(records)} files.")
    print(f"Missing metadata for {missing_meta_count} files.")
    
    manifest_df = pd.DataFrame(records)
    
    # Filter for target emotions only just in case
    manifest_df = manifest_df[manifest_df['emotion_label'].isin(Config.EMOTIONS)]
    
    print(f"Final manifest size after filtering emotions: {len(manifest_df)}")
    manifest_df.to_csv(Config.MANIFEST_PATH, index=False)
    print(f"Saved manifest to {Config.MANIFEST_PATH}")

if __name__ == "__main__":
    build_manifest()
