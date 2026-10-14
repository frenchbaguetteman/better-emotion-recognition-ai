import os

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "Emotion_Balanced")
    RAW_AUDIO_DIR = os.path.join(DATA_ROOT, "emotion_balanced")
    METADATA_CSV = os.path.join(DATA_ROOT, "selected_emotion_data_with_local_global_speaker.csv")
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
    MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")
    SPLITS_PATH = os.path.join(OUTPUT_DIR, "splits.json")
    
    # Model config
    MODEL_NAME = "microsoft/wavlm-large" # Or facebook/hubert-large-ll60k
    SAMPLE_RATE = 16000
    DURATION = 8.0 # seconds for training crops
    
    # Text Model Config
    TEXT_MODEL_NAME = "distilroberta-base"
    MAX_TEXT_LEN = 64 # Short phrases usually in this dataset
    TEXT_DROPOUT = 0.1
    
    # EMOTIONS
    EMOTIONS = ["Angry", "Happy", "Neutral", "Sad"]
    EMOTION2ID = {e: i for i, e in enumerate(EMOTIONS)}
    ID2EMOTION = {i: e for i, e in enumerate(EMOTIONS)}
