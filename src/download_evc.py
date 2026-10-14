import os
import argparse
import requests
from tqdm import tqdm
import zipfile

def download_file_with_progress(url, filename, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    # 1MB chunks
    block_size = 1024 * 1024 
    
    print(f"Downloading {filename}...")
    
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def download_dataset(token=None):
    # Direct URL to the zip file in the dataset repo
    url = "https://huggingface.co/datasets/JHU-SmileLab/NaturalVoices_EVC/resolve/main/Emotion_Balanced.zip"
    
    local_dir = "Emotion_Balanced"
    os.makedirs(local_dir, exist_ok=True)
    zip_path = os.path.join(local_dir, "Emotion_Balanced.zip")
    
    try:
        download_file_with_progress(url, zip_path, token)
        
        print("\nDownload complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
            
        print(f"Extraction complete in {os.path.abspath(local_dir)}")
        
        # Optional: Delete zip to save space
        # os.remove(zip_path)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, help="Hugging Face User Access Token", default=None)
    args = parser.parse_args()
    
    download_dataset(args.token)

