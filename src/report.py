import os
import json
import glob
import pandas as pd
import numpy as np
from config import Config

def generate_report():
    print("Generating report...")
    
    # Find all result files
    result_files = glob.glob(os.path.join(Config.PROJECT_ROOT, "checkpoints", "**", "results.json"), recursive=True)
    
    if not result_files:
        print("No results found.")
        return
    
    results = []
    for f in result_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            results.append(data)
            
    df = pd.DataFrame(results)
    
    # Aggregation
    print("\nPer Fold Results:")
    print(df[['fold', 'accuracy', 'macro_f1', 'ece', 'nll']])
    
    print("\nSummary Statistics:")
    summary = df[['accuracy', 'macro_f1', 'ece', 'nll']].agg(['mean', 'std'])
    print(summary)
    
    # VAD Metrics Averages
    vad_rmses = np.array(df['vad_rmse'].tolist())
    vad_maes = np.array(df['vad_mae'].tolist())
    vad_pearsons = np.array(df['vad_pearson'].tolist())
    
    print("\nVAD RMSE (Valence, Arousal, Dominance):")
    print(f"Mean: {vad_rmses.mean(axis=0)}")
    print(f"Std:  {vad_rmses.std(axis=0)}")
    
    print("\nVAD Pearson (Valence, Arousal, Dominance):")
    print(f"Mean: {vad_pearsons.mean(axis=0)}")
    print(f"Std:  {vad_pearsons.std(axis=0)}")

if __name__ == "__main__":
    generate_report()
