"""
Visualize model comparison results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
results_file = PROJECT_ROOT / "comparison_results.json"
if not results_file.exists():
    print(f"Error: {results_file} not found")
    exit(1)

with open(results_file) as f:
    results = json.load(f)

# Extract data
datasets = sorted(set(list(results['audio_only'].keys()) + list(results['audio_text'].keys())))
audio_only_f1 = [results['audio_only'].get(ds, {}).get('macro_f1', 0) for ds in datasets]
audio_text_f1 = [results['audio_text'].get(ds, {}).get('macro_f1', 0) for ds in datasets]
improvements = [audio_text_f1[i] - audio_only_f1[i] for i in range(len(datasets))]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Side-by-side comparison
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(datasets))
width = 0.35

bars1 = ax1.bar(x - width/2, audio_only_f1, width, label='Audio-Only', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, audio_text_f1, width, label='Audio+Text', color='#A23B72', alpha=0.8)

ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax1.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.legend(fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=9)

# 2. Improvement/Degradation
ax2 = plt.subplot(2, 2, 2)
colors = ['#06A77D' if x >= 0 else '#D62828' for x in improvements]
bars = ax2.bar(datasets, improvements, color=colors, alpha=0.8)

ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1 Improvement', fontsize=12, fontweight='bold')
ax2.set_title('Audio+Text vs Audio-Only (Improvement/Degradation)', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, improvements):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:+.2%}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')

# 3. Per-emotion breakdown for Audio-Only
ax3 = plt.subplot(2, 2, 3)
emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
audio_only_emotions = {}
for dataset in datasets:
    if dataset in results['audio_only']:
        for emotion in emotions:
            if emotion not in audio_only_emotions:
                audio_only_emotions[emotion] = []
            f1 = results['audio_only'][dataset]['per_emotion'].get(emotion, {}).get('f1', 0)
            audio_only_emotions[emotion].append(f1)

x_pos = np.arange(len(datasets))
width = 0.2
for i, emotion in enumerate(emotions):
    ax3.bar(x_pos + i*width, audio_only_emotions[emotion], width, label=emotion, alpha=0.8)

ax3.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax3.set_title('Audio-Only: Per-Emotion Performance', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos + width * 1.5)
ax3.set_xticklabels(datasets)
ax3.legend(fontsize=10)
ax3.set_ylim(0, 1.05)
ax3.grid(axis='y', alpha=0.3)

# 4. Per-emotion breakdown for Audio+Text
ax4 = plt.subplot(2, 2, 4)
audio_text_emotions = {}
for dataset in datasets:
    if dataset in results['audio_text']:
        for emotion in emotions:
            if emotion not in audio_text_emotions:
                audio_text_emotions[emotion] = []
            f1 = results['audio_text'][dataset]['per_emotion'].get(emotion, {}).get('f1', 0)
            audio_text_emotions[emotion].append(f1)

x_pos = np.arange(len(datasets))
width = 0.2
for i, emotion in enumerate(emotions):
    ax4.bar(x_pos + i*width, audio_text_emotions[emotion], width, label=emotion, alpha=0.8)

ax4.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax4.set_title('Audio+Text: Per-Emotion Performance', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos + width * 1.5)
ax4.set_xticklabels(datasets)
ax4.legend(fontsize=10)
ax4.set_ylim(0, 1.05)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to model_comparison.png")

# Create a second figure with summary statistics
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Summary table for Audio-Only
ax = axes[0]
ax.axis('off')
summary_data = []
for dataset in datasets:
    if dataset in results['audio_only']:
        f1 = results['audio_only'][dataset]['macro_f1']
        summary_data.append([dataset, f'{f1:.2%}'])

summary_data.append(['AVERAGE', f'{np.mean(audio_only_f1):.2%}'])

table1 = ax.table(cellText=summary_data, 
                 colLabels=['Dataset', 'Macro F1'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table1.auto_set_font_size(False)
table1.set_fontsize(12)
table1.scale(1, 2.5)

# Style header
for i in range(2):
    table1[(0, i)].set_facecolor('#2E86AB')
    table1[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    color = '#E8F4F8' if i % 2 == 0 else 'white'
    for j in range(2):
        table1[(i, j)].set_facecolor(color)
        if i == len(summary_data):  # Last row (average)
            table1[(i, j)].set_text_props(weight='bold')
            table1[(i, j)].set_facecolor('#2E86AB')
            table1[(i, j)].set_text_props(weight='bold', color='white')

ax.set_title('Audio-Only Model Summary', fontsize=14, fontweight='bold', pad=20)

# Summary table for Audio+Text
ax = axes[1]
ax.axis('off')
summary_data = []
for dataset in datasets:
    if dataset in results['audio_text']:
        f1 = results['audio_text'][dataset]['macro_f1']
        summary_data.append([dataset, f'{f1:.2%}'])

summary_data.append(['AVERAGE', f'{np.mean(audio_text_f1):.2%}'])

table2 = ax.table(cellText=summary_data,
                 colLabels=['Dataset', 'Macro F1'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table2.auto_set_font_size(False)
table2.set_fontsize(12)
table2.scale(1, 2.5)

# Style header
for i in range(2):
    table2[(0, i)].set_facecolor('#A23B72')
    table2[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    color = '#F4E8F4' if i % 2 == 0 else 'white'
    for j in range(2):
        table2[(i, j)].set_facecolor(color)
        if i == len(summary_data):  # Last row (average)
            table2[(i, j)].set_facecolor('#A23B72')
            table2[(i, j)].set_text_props(weight='bold', color='white')

ax.set_title('Audio+Text Model Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'model_summary.png', dpi=300, bbox_inches='tight')
print("✅ Summary table saved to model_summary.png")

plt.show()
