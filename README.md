# Speech emotion recognition

A research prototype for recognizing emotion from speech with a WavLM backbone and attention pooling.
It combines four-class emotion prediction with valence, arousal, and dominance regression, then
adds temperature calibration and an option to abstain on low-confidence predictions.

The configured classes are **Angry, Happy, Neutral, and Sad**. The default backbone is
`microsoft/wavlm-large`.

## Pipeline

1. Match audio recordings to metadata and build a manifest.
2. Group recordings by speaker for cross-validation folds.
3. Fine-tune the classification and regression heads with the backbone.
4. Calibrate confidence on a separate calibration subset.
5. Predict on overlapping audio windows and aggregate the outputs.

## Setup and data

```sh
git clone https://github.com/frenchbaguetteman/better-emotion-recognition-ai.git
cd better-emotion-recognition-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Supply the audio and matching metadata yourself. The expected layout is defined in
[src/config.py](src/config.py):

```text
Emotion_Balanced/
  emotion_balanced/                       # .flac / .wav audio, searched recursively
  selected_emotion_data_with_local_global_speaker.csv
```

The manifest builder expects `Document`, `Part Number`, `start`, `end`, `global_speakers`,
`Emotion`, `Valence`, `Arousal`, and `Dominance` metadata columns. Filenames are matched by
document and part number. The download script is a placeholder; datasets and trained checkpoints
are not included. Loading the backbone also requires access to the pretrained model.

## Development workflow

After preparing compatible data:

```sh
python src/build_manifest.py
python src/make_speaker_folds.py
python src/train.py --fold 0 --seed 42
python src/calibrate.py --fold 0 --seed 42
```

After training, inference accepts an audio path and a checkpoint directory:

```sh
python src/predict.py sample.wav --checkpoint_dir checkpoints/fold_0_seed_42
```

These commands describe the implemented workflow; an end-to-end reproducible run has not been
documented in this repository yet.

## Code map

| Area | File |
| --- | --- |
| Audio loading and preprocessing | [src/dataset.py](src/dataset.py) |
| WavLM, attention pooling, and output heads | [src/model.py](src/model.py) |
| Speaker-grouped fold generation | [src/make_speaker_folds.py](src/make_speaker_folds.py) |
| Training and checkpoint selection | [src/train.py](src/train.py) |
| Confidence calibration | [src/calibrate.py](src/calibrate.py) |
| Metrics and reporting | [src/evaluate.py](src/evaluate.py), [src/report.py](src/report.py) |
| Windowed inference | [src/predict.py](src/predict.py) |

## Research status

No benchmark results are claimed yet. Before reporting final performance:

- Add an untouched test set: evaluation currently combines the validation subset used for checkpoint
  selection with the subset used for calibration.
- Fix the missing `torch.nn` import in `src/evaluate.py` before running evaluation.
- Record dataset provenance, dependency versions, trained checkpoints, and comparisons to a baseline.

The forward-pass test checks output shapes; it does not validate model quality or the full pipeline.
