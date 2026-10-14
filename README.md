# Better Emotion Recognition AI

A research project on speech emotion recognition that holds up outside its training data. A model trained on natural podcast conversations is tested, with no further tuning, on four acted emotional-speech corpora. The project also asks whether adding the words that were spoken, from a transcript, helps beyond the sound of the voice.

The model classifies speech into four emotions: **Angry, Happy, Neutral and Sad**.

## Results

**The audio-only model averages 43.37% macro-F1** across four corpora it never saw in training. For reference, chance is 25%.

| Dataset  | Audio-only | Audio+text* | Δ (pp) |
|----------|-----------:|------------:|-------:|
| SAVEE    | **50.35%** | 48.23% | −2.13 |
| CREMA-D  | **49.46%** | 44.31% | −5.15 |
| TESS     | **42.63%** | 40.87% | −1.75 |
| RAVDESS  | **31.04%** | 29.08% | −1.96 |
| **Average** | **43.37%** | 40.62% | −2.75 |

Δ is audio+text minus audio-only, in percentage points.

\* No transcripts exist for these corpora, so the audio+text model ran with its text input empty.

![Cross-dataset comparison: macro-F1 per dataset, audio+text minus audio-only, and per-emotion F1 for each model](results/model_comparison.png)

- **Audio-only wins on all four corpora.** The largest gap is on CREMA-D.
- **Happy is the weakest class.** It is the lowest-scoring class on every corpus except RAVDESS, where it ties with Neutral. The audio+text model scores near 0 F1 for Happy on TESS.
- **Angry and Sad are the most consistent.** Both stay between about 0.43 and 0.6 F1 on every corpus, for both models.
- **RAVDESS is the hardest corpus.** Happy and Neutral both fall to about 0.1 F1 or lower there.

## Approach

**Training data.** The training clips are segments of MSP-Podcast, which is natural, unscripted podcast speech. They come from the NaturalVoices EVC "Emotion_Balanced" set. Each clip has an emotion label and valence, arousal and dominance scores. The data is split into five folds by speaker, so validation always uses voices the model has not heard.

**Model.** The audio encoder is WavLM-Large, a self-supervised speech model, fine-tuned for this task. An optional second branch reads a Whisper transcript with DistilRoBERTa and combines it with the audio. Alongside the emotion label, the model also predicts valence, arousal and dominance as a secondary objective. Confidence scores are calibrated, so the model can choose not to answer on uncertain clips.

**Evaluation.** Both variants are tested zero-shot on CREMA-D, RAVDESS, SAVEE and TESS. These are widely used acted corpora that differ in speakers, recording setups and scripts. Only clips labeled with one of the four target emotions are used.

## Discussion

- **Domain shift is the main challenge.** The model learned from spontaneous conversation and is tested on emotions performed on cue in a studio. Scores range from 31% to 50% depending on the corpus, so the choice of test corpus matters about as much as the model does.
- **Text didn't help here, but the test isn't fair to it.** The text branch received no input. With empty text, the audio+text model lost 2.75 points on average. It degraded but did not collapse.
- **Text may not help on acted corpora at all.** All four corpora are built from a small set of scripted sentences, mostly chosen to be emotionally neutral. Even with perfect transcripts, the words would say little about the emotion. Text is more likely to help on natural speech like the training data.
- **Happy transfers worst.** One plausible reason is that happiness in casual conversation sounds different from happiness performed for a recording.

## Limitations and next steps

- **No untouched in-domain test set.** Held-out speakers were also used for early stopping and calibration.
- **The text branch is untested with real transcripts on the external corpora.** Transcribing them with Whisper is the obvious next step, though the scripted sentences limit what text can add.
- **Only four emotion classes.** Fear, Disgust, Surprise and Calm clips in the test corpora were excluded.
- **Trained weights are not published.** Each checkpoint is 1.2–1.6 GB.

## Repository

```
src/       data preparation, model, training, calibration, evaluation and inference
scripts/   5-fold training launcher
tests/     model smoke test
results/   evaluation figures
```

Built with PyTorch and Hugging Face Transformers. The dependencies are in `requirements.txt`. Datasets and model weights are not included.
