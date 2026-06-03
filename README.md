# ASR Data Quality Classifier

This project trains a binary classifier for audio-transcript quality control.
It predicts whether a pair is usable for ASR training and reports calibrated
operating points instead of relying only on threshold `0.5`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Expected local files:

- `training.csv`
- `training_audio/training_audio/data2/<folder>/<clone>.wav`

The CSV labels are normalized as:

- `usable` -> `1`
- `unusable` -> `0`

## Train

```powershell
python scripts/train.py --csv training.csv --audio-root training_audio --split-mode group
```

Outputs:

- `artifacts/quality_classifier.joblib`
- `artifacts/features.csv`
- `artifacts/audio_features.csv`
- `artifacts/metadata_with_splits.csv`
- `reports/evaluation.md`
- `reports/metrics.json`
- `reports/threshold_sweep.csv`
- `reports/test_predictions.csv`

The first run extracts audio features and may take several minutes. Later runs
reuse `artifacts/audio_features.csv`.

## Optional Cross-Modal Features

If you generate ASR hypotheses with Whisper or another ASR model, save them as:

```csv
file_path,asr_transcript
500004.../clone8.wav,...
```

Then train with:

```powershell
python scripts/train.py --asr-cache path\to\asr_cache.csv
```

The pipeline will add WER, CER, and token Jaccard similarity features.

## Inference

```powershell
python scripts/predict.py --model artifacts/quality_classifier.joblib --audio path\to\sample.wav --transcript "..."
```

The script returns:

- `prob_usable`
- `threshold_used`
- `prediction`
- `decision`: `accept`, `reject`, or `manual_review`

## Azure Upload

Do not hard-code credentials. Set the SAS token in the environment:

```powershell
$env:AZURE_SAS_TOKEN="..."
python scripts/upload_deliverables.py --container-url "https://<account>.blob.core.windows.net/<container>" --blob-prefix "<your_name>" --local-dir .
```

## Method Summary

The solution follows this approach:

1. Audit label imbalance, duplicate transcripts, and mixed-label transcripts.
2. Use a fixed train/validation/test split. Group split by `folder` is the default
   to reduce leakage between related audio clips.
3. Extract transcript, audio, and optional ASR-vs-transcript features.
4. Train probabilistic tabular models and select the best model on validation
   ROC-AUC / macro-F1.
5. Tune thresholds on validation and report both threshold `0.5` and selected
   operating points.
6. Evaluate once on the held-out test set and document error samples.
