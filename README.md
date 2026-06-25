# uwebasr-calibrate

Confidence calibration scripts for UWebASR speech recognition models.

## Motivation & Overview

This package implements the confidence calibration workflow described in [CALIBRATION.md](CALIBRATION.md).

The primary goal is to provide a reproducible set of open scripts that build an accuracy regressor for any user-provided evaluation set and any selected UWebASR model. Users can run the calibration locally against their own labelled data and obtain a model-specific predictor of recognition accuracy, **without sharing reference transcripts** with the UWebASR team.

Only the audio is sent to the selected UWebASR endpoint for recognition. References remain on the user's machine and are used locally to:
- Compute local training targets
- Partition train/test splits
- Extract calibration features
- Measure validation metrics
- Train the final accuracy regressor

This makes the workflow highly suitable for private evaluation sets while still allowing users to calibrate confidence-derived accuracy estimates for the exact ASR endpoint and model they intend to use.

---

## Calibration Pipeline Steps

The package executes the calibration process through the following itemized stages:

1. **ASR Recognition (Resumable)**: Audio files are converted to 16 kHz mono Ogg/Vorbis via FFmpeg and sent to the UWebASR API. Raw responses are validated and cached locally to prevent repeated network requests during subsequent runs.
2. **Text Normalization**: Both reference and hypothesis text are normalized identically (lowercased, word tokens extracted, standalone punctuation and underscores removed) to prepare for edit-distance calculations.
3. **Deterministic Alignment**: Aligns reference words to hypothesis words using `jiwer`. Reference segment spans are mapped to hypothesis word times to establish accurate time frames.
4. **Speaker-Disjoint Split**: Partition the dataset (80% train, 20% test) by speaker ID (either explicitly provided or extracted from utterance IDs) to guarantee that no speaker's voice is shared between the training set and validation metrics.
5. **Balanced Word-Aligned Segmentation**: Slices utterances into smaller segments of 10 to 256 words. The number of random segmentation variants is automatically adjusted to target approximately 8,000 segments total.
6. **Ensemble decile-mixture Sampling**: Combines segments into 512-word ensemble samples using an accuracy decile-mixture sampling strategy (75% probability from a primary decile, 25% from all other deciles) to ensure a smooth accuracy distribution.
7. **Feature Extraction**: Extracts 20 compact CTC confidence features from the concatenated probability streams, including nearest-rank percentiles, distribution statistics, count ratios, and blank/non-blank run-structure statistics.
8. **Model Training & Hyperparameter Search**: Performs a grid search over key hyperparameters (learning rate, leaves, L2 regularization) using a train-internal validation split. Trains a `HistGradientBoostingRegressor` with L1 loss on the complete train partition.
9. **Affine Calibration**: Fits a least-squares linear calibration to map the raw regressor outputs directly to expected accuracy values, bounded between 0 and 1.
10. **Evaluation & Visualization**: Computes Mean Absolute Error (MAE) and Pearson correlation for the train-internal validation split, held-out ensemble tests, and realistic long-recording windows (`test_real`). Generates calibration reports, prediction CSVs, and scatter plots.

---

## Setting Up on a Clean Machine

### 1. Prerequisites

The script requires **FFmpeg** to convert audio files.

- **macOS** (using Homebrew):
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg python3-pip python3-venv
  ```

### 2. Clone the Repository

```bash
git clone https://github.com/honzas83/uwebasr-calibrate.git
cd uwebasr-calibrate
```

### 3. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the Package

```bash
pip install -e .
```

---

## Usage

Run calibration on a single dataset manifest:

```bash
uwebasr-calibrate \
  --dataset /path/to/dataset/manifest.json \
  --uwebasr-url https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer \
  --output-dir output_dir \
  --target-segments 8000 \
  --jobs 6 \
  --seed 13
```

### Debugging with a Subset

Run a quick test on a subset of the dataset:

```bash
uwebasr-calibrate \
  --dataset /path/to/dataset/manifest.json \
  --uwebasr-url https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer \
  --output-dir output_dir_debug \
  --limit 10 \
  --split-group utterance
```

### Running on All Datasets

A helper script `calibrate_all.sh` is provided to run calibration for all four languages (Czech, German, English, Slovak) with their matching endpoints:

```bash
./calibrate_all.sh
```
