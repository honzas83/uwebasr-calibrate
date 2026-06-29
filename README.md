# uwebasr-calibrate

Confidence calibration scripts for UWebASR speech recognition models.

## Motivation & Overview

This package implements the confidence calibration workflow described in [CALIBRATION.md](CALIBRATION.md) for [UWebASR](https://uwebasr.zcu.cz).

[UWebASR](https://uwebasr.zcu.cz) is an automatic speech recognition (ASR) web service developed by the University of West Bohemia in Pilsen. It provides cloud-based speech-to-text recognition API endpoints supporting multiple languages (such as Czech, English, German, Slovak, and others) using state-of-the-art speech recognition models like Zipformer.

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

Run calibration on a single dataset manifest (for the required structure, see [DATASET.md](DATASET.md)):

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
```

### Running on a Pre-split Dataset

If your dataset is already split into training and testing sets, you should provide a single manifest file containing a `split` field (with values `"train"` or `"test"`) for each row. The tool will automatically detect this split and use it for training and testing rather than partitioning the dataset by speaker. See [DATASET.md](DATASET.md) for more details.

```bash
uwebasr-calibrate \
  --dataset /path/to/manifest.json \
  --uwebasr-url https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer \
  --output-dir output_dir
```


### Running a Multi-Model Calibration Setup

To run joint calibration on multiple datasets and endpoints in a single command (multi-model setup) to train a single unified model, you can specify the `--dataset` and `--uwebasr-url` parameters multiple times, and provide a single `--output-dir`:

```bash
uwebasr-calibrate \
  --dataset datasets/cs/ref.json --uwebasr-url https://uwebasr.zcu.cz/api/v2/speechcloud/generic/cs/zipformer \
  --dataset datasets/de/ref.json --uwebasr-url https://uwebasr.zcu.cz/api/v2/speechcloud/generic/de/zipformer \
  --output-dir output_multi \
  --jobs 6
```


### Running on All Datasets

A helper script `calibrate_all.sh` is provided to run calibration for all four languages (Czech, German, English, Slovak) with their matching endpoints:

```bash
./calibrate_all.sh
```

---

## Calibrated Recognition (Inference)

A standalone script `uwebasr-calibrated.py` is provided for running speech recognition on audio files using the UWebASR API, while simultaneously predicting the word recognition accuracy of the transcript using a trained calibration model.

This script does not depend on the training package and can be run independently as long as `joblib`, `scikit-learn`, and `numpy` are installed.

### Basic Usage

Run recognition and predict accuracy for one or more audio files:

```bash
python3 uwebasr-calibrated.py \
  speechcloud/generic/cs/zipformer \
  audio_file1.wav audio_file2.wav \
  --calibration-model path/to/model.joblib \
  --format json \
  --overwrite
```

### JSON Accuracy Metadata Output

By default, the script partitions the recording into word-aligned windows of `--window-size` words (default is 256). It extracts features and estimates accuracy for each window separately, then aggregates them to compute the overall metrics.

The script saves the resulting metrics to a `<filename>.accuracy.json` file containing the following keys:

- `estimated_accuracy`: The overall predicted word-level accuracy (computed as a word-count-weighted average of window accuracies, or `null`).
- `words_per_minute`: Global words per minute over the active speech duration.
- `audio_length`: Total duration of the audio file in seconds.
- `speech_ratio`: The ratio of active speech frames to total audio frames.
- `non_speech_ratio`: The ratio of non-speech (silence/blank) frames to total audio frames.
- `recognized_word_count`: Total number of recognized words in the transcript.
- `expected_error_count`: Estimated number of word errors in the transcript (summed across all windows).
- `windows`: A list of dictionaries, where each entry contains details for one window:
  - `window_idx`: Index of the window (0-indexed).
  - `start_time` / `end_time`: Time boundaries of the window in seconds.
  - `word_count`: Number of words in the window.
  - `estimated_accuracy`: Predicted accuracy for this window.
  - `words_per_minute`: Words per minute for this window.
  - `audio_length`: Active speech duration of the window in seconds.
  - `speech_ratio` / `non_speech_ratio`: Speech and non-speech ratios inside the window frame boundaries.
  - `expected_error_count`: Estimated number of word errors in this window.

If the calibration model is not provided, the script still outputs this file with all metadata populated and `estimated_accuracy` set to `null`. If UWebASR fails to return CTC probability streams for a file, all metrics will be set to `null`.

### Options

- `MODEL`: The SpeechCloud model identifier (e.g. `speechcloud/generic/cs/zipformer`).
- `FN`: Path to one or more input audio files.
- `--calibration-model PATH`: Path to the trained calibration model (`model.joblib`).
- `--window-size N`: Window size in words for accuracy estimation (defaults to 256, set to 0 to evaluate the entire file as a single window).
- `--uwebasr-url URL`: UWebASR service root (defaults to `https://uwebasr.zcu.cz`).
- `--format FORMAT`: Generate specific output formats (e.g. `json`, `txt`, `s.txt`, `vtt`, `s.vtt`, `jsonl`).
- `--n-workers N`: Number of parallel workers for concurrent API requests (defaults to 1).
- `--output-dir DIR`: Optional output directory for saving all transcript and accuracy files.
- `--overwrite`: Allow overwriting of existing output files.

---

## LINDAT/CLARIAH-CZ Support & References

UWebASR is developed by the Department of Cybernetics at the University of West Bohemia in Pilsen and is integrated into the [LINDAT/CLARIAH-CZ](https://lindat.cz) research infrastructure.

If you use this software or the UWebASR services in your research, please cite the following reference:

- ŠVEC, Jan; LEHEČKA, Jan a IRCING, Pavel. *Current State of the UWebASR - Web-Based ASR Service for Czech, Slovak, German, and English*. CLARIN, 2025. ISSN 2773-2177.

