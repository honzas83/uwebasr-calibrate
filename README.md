# uwebasr-calibrate

Confidence calibration scripts for UWebASR speech recognition models.

## Setting Up on a Clean Machine

Follow these steps to set up and run the calibration workflow from scratch.

### 1. Prerequisites

The script requires **FFmpeg** to convert audio files to Ogg/Vorbis format before sending them to the UWebASR server.

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

Install the package and all its dependencies in editable mode:

```bash
pip install -e .
```

*(Alternatively, if you have `uv` installed, you can run `uv pip install -e .`)*

---

## Usage

You can run calibration on a single dataset manifest:

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

If you want to run a quick test on a subset of the dataset (e.g. the first 10 utterances), you can use the `--limit` and `--split-group` options:

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
