# uwebasr-calibrate

Confidence calibration scripts for UWebASR speech recognition models.

## Installation

```bash
uv pip install -e .
```

## Usage

```bash
uwebasr-calibrate \
  --dataset DATASET_MANIFEST \
  --uwebasr-url https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer \
  --output-dir OUTPUT_DIR \
  --target-segments 8000 \
  --jobs 6 \
  --seed 13
```
