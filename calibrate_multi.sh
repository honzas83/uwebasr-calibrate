#!/bin/bash
set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

DATASETS_BASE="/Users/honzas/remote/data-ntis/projects/SpeechCloud/datasets/MALACH"
UWEBASR_BASE="https://uwebasr.zcu.cz/api/v2/speechcloud/generic"

echo "========================================================================"
echo "Starting Multi-Model Calibration for: cs, de, en, sk"
echo "========================================================================"

uwebasr-calibrate \
  --dataset "${DATASETS_BASE}/MALACH-cs/ref.json" \
  --uwebasr-url "${UWEBASR_BASE}/cs/zipformer" \
  --output-dir "output_cs" \
  --dataset "${DATASETS_BASE}/MALACH-de/ref.json" \
  --uwebasr-url "${UWEBASR_BASE}/de/zipformer" \
  --output-dir "output_de" \
  --dataset "${DATASETS_BASE}/MALACH-en/ref.json" \
  --uwebasr-url "${UWEBASR_BASE}/en/zipformer" \
  --output-dir "output_en" \
  --dataset "${DATASETS_BASE}/MALACH-sk/ref.json" \
  --uwebasr-url "${UWEBASR_BASE}/sk/zipformer" \
  --output-dir "output_sk" \
  --target-segments 8000 \
  --jobs 6 \
  --seed 13 \
  --split-group speaker

echo "Multi-model calibration finished successfully."
