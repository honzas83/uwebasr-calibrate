#!/bin/bash
set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

DATASETS_BASE="/Users/honzas/remote/data-ntis/projects/SpeechCloud/datasets/MALACH"
UWEBASR_BASE="https://uwebasr.zcu.cz/api/v2/speechcloud/generic"

echo "========================================================================"
echo "Starting Joint Multi-Model Calibration for: cs, de, en, sk"
echo "========================================================================"

uwebasr-calibrate \
  --dataset "${DATASETS_BASE}/MALACH-cs/ref.json" \
  --uwebasr-url "${UWEBASR_BASE}/cs/zipformer" \
  --dataset "${DATASETS_BASE}/MALACH-de/ref.json" \
  --uwebasr-url "${UWEBASR_BASE}/de/zipformer" \
  --dataset "${DATASETS_BASE}/MALACH-en/ref.json" \
  --uwebasr-url "${UWEBASR_BASE}/en/zipformer" \
  --dataset "${DATASETS_BASE}/MALACH-sk/ref.json" \
  --uwebasr-url "${UWEBASR_BASE}/sk/zipformer" \
  --output-dir "output_multi" \
  --target-segments 8000 \
  --jobs 6 \
  --seed 13 \
  --split-group speaker

echo "Multi-model joint calibration finished successfully."
