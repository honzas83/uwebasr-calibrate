#!/bin/bash
set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

DATASETS_BASE="/Users/honzas/remote/data-ntis/projects/SpeechCloud/datasets/MALACH"
UWEBASR_BASE="https://uwebasr.zcu.cz/api/v2/speechcloud/generic"

# Languages to calibrate
LANGS=("cs" "de" "en" "sk")

for LANG in "${LANGS[@]}"; do
  echo "========================================================================"
  echo "Starting Calibration for Language: ${LANG}"
  echo "========================================================================"
  
  uwebasr-calibrate \
    --dataset "${DATASETS_BASE}/MALACH-${LANG}/ref.json" \
    --uwebasr-url "${UWEBASR_BASE}/${LANG}/zipformer" \
    --output-dir "output_${LANG}" \
    --target-segments 8000 \
    --jobs 6 \
    --seed 13 \
    --split-group speaker
    
  echo "Finished Calibration for Language: ${LANG}"
  echo ""
done

echo "All calibrations finished successfully."
