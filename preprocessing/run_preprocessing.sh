#!/usr/bin/env bash
# End-to-end preprocessing for DeepSense6G 2022 Multi-Modal, scenarios 31-34.
# Usage:  bash preprocessing/run_preprocessing.sh [--with-augmentation]
set -euo pipefail

PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUG=0
[[ "${1:-}" == "--with-augmentation" ]] && AUG=1

step() { echo; echo "=============== $* ==============="; }

step "1/5  audit raw data"
"$PY" "$HERE/audit_dataset.py" || echo ">> audit reported problems; continuing with the complete scenarios"

step "2/5  radar: range-angle + range-velocity maps"
"$PY" "$HERE/preprocess_radar.py" --split all

step "3/5  lidar: static background estimation"
"$PY" "$HERE/preprocess_lidar.py" --stage background

step "4/5  lidar: background removal"
"$PY" "$HERE/preprocess_lidar.py" --stage filter --split all

step "5/5  unified sample index, power vectors, difficulty metrics, splits"
"$PY" "$HERE/build_index.py"

if [[ $AUG -eq 1 ]]; then
  step "optional  augmentation (adaptation split)"
  "$PY" "$HERE/augment_radar.py" --split adaptation
  "$PY" "$HERE/augment_lidar.py" --split adaptation
  "$PY" "$HERE/augment_image.py" --split adaptation
fi

echo; echo "Done. Index: data/processed/index/samples.csv"
