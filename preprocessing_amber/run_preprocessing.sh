#!/usr/bin/env bash
# AMBER-style preprocessing for DeepSense6G scenarios 31-34.
# Usage:  PY=~/.venvs/beamprep/bin/python bash preprocessing_amber/run_preprocessing.sh
set -euo pipefail

PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step() { echo; echo "=============== $* ==============="; }

step "1/5  audit raw data"
"$PY" "$HERE/audit_dataset.py" || echo ">> audit reported problems; continuing with the complete scenarios"

step "2/5  radar: 2-channel range-angle / range-velocity tensors (AMBER eqs. 5-8)"
"$PY" "$HERE/radar_ra_rv.py" --source all

step "3/5  lidar: BEV point-count histograms, capped at 5/cell (AMBER eq. 11)"
"$PY" "$HERE/lidar_bev.py" --source all

step "4/5  camera: resized frame cache + per-image statistics (AMBER eq. 10)"
"$PY" "$HERE/image_cache.py" --source all

step "5/5  index: sequences, GPS Cartesian, beam history, availability mask, splits"
"$PY" "$HERE/build_index.py"

echo; echo "Done. Index: data/processed_amber/index/samples.csv"
