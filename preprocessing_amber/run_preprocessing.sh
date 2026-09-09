#!/usr/bin/env bash
# AMBER-style preprocessing for DeepSense6G scenarios 31-34.
# Usage:  PY=~/.venvs/beamprep/bin/python bash preprocessing_amber/run_preprocessing.sh
set -euo pipefail

PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step() { echo; echo "=============== $* ==============="; }

step "1/4  radar: 2-channel range-angle / range-velocity tensors (AMBER eqs. 5-8)"
"$PY" "$HERE/radar_ra_rv.py" --split all

step "2/4  lidar: BEV point-count histograms, capped at 5/cell (AMBER eq. 11)"
"$PY" "$HERE/lidar_bev.py" --split all

step "3/4  camera: resized frame cache + per-image statistics (AMBER eq. 10)"
"$PY" "$HERE/image_cache.py" --split all

step "4/4  index: sequences, GPS Cartesian, beam history, availability mask, splits"
"$PY" "$HERE/build_index.py"

echo; echo "Done. Index: data/processed_amber/index/samples.csv"
