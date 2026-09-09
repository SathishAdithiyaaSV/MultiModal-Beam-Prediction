"""Central configuration for DeepSense6G (2022 Multi-Modal Beam Prediction) preprocessing.

Scope: scenarios 31-34 ONLY. The 2023 V2V dataset and all other DeepSense
scenarios are deliberately out of scope and must never be mixed in.

Local layout <-> official DeepSense6G release names:
    data/raw/development  == "Multi_Modal" development set        (scenarios 32, 33, 34)
    data/raw/adaptation   == "Adaptation_dataset_multi_modal"     (scenarios 31, 32, 33)
    (the challenge's unlabelled "Multi_Modal_Test" set is NOT part of this copy)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------- splits on disk
# name -> (raw scenario root, official index csv)
SPLITS = {
    "development": (RAW / "development", RAW / "development" / "ml_challenge_dev_multi_modal.csv"),
    "adaptation": (RAW / "adaptation", RAW / "adaptation" / "ml_challenge_data_adaptation_multi_modal.csv"),
}

# scenarios actually shipped in each split
SPLIT_SCENARIOS = {
    "development": ["scenario32", "scenario33", "scenario34"],
    "adaptation": ["scenario31", "scenario32", "scenario33"],
}

SCENARIOS = ["scenario31", "scenario32", "scenario33", "scenario34"]

# ---------------------------------------------------------------- outputs
BACKGROUND_DIR = PROCESSED / "lidar_background"
INDEX_DIR = PROCESSED / "index"


def lidar_out(split, scenario, aug=False):
    return PROCESSED / (split + ("_aug" if aug else "")) / scenario / "lidar"


def radar_out(split, scenario, kind, aug=False):
    """kind: 'ang' (range-angle map) or 'vel' (range-velocity map)."""
    return PROCESSED / (split + ("_aug" if aug else "")) / scenario / f"radar_{kind}"


def image_out(split, scenario):
    return PROCESSED / (split + "_aug") / scenario / "camera"


def raw_dir(split, scenario, modality):
    """modality: camera_data | lidar_data | radar_data | mmWave_data | GPS_data"""
    unit = "unit2" if modality == "GPS_data_unit2" else "unit1"
    modality = "GPS_data" if modality == "GPS_data_unit2" else modality
    return SPLITS[split][0] / scenario / unit / modality


# ---------------------------------------------------------------- LiDAR params
# Minimum #points for a frame to be admissible as a background-estimation frame.
# Values taken verbatim from the TII reference implementation.
SCENARIO_MIN_POINTS = {
    "scenario31": 16400,
    "scenario32": 18000,
    "scenario33": 18000,
    "scenario34": 18600,
}

# Which (split, scenario) directory to estimate the static LiDAR background from.
# TII used the challenge test split for 32/34; that split is not in this copy, so
# we fall back to the development split for those two scenarios.
BACKGROUND_SOURCE = {
    "scenario31": ("adaptation", "scenario31"),
    "scenario32": ("development", "scenario32"),
    "scenario33": ("adaptation", "scenario33"),
    "scenario34": ("development", "scenario34"),
}

# Distance-dependent background-rejection threshold (metres), TII eq.:
#   thr(p) = FILTER_DISTANCE_MIN + (FILTER_DISTANCE_MAX - FILTER_DISTANCE_MIN) * (|p_xy| / LIDAR_DISTANCE_CST)**4
FILTER_DISTANCE_MIN = 0.3
FILTER_DISTANCE_MAX = 5.0
LIDAR_DISTANCE_CST = 30.0

# Cap on background-estimation frames (TII iterated an entire, much smaller
# directory). Frames are taken in sorted order, so this stays deterministic.
BACKGROUND_MAX_FRAMES = 200

# ---------------------------------------------------------------- radar params
RADAR_FFT_SIZE = 256

# ---------------------------------------------------------------- augmentation
LIDAR_AUG_KEEP_RATIO = 0.9      # random_down_sample ratio
LIDAR_AUG_NOISE_RANGE = 0.4     # +/- metres, uniform, per xyz component
RADAR_AUG_REL_SHIFT = 0.1       # per-bin relative shift, U(0.25*s, s) with s = 0.1*value

# ---------------------------------------------------------------- labels
N_BEAMS = 64  # 64-beam codebook; `unit1_beam` in the official csv is 1-indexed
