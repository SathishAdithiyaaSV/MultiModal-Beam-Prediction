"""Configuration for the AMBER-style preprocessing of DeepSense6G scenarios 31-34.

Implements the representations described in
  "AMBER: An Adaptive Multimodal Mask Transformer for Beam Prediction with
   Missing Modalities" (Wen, Shi, Li, Zhao, Zhao, Wang)
which differ substantially from the TII challenge pipeline in ../preprocessing:

  radar  2D-FFT range-angle + range-velocity maps stacked as TWO CHANNELS of a
         single tensor with ONE joint min-max normalisation, and NO static
         clutter removal          (AMBER eqs. 5-8)
  lidar  bird's-eye-view point-count histogram, capped at 5 points per cell,
         instead of KD-tree background subtraction      (AMBER eq. 11)
  gps    lat/lon converted to Cartesian metres relative to the BS, then
         min-max normalised                              (AMBER eq. 12)
  beam   the W-1 historical beam indices are an INPUT modality  (AMBER eq. 13)
  image  per-image mean/std standardisation, no photometric augmentation
                                                          (AMBER eq. 10)

Values the paper does not state numerically are marked PAPER-UNSPECIFIED below;
each is a documented choice of ours, exposed as a CLI flag.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed_amber"
INDEX_DIR = PROCESSED / "index"

# ply_io lives in the sibling TII pipeline; reuse it rather than duplicating it
sys.path.insert(0, str(ROOT / "preprocessing"))

SPLITS = {
    "development": (RAW / "development", RAW / "development" / "ml_challenge_dev_multi_modal.csv"),
    "adaptation": (RAW / "adaptation", RAW / "adaptation" / "ml_challenge_data_adaptation_multi_modal.csv"),
}
SPLIT_SCENARIOS = {
    "development": ["scenario32", "scenario33", "scenario34"],
    "adaptation": ["scenario31", "scenario32", "scenario33"],
}
SCENARIOS = ["scenario31", "scenario32", "scenario33", "scenario34"]

# ---------------------------------------------------------------- AMBER Table II
N_BEAMS = 64            # codebook size K
SLIDING_WINDOW = 5      # W: image/lidar/radar observations per sample
N_GPS = 2               # GPS observations, tau = t-1 .. t
N_BEAM_HISTORY = 4      # W-1 historical beam indices, tau = t-W+1 .. t-1

# ---------------------------------------------------------------- radar
# PAPER-UNSPECIFIED: NFFT. Both the angle axis (MR = 4 antennas) and the
# velocity axis (AR = 250 chirps) are zero-padded to NFFT, so NFFT must be
# >= 250; 256 is the smallest power of two that qualifies and matches SR = 256,
# giving square (2, 256, 256) tensors.
RADAR_NFFT = 256

# ---------------------------------------------------------------- lidar BEV
# PAPER-UNSPECIFIED: Vg, Hg and the region of interest. Measured over 150 frames
# per scenario, 99% of points fall inside +/- 47 m in x and y, so a +/- 50 m ROI
# on a 256 x 256 grid (0.39 m/cell) retains essentially all of the scene.
BEV_GRID = (256, 256)          # (Vg, Hg)
BEV_X_RANGE = (-50.0, 50.0)    # metres, LiDAR frame
BEV_Y_RANGE = (-50.0, 50.0)
BEV_Z_RANGE = None             # None = keep all z; AMBER applies no z filter
BEV_MAX_PER_CELL = 5           # "the maximum number of points per grid cell is
                               #  capped at five to reduce outlier effects"

# ---------------------------------------------------------------- image
# PAPER-UNSPECIFIED: input resolution. Raw frames are 960x540; they are cached
# resized so ResNet34 training does not re-decode full-resolution JPEGs.
# Set --image-size 0 to skip the cache and read the raw frames instead.
IMAGE_SIZE = (256, 256)        # (W, H)

# ---------------------------------------------------------------- splits
# AMBER: "the dataset is randomly divided into 80% training and 20% testing
# subsets, each corresponding to an independent vehicle pass-by event".
TRAIN_FRAC = 0.8
SPLIT_BLOCK = 50               # contiguous frames per block in 'block' mode
SPLIT_SEED = 2022
# A frame-index gap larger than this starts a new recording session
SESSION_GAP = 10


def radar_out(split, scenario):
    return PROCESSED / split / scenario / "radar_ra_rv"


def bev_out(split, scenario):
    return PROCESSED / split / scenario / "lidar_bev"


def image_out(split, scenario):
    return PROCESSED / split / scenario / "camera"


def raw_dir(split, scenario, modality, unit="unit1"):
    return SPLITS[split][0] / scenario / unit / modality
