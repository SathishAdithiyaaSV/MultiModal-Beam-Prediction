"""Configuration for the AMBER-style preprocessing of DeepSense6G scenarios 31-34.

Implements the representations described in
  "AMBER: An Adaptive Multimodal Mask Transformer for Beam Prediction with
   Missing Modalities" (Wen, Shi, Li, Zhao, Zhao, Wang)
Representations produced:

  radar  2D-FFT range-angle + range-velocity maps stacked as TWO CHANNELS of a
         single tensor with ONE joint min-max normalisation, and NO static
         clutter removal                                  (AMBER eqs. 5-8)
  lidar  bird's-eye-view point-count histogram, capped at 5 points per cell
                                                          (AMBER eq. 11)
  gps    lat/lon converted to Cartesian metres relative to the BS, then
         min-max normalised                               (AMBER eq. 12)
  beam   the W-1 historical beam indices are an INPUT modality  (AMBER eq. 13)
  image  per-image mean/std standardisation, no photometric augmentation
                                                          (AMBER eq. 10)

Values the paper does not state numerically are marked PAPER-UNSPECIFIED below;
each is a documented choice of ours, exposed as a CLI flag.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed_amber"
INDEX_DIR = PROCESSED / "index"

# ---------------------------------------------------------------- data sources
# A "source" is a physical DeepSense6G release on disk. It is NOT the same thing
# as a train/val/test "split" -- see SPLIT POLICY below. Sources are discovered
# at run time, so the pipeline works before and after the official test set is
# added.
SOURCES = {
    "development": RAW / "development",
    "adaptation": RAW / "adaptation",
    "test": RAW / "test",
}

# The official index csv inside each source. The test release has shipped under
# a few different names, so several are accepted.
SOURCE_CSV_NAMES = {
    "development": ["ml_challenge_dev_multi_modal.csv"],
    "adaptation": ["ml_challenge_data_adaptation_multi_modal.csv"],
    "test": ["ml_challenge_test_multi_modal.csv",
             "ml_challenge_data_test_multi_modal.csv",
             "ml_challenge_challenge_multi_modal.csv"],
}


def source_csv(source):
    """Path to a source's index csv, or None if the source is not present."""
    root = SOURCES[source]
    if not root.is_dir():
        return None
    for name in SOURCE_CSV_NAMES[source]:
        if (root / name).exists():
            return root / name
    found = sorted(root.glob("*.csv"))
    return found[0] if found else None


def available_sources():
    """Sources that are actually on disk, in a deterministic order."""
    return [s for s in SOURCES if source_csv(s) is not None]


def source_scenarios(source):
    """Scenario directories present in a source, discovered from disk."""
    root = SOURCES[source]
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and d.name in SCENARIOS)


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

# ---------------------------------------------------------------- SPLIT POLICY
# Data SOURCE (a release on disk) and evaluation SPLIT (what a sample is used
# for) are kept strictly separate, so that no split name ever means two things:
#
#   source 'development' -> split 'train'  (fit parameters)
#                        -> split 'val'    (model selection, early stopping)
#   source 'adaptation'  -> split 'adaptation'
#                           the official 100-sample labelled set. Held out
#                           whole; a secondary in-domain generalisation check,
#                           never trained on and never used for selection.
#   source 'test'        -> split 'test'
#                           RESERVED for the official challenge test release.
#                           Nothing else is ever labelled 'test', so any number
#                           reported on 'test' is on official held-out data.
#
# Until data/raw/test/ exists the 'test' split is simply empty; adding the
# release and re-running build_index.py populates it with no other changes.
#
# AMBER states an 80/20 division, which is applied here as train/val over the
# development source.
TRAIN_FRAC = 0.8               # of the development source; remainder -> val
SPLIT_BLOCK = 50               # contiguous frames per block in 'block' mode
SPLIT_SEED = 2022
# A frame-index gap larger than this starts a new recording session
SESSION_GAP = 10

# Splits that must never be trained or selected on.
HELD_OUT_SPLITS = ("adaptation", "test")
# Splits carrying no usable target.
UNUSABLE_SPLITS = ("no_target", "excluded_nan_pwr")


def radar_out(source, scenario):
    return PROCESSED / source / scenario / "radar_ra_rv"


def bev_out(source, scenario):
    return PROCESSED / source / scenario / "lidar_bev"


def image_out(source, scenario):
    return PROCESSED / source / scenario / "camera"


def raw_dir(source, scenario, modality, unit="unit1"):
    return SOURCES[source] / scenario / unit / modality
