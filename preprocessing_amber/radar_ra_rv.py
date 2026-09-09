"""AMBER radar preprocessing: 2-channel range-angle / range-velocity tensor.

Implements AMBER eqs. (5)-(8). Given the raw radar cube
XR[t] in R^{MR x SR x AR} (4 antennas x 256 samples/chirp x 250 chirps):

  RA[t] = ( sum_a |FFT2D( XR[t][:, :, a] )| )^T          -> (SR, NFFT)
  RV[t] =   sum_a |FFT2D( XR[t][a, :, :] )|              -> (SR, NFFT)

The angle axis (antennas) and the velocity axis (chirps) are zero-padded to
NFFT before the transform so both maps share the spatial size SR x NFFT. The
two maps are then stacked as two channels of one tensor and normalised with a
SINGLE joint min-max, not one per map:

  Xbar[t] = Concat(RA, RV) in R^{2 x SR x NFFT}
  Xtilde  = (Xbar - min(Xbar)) / (max(Xbar) - min(Xbar))

Differences from the TII pipeline in ../preprocessing/preprocess_radar.py:
  * no static clutter removal -- TII subtracts the per-antenna chirp mean before
    the angle FFT, AMBER eq. (5) has no such term;
  * a genuine 2D FFT rather than two successive 1D FFTs;
  * one 2-channel file with a joint normalisation, instead of two separate
    files each normalised on its own.

Output: data/processed_amber/<split>/<scenario>/radar_ra_rv/radar_data_<f>.npy
        (2, 256, NFFT) float32 in [0, 1]

Run:  python preprocessing_amber/radar_ra_rv.py --split all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def range_angle_map(cube, nfft):
    """AMBER eq. (5), first line. cube: (MR, SR, AR) -> (SR, nfft)."""
    spec = np.fft.fft2(cube, s=(nfft, cube.shape[1]), axes=(0, 1))
    return np.abs(spec).sum(axis=2).T


def range_velocity_map(cube, nfft):
    """AMBER eq. (5), second line. cube: (MR, SR, AR) -> (SR, nfft)."""
    spec = np.fft.fft2(cube, s=(cube.shape[1], nfft), axes=(1, 2))
    return np.abs(spec).sum(axis=0)


def joint_minmax(x):
    """AMBER eqs. (7)-(8): one min and one max over the whole 2-channel tensor."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def build_tensor(cube, nfft):
    ra = range_angle_map(cube, nfft)
    rv = range_velocity_map(cube, nfft)
    stacked = np.stack([ra, rv], axis=0)          # AMBER eq. (6)
    return joint_minmax(stacked).astype(np.float32)


def _process(src, dst_dir, nfft, overwrite):
    dst = dst_dir / src.name
    if not overwrite and dst.exists():
        return "skip"
    np.save(dst, build_tensor(np.load(src), nfft))
    return "done"


def run(split, scenario, nfft, n_jobs, overwrite):
    src_dir = config.raw_dir(split, scenario, "radar_data")
    if not src_dir.is_dir() or not any(src_dir.glob("*.npy")):
        print(f"[{split}/{scenario}] no radar_data -- SKIPPED")
        return
    dst_dir = config.radar_out(split, scenario)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.npy"))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_process)(f, dst_dir, nfft, overwrite)
        for f in tqdm(files, desc=f"radar(AMBER) {split}/{scenario}", unit="frm"))
    print(f"[{split}/{scenario}] {res.count('done')} written, {res.count('skip')} present "
          f"-> {dst_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS) + ["all"], default="all")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--nfft", type=int, default=config.RADAR_NFFT)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for split in (list(config.SPLITS) if a.split == "all" else [a.split]):
        for scn in (config.SPLIT_SCENARIOS[split] if a.scenario == "all" else [a.scenario]):
            run(split, scn, a.nfft, a.n_jobs, a.overwrite)


if __name__ == "__main__":
    main()
