"""Radar preprocessing: raw radar cube -> range-angle and range-velocity maps.

Adapted from the TII reference implementation Radar_data_preprocessing.py. The signal
processing (`range_angle_map`, `range_velocity_map`, `minmax`) is unchanged; the
hardcoded /efs paths are replaced by config.py and scenarios are iterated
automatically.

Input : data/raw/<split>/<scenario>/unit1/radar_data/radar_data_<f>.npy
        complex64, (4 antennas, 256 samples/chirp, 250 chirps)
Output: data/processed/<split>/<scenario>/radar_ang/radar_data_<f>.npy  (256, 256) float64 in [0,1]
        data/processed/<split>/<scenario>/radar_vel/radar_data_<f>.npy  (256, 256) float64 in [0,1]

Run:  python preprocessing/preprocess_radar.py            # all splits/scenarios
      python preprocessing/preprocess_radar.py --split adaptation
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def range_angle_map(data, fft_size=config.RADAR_FFT_SIZE):
    data = np.fft.fft(data, axis=1)                 # range FFT
    data = data - np.mean(data, 2, keepdims=True)   # static clutter removal
    data = np.fft.fft(data, fft_size, axis=0)       # angle FFT
    data = np.abs(data).sum(axis=2)                 # sum over chirps
    return data.T


def range_velocity_map(data, fft_size=config.RADAR_FFT_SIZE):
    data = np.fft.fft(data, axis=1)                 # range FFT
    data = np.fft.fft(data, fft_size, axis=2)       # velocity FFT
    data = np.abs(data).sum(axis=0)                 # sum over antennas
    return data


def minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def process_one(src, out_ang, out_vel, overwrite):
    dst_a, dst_v = out_ang / src.name, out_vel / src.name
    if not overwrite and dst_a.exists() and dst_v.exists():
        return "skip"
    data = np.load(src)
    np.save(dst_a, minmax(range_angle_map(data)))
    np.save(dst_v, minmax(range_velocity_map(data)))
    return "done"


def run(split, scenario, n_jobs, overwrite):
    src_dir = config.raw_dir(split, scenario, "radar_data")
    if not src_dir.is_dir():
        print(f"[{split}/{scenario}] no radar_data directory -- SKIPPED")
        return
    files = sorted(src_dir.glob("*.npy"))
    if not files:
        print(f"[{split}/{scenario}] radar_data is empty -- SKIPPED")
        return
    out_ang = config.radar_out(split, scenario, "ang")
    out_vel = config.radar_out(split, scenario, "vel")
    out_ang.mkdir(parents=True, exist_ok=True)
    out_vel.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=n_jobs)(
        delayed(process_one)(f, out_ang, out_vel, overwrite)
        for f in tqdm(files, desc=f"radar {split}/{scenario}", unit="frm")
    )
    print(f"[{split}/{scenario}] {res.count('done')} written, {res.count('skip')} already present "
          f"-> {out_ang.parent}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS) + ["all"], default="all")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for split in (list(config.SPLITS) if a.split == "all" else [a.split]):
        scns = config.SPLIT_SCENARIOS[split] if a.scenario == "all" else [a.scenario]
        for scn in scns:
            run(split, scn, a.n_jobs, a.overwrite)


if __name__ == "__main__":
    main()
