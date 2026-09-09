"""Radar augmentation: per-bin multiplicative jitter on the range-angle /
range-velocity maps.

Adapted from the TII reference implementation radar_data_augmentation.py. The reference's
nested Python loops are vectorised; the perturbation is identical -- each bin
value v is replaced by v + U(0.25 * 0.1v, 0.1v) before min-max normalisation.

Output: data/processed/<split>_aug/<scenario>/radar_{ang,vel}/radar_data_<f>.npy

Run:  python preprocessing/augment_radar.py --split adaptation
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from preprocess_radar import minmax, range_angle_map, range_velocity_map


def jitter(arr, rng):
    s = arr * config.RADAR_AUG_REL_SHIFT
    return arr + rng.uniform(0.25 * s, s)


def run(split, scenario, seed, overwrite):
    src = config.raw_dir(split, scenario, "radar_data")
    if not src.is_dir() or not any(src.glob("*.npy")):
        print(f"[{split}/{scenario}] no radar_data -- SKIPPED")
        return
    out_ang = config.radar_out(split, scenario, "ang", aug=True)
    out_vel = config.radar_out(split, scenario, "vel", aug=True)
    out_ang.mkdir(parents=True, exist_ok=True)
    out_vel.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 0
    for f in tqdm(sorted(src.glob("*.npy")), desc=f"radar-aug {split}/{scenario}", unit="frm"):
        da, dv = out_ang / f.name, out_vel / f.name
        if not overwrite and da.exists() and dv.exists():
            continue
        data = np.load(f)
        np.save(da, minmax(jitter(range_angle_map(data), rng)))
        np.save(dv, minmax(jitter(range_velocity_map(data), rng)))
        n += 1
    print(f"[{split}/{scenario}] augmented {n} frames -> {out_ang.parent}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS) + ["all"], default="adaptation")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for split in (list(config.SPLITS) if a.split == "all" else [a.split]):
        for scn in (config.SPLIT_SCENARIOS[split] if a.scenario == "all" else [a.scenario]):
            run(split, scn, a.seed, a.overwrite)


if __name__ == "__main__":
    main()
