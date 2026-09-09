"""LiDAR augmentation: 2 variants per background-filtered point cloud.

Adapted from the TII reference implementation Lidar_data_augmentation.py
(Open3D -> numpy).
Operates on the OUTPUT of preprocess_lidar.py, as in the reference.

  _1  random_down_sample(0.9)   -- keep a random 90% of points
  _2  additive uniform jitter of +/- 0.4 m independently on x, y, z

Output: data/processed/<split>_aug/<scenario>/lidar/lidar_data_<f>_<k>.ply

Run:  python preprocessing/augment_lidar.py --split adaptation
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ply_io


def run(split, scenario, seed, overwrite):
    src = config.lidar_out(split, scenario)
    if not src.is_dir() or not any(src.glob("*.ply")):
        print(f"[{split}/{scenario}] no preprocessed lidar -- run preprocess_lidar.py first")
        return
    dst = config.lidar_out(split, scenario, aug=True)
    dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    files = sorted(src.glob("*.ply"))
    n = 0
    for f in tqdm(files, desc=f"lidar-aug {split}/{scenario}", unit="frm"):
        o1, o2 = dst / f"{f.stem}_1.ply", dst / f"{f.stem}_2.ply"
        if not overwrite and o1.exists() and o2.exists():
            continue
        xyz = ply_io.read_ply(f)
        keep = rng.random(len(xyz)) < config.LIDAR_AUG_KEEP_RATIO
        ply_io.write_ply(o1, xyz[keep])
        nr = config.LIDAR_AUG_NOISE_RANGE
        ply_io.write_ply(o2, xyz + rng.uniform(-nr, nr, size=xyz.shape))
        n += 1
    print(f"[{split}/{scenario}] augmented {n} clouds x2 -> {dst}")


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
