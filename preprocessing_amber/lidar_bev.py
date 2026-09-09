"""AMBER LiDAR preprocessing: bird's-eye-view point-count histogram.

Implements AMBER eq. (11). The raw point cloud XL[t] in R^{NL x 3} is projected
onto a Vg x Hg BEV grid to form a histogram X^H_L[t], with the point count per
cell capped at five "to reduce outlier effects", then normalised.

This is a completely different representation from the TII pipeline in
../preprocessing/preprocess_lidar.py, which estimates a static background per
scenario and subtracts it with a KD-tree, emitting filtered .ply point clouds.
AMBER keeps every point and discards only the vertical structure, so no
background model and no per-scenario min-points threshold are involved.

Normalisation: the cap of 5 makes the histogram's range known a priori, so
cells are divided by BEV_MAX_PER_CELL. This keeps the mapping from point count
to pixel value identical across every frame and scenario -- a per-frame min-max
would rescale each frame by its own densest cell and destroy that comparability.

Output: data/processed_amber/<split>/<scenario>/lidar_bev/lidar_data_<f>.npy
        (1, Vg, Hg) float32 in [0, 1]

Run:  python preprocessing_amber/lidar_bev.py --split all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ply_io


def bev_histogram(xyz, grid=config.BEV_GRID, x_range=config.BEV_X_RANGE,
                  y_range=config.BEV_Y_RANGE, z_range=config.BEV_Z_RANGE,
                  max_per_cell=config.BEV_MAX_PER_CELL):
    """Project (N, 3) points to a normalised (1, Vg, Hg) BEV count histogram."""
    vg, hg = grid
    if z_range is not None:
        xyz = xyz[(xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])]

    # histogram2d drops out-of-range points, which is the intended ROI crop
    hist, _, _ = np.histogram2d(
        xyz[:, 0], xyz[:, 1], bins=(vg, hg), range=(x_range, y_range))
    np.clip(hist, 0, max_per_cell, out=hist)
    return (hist / max_per_cell).astype(np.float32)[None, :, :]


def _process(src, dst_dir, kwargs, overwrite):
    dst = dst_dir / (src.stem + ".npy")
    if not overwrite and dst.exists():
        return "skip"
    try:
        xyz = ply_io.read_ply(src)
    except (ply_io.TruncatedPLYError, ValueError) as exc:
        return f"corrupt:{src.name}:{exc}"
    np.save(dst, bev_histogram(xyz, **kwargs))
    return "done"


def run(split, scenario, kwargs, n_jobs, overwrite):
    src_dir = config.raw_dir(split, scenario, "lidar_data")
    if not src_dir.is_dir() or not any(src_dir.glob("*.ply")):
        print(f"[{split}/{scenario}] no lidar_data -- SKIPPED")
        return
    dst_dir = config.bev_out(split, scenario)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.ply"))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_process)(f, dst_dir, kwargs, overwrite)
        for f in tqdm(files, desc=f"bev {split}/{scenario}", unit="frm"))
    corrupt = [r for r in res if r.startswith("corrupt:")]
    print(f"[{split}/{scenario}] {res.count('done')} written, {res.count('skip')} present, "
          f"{len(corrupt)} unreadable -> {dst_dir}")
    for c in corrupt:
        print(f"   CORRUPT {c.split(':', 2)[1]} -- re-download this file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS) + ["all"], default="all")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--grid", type=int, nargs=2, default=list(config.BEV_GRID),
                    metavar=("VG", "HG"))
    ap.add_argument("--x-range", type=float, nargs=2, default=list(config.BEV_X_RANGE))
    ap.add_argument("--y-range", type=float, nargs=2, default=list(config.BEV_Y_RANGE))
    ap.add_argument("--z-range", type=float, nargs=2, default=None,
                    help="optional z filter in metres; AMBER applies none")
    ap.add_argument("--max-per-cell", type=int, default=config.BEV_MAX_PER_CELL)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    kwargs = dict(grid=tuple(a.grid), x_range=tuple(a.x_range), y_range=tuple(a.y_range),
                  z_range=tuple(a.z_range) if a.z_range else None,
                  max_per_cell=a.max_per_cell)
    for split in (list(config.SPLITS) if a.split == "all" else [a.split]):
        for scn in (config.SPLIT_SCENARIOS[split] if a.scenario == "all" else [a.scenario]):
            run(split, scn, kwargs, a.n_jobs, a.overwrite)


if __name__ == "__main__":
    main()
