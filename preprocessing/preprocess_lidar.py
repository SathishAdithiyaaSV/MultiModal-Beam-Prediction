"""LiDAR preprocessing: static-background estimation, then background removal.

Adapted from the TII reference implementation Lidar_data_preprocessing.py,
ported from Open3D to
numpy + scipy.spatial.cKDTree (Open3D publishes no wheels for Python 3.14).
The algorithm is unchanged:

Stage 1 -- background estimation (per scenario)
  Seed the background with the first admissible frame (>= SCENARIO_MIN_POINTS
  points), then, for every further admissible frame in the source directory,
  keep only background points that still have a nearest neighbour closer than
  thr(p) in that frame, replacing each survivor by the midpoint of the pair.
  Points that do not recur in every frame (i.e. moving objects) are eroded away.

Stage 2 -- background removal (per frame, all splits/scenarios)
  Drop every point whose nearest background neighbour is closer than thr(p);
  what remains is the foreground (vehicles).

  thr(p) = 0.3 + (5.0 - 0.3) * (|p_xy| / 30)**4      [metres]
  Nearest neighbours are found in 3D; the acceptance distance is measured in
  the XY plane only -- exactly as in the reference implementation.

Deviations from the reference, both deliberate and documented:
  * frames are visited in sorted order (Open3D version used os.listdir order),
    making the background deterministic and reproducible;
  * the number of background-estimation frames is capped at
    config.BACKGROUND_MAX_FRAMES -- TII estimated the background from the small
    challenge test split, which is not part of this dataset copy;
  * per-point Python loops are replaced by batched cKDTree queries (same result,
    ~3 orders of magnitude faster).

Run:  python preprocessing/preprocess_lidar.py --stage background
      python preprocessing/preprocess_lidar.py --stage filter
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ply_io


def threshold(xyz):
    """Distance-dependent background-rejection threshold, per point."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    return config.FILTER_DISTANCE_MIN + (
        config.FILTER_DISTANCE_MAX - config.FILTER_DISTANCE_MIN
    ) * (r / config.LIDAR_DISTANCE_CST) ** 4


def _nn_xy_distance(query, reference):
    """For each query point, the XY distance to its 3D nearest reference point."""
    _, idx = cKDTree(reference).query(query, k=1, workers=-1)
    nn = reference[idx]
    return np.hypot(query[:, 0] - nn[:, 0], query[:, 1] - nn[:, 1]), nn


# ------------------------------------------------------------------ stage 1
def estimate_background(scenario, max_frames, verbose=True):
    split, scn = config.BACKGROUND_SOURCE[scenario]
    src = config.raw_dir(split, scn, "lidar_data")
    min_points = config.SCENARIO_MIN_POINTS[scenario]
    files = sorted(src.glob("*.ply"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"{scenario}: no .ply files in {src}")

    admissible = []
    seed = None
    n_corrupt = 0
    for f in files:
        try:
            xyz = ply_io.read_ply(f)
        except (ply_io.TruncatedPLYError, ValueError):
            n_corrupt += 1
            continue
        if len(xyz) < min_points:
            continue
        if seed is None:
            seed = xyz
        else:
            admissible.append(f)
        if len(admissible) >= max_frames:
            break
    if seed is None:
        raise RuntimeError(
            f"{scenario}: no frame in {src} reaches min_points={min_points}")

    background = seed
    if verbose:
        print(f"[{scenario}] background source={split}/{scn}  min_points={min_points}  "
              f"seed={len(background)} pts  refine over {len(admissible)} frames"
              + (f"  ({n_corrupt} unreadable, skipped)" if n_corrupt else ""))

    for f in tqdm(admissible, desc=f"lidar-bg {scenario}", unit="frm", disable=not verbose):
        try:
            frame = ply_io.read_ply(f)
        except (ply_io.TruncatedPLYError, ValueError):
            continue
        dist, nn = _nn_xy_distance(background, frame)
        keep = dist < threshold(background)
        background = (background[keep] + nn[keep]) / 2.0
        if len(background) == 0:
            raise RuntimeError(f"{scenario}: background eroded to zero points at {f.name}")

    config.BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
    out = config.BACKGROUND_DIR / f"{scenario}_background.ply"
    ply_io.write_ply(out, background)
    print(f"[{scenario}] background = {len(background)} pts -> {out}")
    return out


# ------------------------------------------------------------------ stage 2
def _filter_one(src, dst_dir, background, overwrite):
    dst = dst_dir / src.name
    if not overwrite and dst.exists():
        return "skip"
    try:
        xyz = ply_io.read_ply(src)
    except (ply_io.TruncatedPLYError, ValueError) as exc:
        # A handful of DeepSense6G .ply files are truncated on download. Report
        # and skip rather than aborting the whole scenario.
        return f"corrupt:{src.name}:{exc}"
    if len(xyz) == 0:
        ply_io.write_ply(dst, xyz)
        return "done"
    dist, _ = _nn_xy_distance(xyz, background)
    ply_io.write_ply(dst, xyz[dist >= threshold(xyz)])
    return "done"


def filter_split(split, scenario, n_jobs, overwrite):
    bg_path = config.BACKGROUND_DIR / f"{scenario}_background.ply"
    if not bg_path.exists():
        print(f"[{split}/{scenario}] no background yet -- run --stage background first")
        return
    src_dir = config.raw_dir(split, scenario, "lidar_data")
    files = sorted(src_dir.glob("*.ply")) if src_dir.is_dir() else []
    if not files:
        print(f"[{split}/{scenario}] no lidar_data -- SKIPPED")
        return
    background = ply_io.read_ply(bg_path)
    dst_dir = config.lidar_out(split, scenario)
    dst_dir.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_filter_one)(f, dst_dir, background, overwrite)
        for f in tqdm(files, desc=f"lidar {split}/{scenario}", unit="frm")
    )
    corrupt = [r for r in res if r.startswith("corrupt:")]
    print(f"[{split}/{scenario}] {res.count('done')} written, {res.count('skip')} already present, "
          f"{len(corrupt)} unreadable -> {dst_dir}")
    for c in corrupt:
        print(f"   CORRUPT {c.split(':', 2)[1]} -- re-download this file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["background", "filter", "all"], default="all")
    ap.add_argument("--split", choices=list(config.SPLITS) + ["all"], default="all")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--max-frames", type=int, default=config.BACKGROUND_MAX_FRAMES)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if a.stage in ("background", "all"):
        scns = config.SCENARIOS if a.scenario == "all" else [a.scenario]
        for scn in scns:
            src = config.raw_dir(*config.BACKGROUND_SOURCE[scn], "lidar_data")
            if not src.is_dir() or not any(src.glob("*.ply")):
                print(f"[{scn}] background source {src} unavailable -- SKIPPED")
                continue
            estimate_background(scn, a.max_frames)

    if a.stage in ("filter", "all"):
        for split in (list(config.SPLITS) if a.split == "all" else [a.split]):
            scns = config.SPLIT_SCENARIOS[split] if a.scenario == "all" else [a.scenario]
            for scn in scns:
                filter_split(split, scn, a.n_jobs, a.overwrite)


if __name__ == "__main__":
    main()
