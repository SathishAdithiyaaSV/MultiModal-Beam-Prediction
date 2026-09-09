"""AMBER image preprocessing.

AMBER eq. (10) standardises each image by its own mean and standard deviation
before the ResNet34 backbone:  PI( (XI - mu_I) / sigma_I ).

Per-image standardisation is a one-line operation in the data loader and storing
float32 tensors would be wasteful (960x540x3 float32 = 6.2 MB per frame, ~54 GB
for the dataset), so this step does NOT write normalised tensors. It only
caches resized uint8 JPEGs so training does not repeatedly decode
full-resolution frames, and records each cached frame's channel mean/std in
data/processed_amber/index/image_stats.csv for reference.

Note that AMBER applies no photometric augmentation to the camera stream; the
7-variant augmentation in ../preprocessing/augment_image.py belongs to the TII
pipeline and is deliberately absent here.

Output: data/processed_amber/<split>/<scenario>/camera/image_<f>.jpg
Run:  python preprocessing_amber/image_cache.py --split all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def _process(src, dst_dir, size, overwrite):
    dst = dst_dir / src.name
    if not overwrite and dst.exists():
        with Image.open(dst) as im:
            a = np.asarray(im, dtype=np.float32) / 255.0
        return dst.name, a.mean(), a.std(), "skip"
    with Image.open(src) as im:
        im = im.convert("RGB")
        if size:
            im = im.resize(size, Image.BILINEAR)
        im.save(dst, "JPEG", quality=95)
        a = np.asarray(im, dtype=np.float32) / 255.0
    return dst.name, a.mean(), a.std(), "done"


def run(split, scenario, size, n_jobs, overwrite):
    src_dir = config.raw_dir(split, scenario, "camera_data")
    if not src_dir.is_dir() or not any(src_dir.glob("*.jpg")):
        print(f"[{split}/{scenario}] no camera_data -- SKIPPED")
        return []
    dst_dir = config.image_out(split, scenario)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.jpg"))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_process)(f, dst_dir, size, overwrite)
        for f in tqdm(files, desc=f"image {split}/{scenario}", unit="img"))
    done = sum(1 for r in res if r[3] == "done")
    print(f"[{split}/{scenario}] {done} written, {len(res)-done} present -> {dst_dir}")
    return [(split, scenario, n, m, s) for n, m, s, _ in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS) + ["all"], default="all")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--image-size", type=int, nargs=2, default=list(config.IMAGE_SIZE),
                    metavar=("W", "H"), help="0 0 to copy at native resolution")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    size = tuple(a.image_size) if all(a.image_size) else None
    rows = []
    for split in (list(config.SPLITS) if a.split == "all" else [a.split]):
        for scn in (config.SPLIT_SCENARIOS[split] if a.scenario == "all" else [a.scenario]):
            rows += run(split, scn, size, a.n_jobs, a.overwrite)
    if rows:
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        out = config.INDEX_DIR / "image_stats.csv"
        pd.DataFrame(rows, columns=["split", "scenario", "file", "mean", "std"]).to_csv(
            out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
