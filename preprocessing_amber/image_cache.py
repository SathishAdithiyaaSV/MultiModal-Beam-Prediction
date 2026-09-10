"""AMBER image preprocessing.

AMBER eq. (10) standardises each image by its own mean and standard deviation
before the ResNet34 backbone:  PI( (XI - mu_I) / sigma_I ).

Per-image standardisation is a one-line operation in the data loader and storing
float32 tensors would be wasteful (960x540x3 float32 = 6.2 MB per frame, ~54 GB
for the dataset), so this step does NOT write normalised tensors. It only
caches resized uint8 JPEGs so training does not repeatedly decode
full-resolution frames, and records each cached frame's channel mean/std in
data/processed_amber/index/image_stats.csv for reference.

Note that AMBER applies no photometric augmentation to the camera stream, so
none is performed here.

Output: data/processed_amber/<source>/<scenario>/camera/image_<f>.jpg
Run:  python preprocessing_amber/image_cache.py --source all
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


def run(source, scenario, size, n_jobs, overwrite):
    src_dir = config.raw_dir(source, scenario, "camera_data")
    if not src_dir.is_dir() or not any(src_dir.glob("*.jpg")):
        print(f"[{source}/{scenario}] no camera_data -- SKIPPED")
        return []
    dst_dir = config.image_out(source, scenario)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.jpg"))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_process)(f, dst_dir, size, overwrite)
        for f in tqdm(files, desc=f"image {source}/{scenario}", unit="img"))
    done = sum(1 for r in res if r[3] == "done")
    print(f"[{source}/{scenario}] {done} written, {len(res)-done} present -> {dst_dir}")
    return [(source, scenario, n, m, s) for n, m, s, _ in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(config.SOURCES) + ["all"], default="all",
                    help="which DeepSense6G release on disk to process")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--image-size", type=int, nargs=2, default=list(config.IMAGE_SIZE),
                    metavar=("W", "H"), help="0 0 to copy at native resolution")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    size = tuple(a.image_size) if all(a.image_size) else None
    rows = []
    sources = config.available_sources() if a.source == "all" else [a.source]
    for source in sources:
        if config.source_csv(source) is None:
            print(f"[{source}] not present in data/raw -- SKIPPED")
            continue
        for scn in (config.source_scenarios(source) if a.scenario == "all" else [a.scenario]):
            rows += run(source, scn, size, a.n_jobs, a.overwrite)
    if rows:
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        out = config.INDEX_DIR / "image_stats.csv"
        pd.DataFrame(rows, columns=["source", "scenario", "file", "mean", "std"]).to_csv(
            out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
