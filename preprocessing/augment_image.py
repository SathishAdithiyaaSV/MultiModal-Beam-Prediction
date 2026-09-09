"""Camera augmentation: 7 photometric variants per image.

Adapted from the TII reference implementation Image_data_augmentation.py. The reference
used torchvision.transforms.functional.adjust_*; for PIL inputs those calls are
thin wrappers around PIL.ImageEnhance / PIL.ImageFilter, so the same operations
are applied here directly through PIL, keeping torch out of the preprocessing
dependency set. Factor ranges are unchanged.

Variants (suffix _1 .. _7): brightness, contrast, gamma, hue, saturation,
sharpness, gaussian blur.

Output: data/processed/<split>_aug/<scenario>/camera/image_<f>_<k>.jpg

Run:  python preprocessing/augment_image.py --split adaptation
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def adjust_gamma(img, gamma, gain=1.0):
    """Equivalent to torchvision.transforms.functional.adjust_gamma on a PIL image."""
    a = np.asarray(img).astype(np.float32) / 255.0
    a = np.clip(gain * a ** gamma, 0.0, 1.0)
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode=img.mode)


def adjust_hue(img, factor):
    """Equivalent to torchvision.transforms.functional.adjust_hue on a PIL image."""
    mode = img.mode
    h, s, v = img.convert("HSV").split()
    # widen before the shift: the hue channel wraps around, and numpy 2 refuses
    # to add a negative Python int to a uint8 array
    shifted = (np.asarray(h).astype(np.int16) + int(factor * 255)) % 256
    h = Image.fromarray(shifted.astype(np.uint8), mode="L")
    return Image.merge("HSV", (h, s, v)).convert(mode)


def variants(img, rng):
    return [
        ImageEnhance.Brightness(img).enhance(rng.uniform(0.5, 3)),      # _1
        ImageEnhance.Contrast(img).enhance(rng.uniform(0.5, 4)),        # _2
        adjust_gamma(img, rng.uniform(0.5, 3)),                         # _3
        adjust_hue(img, rng.uniform(-0.5, 0.5)),                        # _4
        ImageEnhance.Color(img).enhance(rng.uniform(0, 4)),             # _5
        ImageEnhance.Sharpness(img).enhance(rng.uniform(0, 10)),        # _6
        img.filter(ImageFilter.GaussianBlur(radius=4)),                 # _7 (kernel 9x7, sigma 3-5)
    ]


def run(split, scenario, seed, overwrite):
    src = config.raw_dir(split, scenario, "camera_data")
    if not src.is_dir():
        print(f"[{split}/{scenario}] no camera_data -- SKIPPED")
        return
    dst = config.image_out(split, scenario)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.jpg"))
    rng = random.Random(seed)
    n = 0
    for f in tqdm(files, desc=f"image-aug {split}/{scenario}", unit="img"):
        outs = [dst / f"{f.stem}_{k}.jpg" for k in range(1, 8)]
        if not overwrite and all(o.exists() for o in outs):
            continue
        with Image.open(f) as img:
            img = img.convert("RGB")
            for o, v in zip(outs, variants(img, rng)):
                v.save(o, "JPEG")
        n += 1
    print(f"[{split}/{scenario}] augmented {n} images x7 -> {dst}")


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
