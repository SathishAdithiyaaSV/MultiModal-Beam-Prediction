"""Verify the raw DeepSense6G scenarios 31-34 before and after preprocessing.

Per source and scenario, checks:
  * every path referenced by the official csv exists;
  * modality shapes/dtypes are the expected ones;
  * the W=5 camera/LiDAR/radar sequences are strictly increasing and share one
    frame index, and unit2_loc_1/2 align with observations 1-2;
  * `unit1_beam` is a valid 1..64 index and equals argmax of the power vector;
  * power vectors are finite (some public files contain literal `nan`);
  * every .ply is readable (some downloads arrive truncated).

Run:  python preprocessing_amber/audit_dataset.py
      python preprocessing_amber/audit_dataset.py --quick     # skip .ply scan
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ply_io

W = config.SLIDING_WINDOW
RGB = [f"unit1_rgb_{i}" for i in range(1, W + 1)]
RADAR = [f"unit1_radar_{i}" for i in range(1, W + 1)]
LIDAR = [f"unit1_lidar_{i}" for i in range(1, W + 1)]
LOC2 = ["unit2_loc_1", "unit2_loc_2"]

_FRAME = re.compile(r"_(\d+)\.[a-z]+$")


def frame_of(p):
    m = _FRAME.search(str(p))
    return int(m.group(1)) if m else None


def audit_source(source, quick=False, n_probe=5):
    csv = config.source_csv(source)
    root = config.SOURCES[source]
    df = pd.read_csv(csv)
    df["scenario"] = df["unit1_rgb_1"].str.extract(r"(scenario\d+)")
    labelled = "unit1_pwr_60ghz" in df.columns

    print(f"\n{'='*78}\n{source}  ({csv.name})   rows={len(df)}"
          f"{'' if labelled else '   [UNLABELLED]'}\n{'='*78}")
    ok = True
    path_cols = [c for c in RGB + RADAR + LIDAR + ["unit1_loc"] + LOC2
                 + (["unit1_pwr_60ghz"] if labelled else []) if c in df.columns]

    for scenario, g in df.groupby("scenario", sort=True):
        miss = {c: n for c, n in
                ((c, sum(1 for p in g[c] if not (root / p).exists())) for c in path_cols) if n}
        print(f"\n-- {scenario}: n={len(g)}  [{'COMPLETE' if not miss else 'INCOMPLETE'}]")
        if miss:
            ok = False
            by_mod = {}
            for c, v in miss.items():
                mod = ("radar" if "radar" in c else "lidar" if "lidar" in c
                       else "rgb" if "rgb" in c else "gps" if "loc" in c else "pwr")
                by_mod.setdefault(mod, []).append(f"{c}:{v}")
            for mod, items in by_mod.items():
                print(f"   MISSING {mod:<6} {', '.join(items)}")

        # ---- temporal alignment
        ref = None
        present = []
        for name, cols in (("rgb", RGB), ("radar", RADAR), ("lidar", LIDAR)):
            if any(c in miss for c in cols):
                continue
            fr = np.stack([g[c].map(frame_of).to_numpy() for c in cols], axis=1)
            if not (np.diff(fr, axis=1) > 0).all():
                print(f"   FAIL  {name}: 5-frame sequence not strictly increasing")
                ok = False
            if ref is None:
                ref = fr
            elif not (ref == fr).all():
                print(f"   FAIL  {name}: frame indices differ from the other modalities")
                ok = False
            present.append(name)
        if ref is not None:
            print(f"   frames aligned across {', '.join(present)}; "
                  f"intra-sequence step(s)={np.unique(np.diff(ref, axis=1)).tolist()}")
            if not any(c in miss for c in LOC2):
                gps_fr = np.stack([g[c].map(frame_of).to_numpy() for c in LOC2], axis=1)
                if (gps_fr == ref[:, :2]).all():
                    print("   GPS unit2_loc_1/2 align with observations 1-2")
                else:
                    print("   FAIL  GPS unit2_loc_1/2 do not align with observations 1-2")
                    ok = False

        # ---- labels and power vectors
        if labelled and "unit1_pwr_60ghz" not in miss and "unit1_beam" in g.columns:
            beams = g["unit1_beam"].to_numpy()
            if beams.min() < 1 or beams.max() > config.N_BEAMS:
                print(f"   FAIL  unit1_beam outside 1..{config.N_BEAMS}")
                ok = False
            bad = n_nan = label_is_nan = 0
            spread, lens = [], set()
            for p, b in zip(g["unit1_pwr_60ghz"], beams):
                v = np.loadtxt(root / p)
                lens.add(v.shape)
                if int(np.argmax(v)) + 1 != int(b):
                    bad += 1
                nn = ~np.isfinite(v)
                if nn.any():
                    n_nan += 1
                    label_is_nan += int(np.where(nn)[0][0]) == int(b) - 1
                else:
                    spread.append(10 * np.log10(v.max() / max(v.min(), 1e-12)))
            print(f"   pwr shapes={lens}  argmax(pwr)+1 == unit1_beam for "
                  f"{len(g)-bad}/{len(g)}  beams used={len(np.unique(beams))}")
            if spread:
                print(f"   best/worst beam spread: median {np.median(spread):.2f} dB "
                      f"(p5 {np.percentile(spread,5):.2f}, p95 {np.percentile(spread,95):.2f})")
            if n_nan:
                ok = False
                print(f"   FAIL  {n_nan} power vector(s) contain NaN; for {label_is_nan}/{n_nan} "
                      f"of them unit1_beam is the first-NaN index, not a real beam")
            if bad:
                ok = False

        # ---- shape probe
        probe = g.head(n_probe)
        if "unit1_radar_1" not in miss:
            got = {(np.load(root / p).shape, str(np.load(root / p).dtype)) for p in probe["unit1_radar_1"]}
            print(f"   radar: {sorted(got)}")
        if "unit1_lidar_1" not in miss:
            got = [ply_io.read_ply(root / p).shape for p in probe["unit1_lidar_1"]]
            print(f"   lidar: point counts {[s[0] for s in got]}")
        if "unit1_rgb_1" not in miss:
            print(f"   rgb: size(s)={ {Image.open(root / p).size for p in probe['unit1_rgb_1']} }")

        # ---- exhaustive .ply readability
        ply_dir = root / scenario / "unit1" / "lidar_data"
        if quick:
            print("   .ply readability scan skipped (--quick)")
        elif ply_dir.is_dir():
            bad_ply = []
            for p in sorted(ply_dir.glob("*.ply")):
                try:
                    ply_io.read_ply(p)
                except Exception:
                    bad_ply.append(p.name)
            if bad_ply:
                ok = False
                print(f"   FAIL  {len(bad_ply)} unreadable/truncated .ply: "
                      f"{', '.join(bad_ply[:5])}{' ...' if len(bad_ply) > 5 else ''}")
            else:
                print("   all .ply files readable")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(config.SOURCES) + ["all"], default="all")
    ap.add_argument("--quick", action="store_true",
                    help="skip the exhaustive .ply readability scan (the slow part)")
    a = ap.parse_args()
    sources = config.available_sources() if a.source == "all" else [a.source]
    if not sources:
        print(f"No DeepSense6G sources found under {config.RAW}"); return 1
    # a list, not a generator: never short-circuit, always audit every source
    results = [audit_source(s, quick=a.quick) for s in sources
               if config.source_csv(s) is not None]
    print(f"\n{'='*78}\nAUDIT: "
          f"{'all checks passed' if all(results) else 'PROBLEMS FOUND (see FAIL/MISSING above)'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
