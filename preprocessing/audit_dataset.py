"""Step 0/3 sanity check: verify the raw DeepSense6G scenarios 31-34 on disk.

Checks, per split and per scenario:
  * every path referenced by the official index csv exists;
  * modality shapes/dtypes are the expected ones;
  * the 5-observation camera/LiDAR/radar sequences and the 2 GPS samples are
    temporally ordered and aligned on a common frame index;
  * `unit1_beam` is a valid 1..64 codebook index and equals argmax of the
    64-dim mmWave power vector it points at.

Run:  python preprocessing/audit_dataset.py
"""
import argparse
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config
import ply_io

RGB = [f"unit1_rgb_{i}" for i in range(1, 6)]
RADAR = [f"unit1_radar_{i}" for i in range(1, 6)]
LIDAR = [f"unit1_lidar_{i}" for i in range(1, 6)]
LOC2 = ["unit2_loc_1", "unit2_loc_2"]
PATH_COLS = RGB + RADAR + LIDAR + ["unit1_loc"] + LOC2 + ["unit1_pwr_60ghz"]

_FRAME = re.compile(r"_(\d+)\.[a-z]+$")


def frame_of(path):
    m = _FRAME.search(str(path))
    return int(m.group(1)) if m else None


def load_index(split):
    root, csv = config.SPLITS[split]
    df = pd.read_csv(csv)
    df["scenario"] = df["unit1_rgb_1"].str.extract(r"(scenario\d+)")
    return root, df


def audit_split(split, n_shape_probe=5, quick=False):
    root, df = load_index(split)
    print(f"\n{'='*78}\n{split}  ({config.SPLITS[split][1].name})   rows={len(df)}\n{'='*78}")
    ok_overall = True

    for scenario, g in df.groupby("scenario", sort=True):
        missing = Counter()
        for col in PATH_COLS:
            missing[col] = sum(1 for p in g[col] if not (root / p).exists())
        missing = {k: v for k, v in missing.items() if v}

        complete_cols = [c for c in PATH_COLS if c not in missing]
        status = "COMPLETE" if not missing else "INCOMPLETE"
        print(f"\n-- {scenario}: n={len(g)}  [{status}]")

        if missing:
            ok_overall = False
            by_mod = {}
            for c, v in missing.items():
                mod = ("radar" if "radar" in c else "lidar" if "lidar" in c
                       else "rgb" if "rgb" in c else "gps" if "loc" in c else "pwr")
                by_mod.setdefault(mod, []).append(f"{c}:{v}")
            for mod, items in by_mod.items():
                print(f"   MISSING {mod:<6} {', '.join(items)}")

        # ---- temporal alignment across modalities present for this scenario
        seq_groups = [("rgb", RGB), ("radar", RADAR), ("lidar", LIDAR)]
        ref = None
        for name, cols in seq_groups:
            if any(c in missing for c in cols):
                continue
            fr = np.stack([g[c].map(frame_of).to_numpy() for c in cols], axis=1)
            if not (np.diff(fr, axis=1) > 0).all():
                print(f"   FAIL  {name}: 5-frame sequence is not strictly increasing")
                ok_overall = False
            if ref is None:
                ref = (name, fr)
            elif not (ref[1] == fr).all():
                print(f"   FAIL  {name}: frame indices differ from {ref[0]}")
                ok_overall = False
        if ref is not None:
            step = np.unique(np.diff(ref[1], axis=1))
            print(f"   frames aligned across {', '.join(n for n,c in seq_groups if not any(x in missing for x in c))}"
                  f"; intra-sequence step(s)={step.tolist()}")
            if not all(c in missing for c in LOC2):
                gps_fr = np.stack([g[c].map(frame_of).to_numpy() for c in LOC2], axis=1)
                if (gps_fr == ref[1][:, :2]).all():
                    print("   GPS unit2_loc_1/2 align with observations 1-2")
                else:
                    print("   FAIL  GPS unit2_loc_1/2 do not align with observations 1-2")
                    ok_overall = False

        # ---- labels vs power vectors
        beams = g["unit1_beam"].to_numpy()
        if beams.min() < 1 or beams.max() > config.N_BEAMS:
            print(f"   FAIL  unit1_beam out of 1..{config.N_BEAMS}: [{beams.min()},{beams.max()}]")
            ok_overall = False
        if "unit1_pwr_60ghz" not in missing:
            bad = 0
            lens = set()
            n_nan = 0
            label_is_nan_bin = 0
            spread_db = []
            for p, b in zip(g["unit1_pwr_60ghz"], beams):
                v = np.loadtxt(root / p)
                lens.add(v.shape)
                if int(np.argmax(v)) + 1 != int(b):
                    bad += 1
                nn = ~np.isfinite(v)
                if nn.any():
                    n_nan += 1
                    if int(np.where(nn)[0][0]) == int(b) - 1:
                        label_is_nan_bin += 1
                else:
                    spread_db.append(10 * np.log10(v.max() / max(v.min(), 1e-12)))
            print(f"   pwr vector shapes={ {s for s in lens} }  "
                  f"argmax(pwr)+1 == unit1_beam for {len(g)-bad}/{len(g)} samples")
            if spread_db:
                print(f"   pwr best/worst beam spread: median {np.median(spread_db):.2f} dB "
                      f"(p5 {np.percentile(spread_db,5):.2f}, p95 {np.percentile(spread_db,95):.2f})")
            if n_nan:
                ok_overall = False
                print(f"   FAIL  {n_nan} power vector(s) contain NaN bins; for "
                      f"{label_is_nan_bin}/{n_nan} of them unit1_beam is the first-NaN "
                      f"index, i.e. the official label is not a real beam")
            if bad:
                ok_overall = False
            print(f"   beam label range=[{beams.min()},{beams.max()}] (1-indexed), "
                  f"{len(np.unique(beams))} distinct beams used")
        else:
            print(f"   beam label range=[{beams.min()},{beams.max()}] "
                  f"(1-indexed) -- power vectors ABSENT, cannot verify")

        # ---- shape probe on the first few available samples
        probe = g.head(n_shape_probe)
        shapes = {}
        for col, loader, key in (
            ("unit1_radar_1", lambda p: np.load(p), "radar"),
            ("unit1_lidar_1", ply_io.read_ply, "lidar"),
        ):
            if col in missing:
                continue
            got = []
            for p in probe[col]:
                a = loader(root / p)
                got.append((a.shape, str(a.dtype)))
            shapes[key] = got

        # every .ply on disk must be readable (some downloads arrive truncated)
        if quick:
            print("   .ply readability scan skipped (--quick)")
        else:
            ply_dir = root / scenario / "unit1" / "lidar_data"
            unreadable = []
            if ply_dir.is_dir():
                for p in sorted(ply_dir.glob("*.ply")):
                    try:
                        ply_io.read_ply(p)
                    except Exception as exc:
                        unreadable.append((p.name, exc))
            if unreadable:
                ok_overall = False
                print(f"   FAIL  {len(unreadable)} unreadable/truncated .ply file(s): "
                      f"{', '.join(n for n, _ in unreadable[:5])}"
                      f"{' ...' if len(unreadable) > 5 else ''}")
            else:
                print(f"   all .ply files in lidar_data readable")
        for k, v in shapes.items():
            uniq = {s for s, _ in v}
            dt = {d for _, d in v}
            print(f"   {k}: shape(s)={sorted(uniq)} dtype={dt}")
        if "unit1_rgb_1" not in missing:
            from PIL import Image
            sizes = {Image.open(root / p).size for p in probe["unit1_rgb_1"]}
            print(f"   rgb: size(s)={sizes}")
        if "unit1_loc" not in missing:
            u1 = np.loadtxt(root / probe["unit1_loc"].iloc[0])
            print(f"   unit1_loc (static bs): {u1.tolist()}")
        if "unit2_loc_1" not in missing:
            u2 = np.loadtxt(root / probe["unit2_loc_1"].iloc[0])
            print(f"   unit2_loc_1 sample:    {u2.tolist()}")

    return ok_overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS) + ["all"], default="all")
    ap.add_argument("--quick", action="store_true",
                    help="skip the exhaustive .ply readability scan (the slow part)")
    a = ap.parse_args()
    splits = list(config.SPLITS) if a.split == "all" else [a.split]
    # list, not a generator: never short-circuit, always audit every split
    results = [audit_split(s, quick=a.quick) for s in splits]
    ok = all(results)
    print(f"\n{'='*78}\nAUDIT: {'all checks passed' if ok else 'PROBLEMS FOUND (see FAIL/MISSING above)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
