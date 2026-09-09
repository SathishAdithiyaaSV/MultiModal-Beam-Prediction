"""Build the unified, reproducible sample index for scenarios 31-34.

Consumes the two official csvs plus the outputs of preprocess_radar.py /
preprocess_lidar.py and emits a single tidy index in data/processed/index/:

  samples.csv          one row per sample; raw + processed paths, scenario,
                       0-indexed beam label, difficulty metrics, split
  beam_pwr.npy         (N, 64) float32 matrix of the 64-beam power vectors,
                       row i == samples.csv row i  (`pwr_row` column)
  split_summary.csv    per split x scenario counts
  meta.json            provenance: config values, seed, per-scenario status

Beam labels: the official `unit1_beam` column is 1-indexed; `beam` here is
0-indexed (argmax of the power vector) and is asserted to agree with it.

Difficulty metrics -- computed from the power vector, i.e. from the TARGET.
They exist only for offline analysis of per-sample beam ambiguity and must
never be fed to a model as an input feature:
  margin_db        10*log10(P_best / P_second_best)   -- small => ambiguous
  entropy_bits     Shannon entropy of P/sum(P) in bits (0 .. 6)
  n_within_3db     number of beams with power >= P_best / 2  (>=1)
  n_within_10pct   number of beams with power >= 0.9 * P_best (>=1)

Caveat, measured on this data: the whole 64-beam power vector spans only about
2.6 dB (scenario 32) to 5.6 dB (scenario 33) between best and worst beam, so
`n_within_3db` saturates at 64 for most scenario-32 samples and is close to
useless there. `margin_db`, `entropy_bits` and `n_within_10pct` stay
discriminative and should be preferred for the ambiguity analysis.

Corrupt power vectors: 58 development samples have literal `nan` entries in
their mmWave_power_*.txt file. For every one of them the official `unit1_beam`
equals the index of the FIRST NaN rather than the argmax of the finite values,
i.e. the official label was produced by `np.argmax` on a NaN-containing vector
and is meaningless. Those samples get `beam` = `nanargmax` (the defensible
label), keep the official value in `beam_official`, are marked by `pwr_n_nan`,
and are routed to the `excluded_nan_pwr` split so they never enter
train/val/test. Pass --keep-nan-pwr to include them anyway.

Splits: contiguous blocks of `--block` consecutive frames are assigned whole to
train/val/test with a seeded shuffle. Consecutive DeepSense6G samples overlap in
time and are strongly correlated, so a per-sample random split would leak the
test set into training. The adaptation set is never split; it is held out whole
as its own `adaptation` split.

Run:  python preprocessing/build_index.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

RGB = [f"unit1_rgb_{i}" for i in range(1, 6)]
RADAR = [f"unit1_radar_{i}" for i in range(1, 6)]
LIDAR = [f"unit1_lidar_{i}" for i in range(1, 6)]
LOC2 = ["unit2_loc_1", "unit2_loc_2"]
REQUIRED = RGB + RADAR + LIDAR + ["unit1_loc"] + LOC2 + ["unit1_pwr_60ghz"]

_FRAME = re.compile(r"_(\d+)\.[a-z]+$")


def frame_of(p):
    return int(_FRAME.search(str(p)).group(1))


def difficulty(pwr):
    """Per-sample beam-ambiguity measures for a (N, 64) power matrix.

    NaN entries (58 corrupt development samples) are ignored rather than
    propagated, so the measures describe the finite beams only.
    """
    p = np.where(np.isfinite(pwr), pwr.astype(np.float64), np.nan)
    p = np.clip(p, 1e-12, None)
    srt = np.sort(p, axis=1)                       # NaNs sort to the end
    n_ok = np.isfinite(p).sum(axis=1)
    best = np.nanmax(p, axis=1)
    second = srt[np.arange(len(p)), np.maximum(n_ok - 2, 0)]
    margin_db = 10.0 * np.log10(best / second)
    q = p / np.nansum(p, axis=1, keepdims=True)
    entropy_bits = -np.nansum(q * np.log2(q), axis=1)
    n_within_3db = (p >= (best[:, None] / 10 ** 0.3)).sum(axis=1)
    n_within_10pct = (p >= 0.9 * best[:, None]).sum(axis=1)
    return margin_db, entropy_bits, n_within_3db, n_within_10pct


def scenario_status(root, g):
    """Which required columns are fully present on disk for this scenario."""
    return {c: sum(1 for p in g[c] if not (root / p).exists()) for c in REQUIRED}


def collect(split, allow_incomplete):
    root, csv = config.SPLITS[split]
    df = pd.read_csv(csv)
    df["scenario"] = df["unit1_rgb_1"].str.extract(r"(scenario\d+)")
    kept, status = [], {}

    for scn, g in df.groupby("scenario", sort=True):
        miss = {c: n for c, n in scenario_status(root, g).items() if n}
        status[scn] = {"n_samples": int(len(g)), "missing": miss}
        if miss and not allow_incomplete:
            mods = sorted({("radar" if "radar" in c else "lidar" if "lidar" in c
                            else "rgb" if "rgb" in c else "gps" if "loc" in c else "pwr")
                           for c in miss})
            print(f"  !! {split}/{scn}: EXCLUDED -- incomplete raw data "
                  f"({', '.join(mods)} missing). Re-download this scenario, or pass "
                  f"--allow-incomplete to index it anyway.")
            status[scn]["included"] = False
            continue
        status[scn]["included"] = True
        kept.append(g)

    if not kept:
        return pd.DataFrame(), status

    out = pd.concat(kept, ignore_index=True)
    out["split_src"] = split
    out["frame"] = out["unit1_rgb_1"].map(frame_of)
    out["sample_id"] = out["split_src"] + "/" + out["scenario"] + "/" + out["frame"].astype(str)
    return out, status


def resolve_paths(rows, split):
    """Add processed-modality paths (relative to repo root) and GPS values."""
    root = config.SPLITS[split][0]
    rel = lambda p: str(Path(p).relative_to(config.ROOT))

    for i in range(1, 6):
        rows[f"radar_ang_{i}"] = [
            rel(config.radar_out(split, s, "ang") / Path(p).name)
            for s, p in zip(rows["scenario"], rows[f"unit1_radar_{i}"])]
        rows[f"radar_vel_{i}"] = [
            rel(config.radar_out(split, s, "vel") / Path(p).name)
            for s, p in zip(rows["scenario"], rows[f"unit1_radar_{i}"])]
        rows[f"lidar_fg_{i}"] = [
            rel(config.lidar_out(split, s) / Path(p).name)
            for s, p in zip(rows["scenario"], rows[f"unit1_lidar_{i}"])]
        rows[f"rgb_{i}"] = [rel(root / p) for p in rows[f"unit1_rgb_{i}"]]

    # GPS: unit1 is a single static basestation position per scenario
    u1 = {}
    for s, p in zip(rows["scenario"], rows["unit1_loc"]):
        if s not in u1:
            u1[s] = np.loadtxt(root / p)
    rows["bs_lat"] = [u1[s][0] for s in rows["scenario"]]
    rows["bs_lon"] = [u1[s][1] for s in rows["scenario"]]
    for k, col in enumerate(LOC2, start=1):
        v = np.array([np.loadtxt(root / p) for p in rows[col]])
        rows[f"ue_lat_{k}"], rows[f"ue_lon_{k}"] = v[:, 0], v[:, 1]
    return rows


def load_pwr(rows, split):
    root = config.SPLITS[split][0]
    return np.stack([np.loadtxt(root / p) for p in rows["unit1_pwr_60ghz"]]).astype(np.float32)


def assign_splits(rows, block, seed, fracs, keep_nan_pwr=False):
    """Contiguous-block train/val/test assignment inside the development split."""
    rng = np.random.default_rng(seed)
    rows["split"] = "adaptation"
    if not keep_nan_pwr:
        rows.loc[rows["pwr_n_nan"] > 0, "split"] = "excluded_nan_pwr"
    dev = (rows["split_src"] == "development") & (rows["split"] != "excluded_nan_pwr")
    if not dev.any():
        return rows

    for scn in sorted(rows.loc[dev, "scenario"].unique()):
        m = dev & (rows["scenario"] == scn)
        idx = rows.index[m][np.argsort(rows.loc[m, "frame"].to_numpy(), kind="stable")]
        blocks = [idx[i:i + block] for i in range(0, len(idx), block)]
        order = rng.permutation(len(blocks))
        n_tr = int(round(fracs[0] * len(blocks)))
        n_va = int(round(fracs[1] * len(blocks)))
        for rank, b in enumerate(order):
            name = "train" if rank < n_tr else "val" if rank < n_tr + n_va else "test"
            rows.loc[blocks[b], "split"] = name
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=50,
                    help="contiguous frames per split block (leakage control)")
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--keep-nan-pwr", action="store_true",
                    help="keep the 58 samples whose power vector contains NaN bins")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="index scenarios with missing raw modalities (NOT recommended)")
    a = ap.parse_args()

    frames, pwrs, status = [], [], {}
    for split in config.SPLITS:
        print(f"[{split}]")
        rows, st = collect(split, a.allow_incomplete)
        status[split] = st
        if rows.empty:
            continue
        rows = resolve_paths(rows, split)
        pwr = load_pwr(rows, split)
        n_nan = (~np.isfinite(pwr)).sum(axis=1)
        beam = np.nanargmax(np.where(np.isfinite(pwr), pwr, -np.inf), axis=1)
        official = rows["unit1_beam"].to_numpy() - 1
        clean = n_nan == 0
        assert (beam[clean] == official[clean]).all(), \
            f"{split}: argmax(pwr) disagrees with unit1_beam on NaN-free samples"
        if (~clean).any():
            print(f"  !! {split}: {int((~clean).sum())} samples have NaN power bins; "
                  f"their official unit1_beam is the first-NaN index, not a real beam "
                  f"-> relabelled with nanargmax and routed to 'excluded_nan_pwr'")
        rows["beam"] = beam
        rows["beam_official"] = official
        rows["pwr_n_nan"] = n_nan
        rows["pwr_best"] = np.nanmax(np.where(np.isfinite(pwr), pwr, -np.inf), axis=1)
        m, e, n3, n10 = difficulty(pwr)
        (rows["margin_db"], rows["entropy_bits"],
         rows["n_within_3db"], rows["n_within_10pct"]) = m, e, n3, n10
        frames.append(rows)
        pwrs.append(pwr)
        print(f"  indexed {len(rows)} samples "
              f"({', '.join(sorted(rows.scenario.unique()))})")

    if not frames:
        print("\nNothing could be indexed."); return 1

    rows = pd.concat(frames, ignore_index=True)
    pwr = np.concatenate(pwrs, axis=0)
    rows["pwr_row"] = np.arange(len(rows))
    rows = assign_splits(rows, a.block, a.seed, (a.train_frac, a.val_frac), a.keep_nan_pwr)

    keep = (["sample_id", "split_src", "scenario", "frame", "split", "beam", "beam_official",
             "pwr_row", "pwr_n_nan", "pwr_best", "margin_db", "entropy_bits",
             "n_within_3db", "n_within_10pct", "bs_lat", "bs_lon"]
            + [f"ue_lat_{k}" for k in (1, 2)] + [f"ue_lon_{k}" for k in (1, 2)]
            + [f"rgb_{i}" for i in range(1, 6)]
            + [f"radar_ang_{i}" for i in range(1, 6)]
            + [f"radar_vel_{i}" for i in range(1, 6)]
            + [f"lidar_fg_{i}" for i in range(1, 6)])

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows[keep].to_csv(config.INDEX_DIR / "samples.csv", index=False)
    np.save(config.INDEX_DIR / "beam_pwr.npy", pwr)

    summary = (rows.groupby(["split", "scenario"])
                   .agg(n=("sample_id", "size"),
                        beams=("beam", "nunique"),
                        margin_db_median=("margin_db", "median"),
                        entropy_bits_median=("entropy_bits", "median"),
                        n_within_3db_median=("n_within_3db", "median"),
                        n_within_10pct_median=("n_within_10pct", "median"))
                   .reset_index())
    summary.to_csv(config.INDEX_DIR / "split_summary.csv", index=False)

    (config.INDEX_DIR / "meta.json").write_text(json.dumps({
        "scope": "DeepSense6G 2022 Multi-Modal Beam Prediction, scenarios 31-34 only",
        "n_samples": int(len(rows)), "n_beams": config.N_BEAMS,
        "beam_label": "0-indexed argmax over the finite bins of the 64-dim power vector",
        "nan_power_policy": "kept" if a.keep_nan_pwr else "excluded_nan_pwr split",
        "split_policy": {"block_frames": a.block, "seed": a.seed,
                         "fracs": [a.train_frac, a.val_frac,
                                   round(1 - a.train_frac - a.val_frac, 4)],
                         "adaptation": "held out whole, never split"},
        "lidar": {"filter_distance_min": config.FILTER_DISTANCE_MIN,
                  "filter_distance_max": config.FILTER_DISTANCE_MAX,
                  "lidar_distance_cst": config.LIDAR_DISTANCE_CST,
                  "min_points": config.SCENARIO_MIN_POINTS,
                  "background_source": {k: list(v) for k, v in config.BACKGROUND_SOURCE.items()},
                  "background_max_frames": config.BACKGROUND_MAX_FRAMES},
        "radar": {"fft_size": config.RADAR_FFT_SIZE},
        "scenario_status": status,
    }, indent=2))

    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {config.INDEX_DIR}/samples.csv  ({len(rows)} rows, {len(keep)} cols)")
    print(f"wrote {config.INDEX_DIR}/beam_pwr.npy  {pwr.shape} {pwr.dtype}")
    print(f"wrote {config.INDEX_DIR}/split_summary.csv, meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
