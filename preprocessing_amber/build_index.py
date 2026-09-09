"""Build the AMBER-style sample index for DeepSense6G scenarios 31-34.

One row per sample, matching AMBER's input specification (Sec. II-B):

  image / lidar / radar   W = 5 observations, tau = t-W+1 .. t
  GPS                     2 observations,     tau = t-1 .. t, Cartesian metres
                          relative to the BS                       (eq. 12)
  beam history            W-1 = 4 indices,    tau = t-W+1 .. t-1   (eq. 13)
  target                  optimal beam at t = argmax of the 64-beam power vector

and AMBER's 5-element modality-availability vector m = [mI, mL, mR, mB, mG]
(eq. 3), computed from what is actually on disk. AMBER is built to train and
infer under arbitrary missing modalities, so samples are NOT dropped for having
an unavailable modality; the mask records it.

Beam history is legitimate side information, not label leakage: it uses only
tau <= t-1, while the target is at t. It is derived from the per-frame
mmWave_power_<f>.txt files, which exist independently of the csv's single
`unit1_pwr_60ghz` column.

GPS min-max normalisation (eq. 12) is fit on the TRAINING split only and the
constants are written to gps_norm.json, so val/test are transformed with train
statistics rather than their own.

Outputs in data/processed_amber/index/:
  samples.csv     one row per sample: paths, GPS metres, beam history, target,
                  availability mask, split
  beam_pwr.npy    (N, 64) float32 target power vectors
  gps_norm.json   train-fit min/max for the GPS Cartesian coordinates
  meta.json       provenance, AMBER hyperparameters, paper-unspecified choices

Run:  python preprocessing_amber/build_index.py
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

W = config.SLIDING_WINDOW
RGB = [f"unit1_rgb_{i}" for i in range(1, W + 1)]
RADAR = [f"unit1_radar_{i}" for i in range(1, W + 1)]
LIDAR = [f"unit1_lidar_{i}" for i in range(1, W + 1)]
LOC2 = ["unit2_loc_1", "unit2_loc_2"]

_FRAME = re.compile(r"_(\d+)\.[a-z]+$")


def frame_of(p):
    return int(_FRAME.search(str(p)).group(1))


def latlon_to_xy(lat, lon, lat0, lon0):
    """Local equirectangular projection to metres relative to (lat0, lon0).

    Over the <100 m extent of a DeepSense6G scene this is accurate to well under
    a centimetre, and unlike UTM it needs no zone bookkeeping.
    """
    phi = np.deg2rad(lat0)
    m_lat = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    m_lon = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi)
    return (np.asarray(lon) - lon0) * m_lon, (np.asarray(lat) - lat0) * m_lat


def beam_from_power(path):
    """(beam index, n_nan) for one mmWave_power file, or (-1, -1) if absent."""
    if not path.exists():
        return -1, -1
    v = np.loadtxt(path)
    finite = np.isfinite(v)
    if not finite.any():
        return -1, int((~finite).sum())
    return int(np.nanargmax(np.where(finite, v, -np.inf))), int((~finite).sum())


def collect(split):
    root, csv = config.SPLITS[split]
    df = pd.read_csv(csv)
    df["scenario"] = df["unit1_rgb_1"].str.extract(r"(scenario\d+)")
    df["split_src"] = split
    df["frame"] = df["unit1_rgb_1"].map(frame_of)
    df["sample_id"] = split + "/" + df["scenario"] + "/" + df["frame"].astype(str)
    return root, df


def build(split, rows, root):
    rel = lambda p: str(Path(p).relative_to(config.ROOT))
    out = {}

    # ---- per-observation modality paths (processed for radar/lidar/image)
    avail_img = np.ones(len(rows), bool)
    avail_lid = np.ones(len(rows), bool)
    avail_rad = np.ones(len(rows), bool)
    for i in range(1, W + 1):
        img = [config.image_out(split, s) / Path(p).name
               for s, p in zip(rows["scenario"], rows[RGB[i - 1]])]
        lid = [config.bev_out(split, s) / (Path(p).stem + ".npy")
               for s, p in zip(rows["scenario"], rows[LIDAR[i - 1]])]
        rad = [config.radar_out(split, s) / Path(p).name
               for s, p in zip(rows["scenario"], rows[RADAR[i - 1]])]
        out[f"image_{i}"] = [rel(p) for p in img]
        out[f"lidar_bev_{i}"] = [rel(p) for p in lid]
        out[f"radar_{i}"] = [rel(p) for p in rad]
        avail_img &= np.array([p.exists() for p in img])
        avail_lid &= np.array([p.exists() for p in lid])
        avail_rad &= np.array([p.exists() for p in rad])

    # ---- GPS -> Cartesian metres relative to the BS (AMBER eq. 12)
    bs = {}
    for s, p in zip(rows["scenario"], rows["unit1_loc"]):
        if s not in bs:
            bs[s] = np.loadtxt(root / p)
    avail_gps = np.ones(len(rows), bool)
    for k, col in enumerate(LOC2, start=1):
        xs, ys, ok = [], [], []
        for s, p in zip(rows["scenario"], rows[col]):
            f = root / p
            if not f.exists():
                xs.append(np.nan); ys.append(np.nan); ok.append(False); continue
            lat, lon = np.loadtxt(f)
            x, y = latlon_to_xy(lat, lon, bs[s][0], bs[s][1])
            xs.append(float(x)); ys.append(float(y)); ok.append(True)
        out[f"ue_x_{k}"], out[f"ue_y_{k}"] = xs, ys
        avail_gps &= np.array(ok)
    out["bs_lat"] = [bs[s][0] for s in rows["scenario"]]
    out["bs_lon"] = [bs[s][1] for s in rows["scenario"]]

    # ---- beam history: observations 1..W-1, i.e. tau = t-W+1 .. t-1
    avail_beam = np.ones(len(rows), bool)
    for i in range(1, W):
        idx, nan = [], []
        for s, p in zip(rows["scenario"], rows[LIDAR[i - 1]]):
            f = config.raw_dir(split, s, "mmWave_data") / f"mmWave_power_{frame_of(p)}.txt"
            b, n = beam_from_power(f)
            idx.append(b); nan.append(n)
        out[f"beam_hist_{i}"] = idx
        avail_beam &= np.array(idx) >= 0
    out["n_beam_hist_available"] = sum(
        (np.array(out[f"beam_hist_{i}"]) >= 0).astype(int) for i in range(1, W))

    # ---- target: beam at t from the csv's power vector
    # per-row: a scenario missing its power files must not poison the rest
    pwr = np.stack([
        np.loadtxt(root / p) if (root / p).exists() else np.full(config.N_BEAMS, np.nan)
        for p in rows["unit1_pwr_60ghz"]
    ]).astype(np.float32)
    n_nan = (~np.isfinite(pwr)).sum(axis=1)
    beam = np.where(n_nan < config.N_BEAMS,
                    np.nanargmax(np.where(np.isfinite(pwr), pwr, -np.inf), axis=1), -1)
    out["beam"] = beam
    out["beam_official"] = rows["unit1_beam"].to_numpy() - 1
    out["pwr_n_nan"] = n_nan

    # ---- AMBER availability vector m = [mI, mL, mR, mB, mG]  (eq. 3)
    out["m_image"] = avail_img.astype(int)
    out["m_lidar"] = avail_lid.astype(int)
    out["m_radar"] = avail_rad.astype(int)
    out["m_beam"] = avail_beam.astype(int)
    out["m_gps"] = avail_gps.astype(int)

    return pd.DataFrame(out, index=rows.index), pwr


def sessions(frames, gap):
    """Label contiguous recording sessions; a frame gap > `gap` starts a new one."""
    order = np.argsort(frames, kind="stable")
    lab = np.empty(len(frames), int)
    cur = 0
    lab[order[0]] = 0
    for a, b in zip(order[:-1], order[1:]):
        if frames[b] - frames[a] > gap:
            cur += 1
        lab[b] = cur
    return lab


def assign_splits(rows, mode, train_frac, block, seed, gap):
    rng = np.random.default_rng(seed)
    rows["split"] = "adaptation"
    # no usable target -> cannot train or evaluate on it
    rows.loc[rows["pwr_n_nan"] == config.N_BEAMS, "split"] = "no_target"
    # partially-NaN power vector: official label is the first-NaN index (see
    # ../README.md section 7.2), so the label is not a real beam
    rows.loc[(rows["pwr_n_nan"] > 0) & (rows["pwr_n_nan"] < config.N_BEAMS),
             "split"] = "excluded_nan_pwr"
    rows.loc[rows["beam"] < 0, "split"] = "no_target"
    dev = (rows["split_src"] == "development") & (rows["split"] == "adaptation")
    rows.loc[(rows["split_src"] == "development") & ~dev & (rows["split"] == "adaptation"),
             "split"] = "no_target"

    for scn in sorted(rows.loc[dev, "scenario"].unique()):
        m = dev & (rows["scenario"] == scn)
        idx = rows.index[m].to_numpy()
        fr = rows.loc[m, "frame"].to_numpy()
        if mode == "random":
            # faithful to the paper's wording, but consecutive DeepSense6G frames
            # overlap in time, so this leaks the test set into training
            perm = rng.permutation(len(idx))
            cut = int(round(train_frac * len(idx)))
            rows.loc[idx[perm[:cut]], "split"] = "train"
            rows.loc[idx[perm[cut:]], "split"] = "test"
            continue
        if mode == "session":
            groups = [idx[sessions(fr, gap) == s] for s in np.unique(sessions(fr, gap))]
        else:  # block
            o = np.argsort(fr, kind="stable")
            groups = [idx[o][i:i + block] for i in range(0, len(idx), block)]
        order = rng.permutation(len(groups))
        n_tr = int(round(train_frac * len(groups)))
        for rank, gi in enumerate(order):
            rows.loc[groups[gi], "split"] = "train" if rank < n_tr else "test"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-mode", choices=["block", "session", "random"], default="block",
                    help="block: contiguous 50-frame blocks (default, leakage-controlled); "
                         "session: whole recording sessions; "
                         "random: per-sample, matches the paper's wording but leaks")
    ap.add_argument("--train-frac", type=float, default=config.TRAIN_FRAC)
    ap.add_argument("--block", type=int, default=config.SPLIT_BLOCK)
    ap.add_argument("--session-gap", type=int, default=config.SESSION_GAP)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    a = ap.parse_args()

    frames, pwrs = [], []
    for split in config.SPLITS:
        root, rows = collect(split)
        print(f"[{split}] {len(rows)} csv rows")
        feats, pwr = build(split, rows, root)
        merged = pd.concat([rows[["sample_id", "split_src", "scenario", "frame"]], feats], axis=1)
        frames.append(merged)
        pwrs.append(pwr)
        for name, col in [("image", "m_image"), ("lidar", "m_lidar"), ("radar", "m_radar"),
                          ("beam-hist", "m_beam"), ("gps", "m_gps")]:
            print(f"   available {name:<10} {int(merged[col].sum()):>5}/{len(merged)}")

    rows = pd.concat(frames, ignore_index=True)
    pwr = np.concatenate(pwrs, axis=0)
    rows["pwr_row"] = np.arange(len(rows))
    rows = assign_splits(rows, a.split_mode, a.train_frac, a.block, a.seed, a.session_gap)

    # ---- GPS min-max fit on the training split only (AMBER eq. 12)
    tr = rows["split"] == "train"
    cols = ["ue_x_1", "ue_y_1", "ue_x_2", "ue_y_2"]
    gps_norm = {c: {"min": float(np.nanmin(rows.loc[tr, c])),
                    "max": float(np.nanmax(rows.loc[tr, c]))} for c in cols} if tr.any() else {}

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.INDEX_DIR / "samples.csv", index=False)
    np.save(config.INDEX_DIR / "beam_pwr.npy", pwr)
    (config.INDEX_DIR / "gps_norm.json").write_text(json.dumps(
        {"fit_on": "split == 'train'", "columns": gps_norm,
         "apply": "(v - min) / (max - min), clipped to [0,1] for unseen extremes"}, indent=2))
    (config.INDEX_DIR / "meta.json").write_text(json.dumps({
        "pipeline": "AMBER-style preprocessing (Wen et al.), DeepSense6G scenarios 31-34",
        "n_samples": int(len(rows)),
        "amber_params": {"K_beams": config.N_BEAMS, "W": W, "n_gps": config.N_GPS,
                         "n_beam_history": config.N_BEAM_HISTORY},
        "paper_unspecified_choices": {
            "radar_NFFT": config.RADAR_NFFT,
            "bev_grid": list(config.BEV_GRID),
            "bev_x_range": list(config.BEV_X_RANGE),
            "bev_y_range": list(config.BEV_Y_RANGE),
            "bev_max_per_cell": config.BEV_MAX_PER_CELL,
            "image_size": list(config.IMAGE_SIZE),
        },
        "split": {"mode": a.split_mode, "train_frac": a.train_frac, "block": a.block,
                  "session_gap": a.session_gap, "seed": a.seed,
                  "adaptation": "held out whole, never split"},
        "beam_history": "derived from per-frame mmWave_power files at tau <= t-1 only",
    }, indent=2))

    summary = (rows.groupby(["split", "scenario"])
                   .agg(n=("sample_id", "size"),
                        m_image=("m_image", "mean"), m_lidar=("m_lidar", "mean"),
                        m_radar=("m_radar", "mean"), m_beam=("m_beam", "mean"),
                        m_gps=("m_gps", "mean")).round(3).reset_index())
    summary.to_csv(config.INDEX_DIR / "split_summary.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {config.INDEX_DIR}/samples.csv ({len(rows)} rows, {len(rows.columns)} cols)")
    print(f"wrote {config.INDEX_DIR}/beam_pwr.npy {pwr.shape}, gps_norm.json, meta.json")


if __name__ == "__main__":
    main()
