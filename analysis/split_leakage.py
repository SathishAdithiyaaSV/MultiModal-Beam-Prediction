"""Quantify the temporal leakage a random split induces on DeepSense6G.

This project reports Top-1 0.4604 where the AMBER paper reports 0.6415. The
difference has been *asserted* to come from our stricter protocol. This script
measures it instead, which matters because "your numbers are low" is the first
objection any reviewer raises and an assertion will not answer it.

The argument has two halves, and neither needs a GPU.

**Windows physically overlap.** Each sample is W=5 consecutive observations
ending at `frame_target`. Two samples whose targets differ by d < 5 therefore
share 5-d of their input frames. That is not mere correlation between nearby
samples -- it is the same image, LiDAR sweep and radar frame appearing on both
sides of the split. Under a random split a large share of validation samples
are in exactly this position; under contiguous-block splitting, none are.

**The label barely moves.** The optimal beam changes slowly, so a leaked
neighbour hands over most of the answer. We measure P(same beam | d frames
apart) against the 1/64 = 1.6% a random guess would achieve.

Together these bound how much a random split inflates reported accuracy without
retraining anything.

Usage:
    python analysis/split_leakage.py [--trials 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "results/adaptive_gate/full_run"
OUT = REPO / "results/split_leakage"
W = 5                      # observations per sample; windows overlap when d < W
N_BEAMS = 64


def load() -> pd.DataFrame:
    """All labelled development samples, with scenario and target frame."""
    cols = ["sample_id", "scenario", "true_beam"]
    tr = pd.read_csv(RUN / "table_train.csv", usecols=cols).assign(split="train")
    ev = pd.read_csv(RUN / "table_val.csv", usecols=cols).assign(split="val")
    df = pd.concat([tr, ev], ignore_index=True)
    df["frame"] = df.sample_id.str.rsplit("/", n=1).str[-1].astype(int)
    return df


def nearest_gap(val_frames: np.ndarray, train_frames: np.ndarray) -> np.ndarray:
    """For each val frame, the distance to the closest train frame."""
    if len(train_frames) == 0:
        return np.full(len(val_frames), np.inf)
    t = np.sort(train_frames)
    i = np.searchsorted(t, val_frames)
    left = np.where(i > 0, val_frames - t[np.clip(i - 1, 0, len(t) - 1)], np.inf)
    right = np.where(i < len(t), t[np.clip(i, 0, len(t) - 1)] - val_frames, np.inf)
    return np.minimum(left, right)


def leakage(df: pd.DataFrame, assign: pd.Series) -> dict:
    """Share of val samples whose window overlaps a training window."""
    gaps = []
    for scn, g in df.groupby("scenario"):
        a = assign.loc[g.index]
        v = g.loc[a == "val", "frame"].to_numpy()
        t = g.loc[a == "train", "frame"].to_numpy()
        gaps.append(nearest_gap(v, t))
    gaps = np.concatenate(gaps)
    return {
        "n_val": int(len(gaps)),
        "share_overlapping": float(np.mean(gaps < W)),     # shares input frames
        "share_adjacent": float(np.mean(gaps <= 1)),       # all but one frame
        "share_within_10": float(np.mean(gaps <= 10)),
        "median_gap": float(np.median(gaps)),
    }


def random_split(df: pd.DataFrame, frac: float, rng) -> pd.Series:
    a = pd.Series("train", index=df.index)
    a.iloc[rng.permutation(len(df))[:int(round(frac * len(df)))]] = "val"
    return a


def beam_persistence(df: pd.DataFrame, max_d: int = 12) -> pd.DataFrame:
    """P(same beam) as a function of temporal separation."""
    rows = []
    for d in range(1, max_d + 1):
        same = tot = 0
        for _scn, g in df.groupby("scenario"):
            g = g.sort_values("frame")
            f = g.frame.to_numpy()
            b = g.true_beam.to_numpy()
            j = np.searchsorted(f, f + d)
            ok = (j < len(f)) & (np.take(f, np.clip(j, 0, len(f) - 1)) == f + d)
            same += int(np.sum(b[ok] == np.take(b, np.clip(j, 0, len(f) - 1))[ok]))
            tot += int(ok.sum())
        if tot:
            rows.append({"frame_gap": d, "p_same_beam": same / tot, "n_pairs": tot})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()

    df = load()
    val_frac = float((df.split == "val").mean())
    print(f"{len(df)} labelled development samples, "
          f"{(df.split=='val').sum()} val ({val_frac:.0%})")
    print(f"window length W={W}: targets within {W-1} frames share input data\n")

    ours = leakage(df, df.split)
    rng = np.random.default_rng(0)
    trials = [leakage(df, random_split(df, val_frac, rng)) for _ in range(a.trials)]
    rnd = {k: float(np.mean([t[k] for t in trials])) for k in ours}

    print(f"{'':<34}{'block split (ours)':>20}{'random split':>16}")
    for k, lbl in [("share_overlapping", f"val sharing input frames (d<{W})"),
                   ("share_adjacent", "val within 1 frame of train"),
                   ("share_within_10", "val within 10 frames of train"),
                   ("median_gap", "median frames to nearest train")]:
        f = "{:>19.1%}" if k != "median_gap" else "{:>19.1f}"
        g = "{:>16.1%}" if k != "median_gap" else "{:>16.1f}"
        print(f"{lbl:<34}" + f.format(ours[k]) + g.format(rnd[k]))

    bp = beam_persistence(df)
    print(f"\nbeam persistence (chance = {1/N_BEAMS:.1%}):")
    for _, r in bp.head(6).iterrows():
        print(f"  {int(r.frame_gap)} frame(s) apart   "
              f"same beam {r.p_same_beam:>6.1%}   ({int(r.n_pairs)} pairs)")

    p1 = float(bp.loc[bp.frame_gap == 1, "p_same_beam"].iloc[0])
    print(f"\n=> Under a random split {rnd['share_overlapping']:.0%} of validation "
          f"samples share input frames with training data,")
    print(f"   and an immediate neighbour carries the same beam {p1:.0%} of the "
          f"time against {1/N_BEAMS:.1%} chance.")
    print(f"   Our block split leaves {ours['share_overlapping']:.1%} overlapping.")

    OUT.mkdir(parents=True, exist_ok=True)
    bp.to_csv(OUT / "beam_persistence.csv", index=False)
    json.dump({"window": W, "n_samples": int(len(df)), "val_frac": val_frac,
               "block_split": ours, "random_split_mean": rnd,
               "random_trials": a.trials,
               "p_same_beam_1_frame": p1, "chance": 1 / N_BEAMS},
              open(OUT / "split_leakage.json", "w"), indent=2)
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
