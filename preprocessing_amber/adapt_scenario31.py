"""Convert the standalone scenario-31 release into the challenge CSV format.

Scenario 31 ships separately from the ML-challenge development release, in a
different shape, and this is the only way to obtain more than the 50 labelled
scenario-31 samples the `adaptation` split provides. Since every generalisation
claim in this project rests on those 50 samples, converting the standalone
release to 7,012 is the single highest-value preprocessing task available.

Two differences have to be reconciled.

**One row per frame, not per window.** `scenario31_dev.csv` has one row per
captured frame with a single `unit1_rgb`, whereas `build_index.py` expects the
challenge layout of one row per *sample* with `unit1_rgb_1..5` naming five
consecutive observations. We rebuild the windows ourselves, sliding within each
`seq_index` and requiring strictly consecutive frame indices so a window never
straddles a gap in the capture.

**Paths carry no scenario component.** Entries read `./unit1/camera_data/...`,
but `build_index.py` recovers the scenario by matching `scenario\\d+` against the
image path. We prefix each path with `./scenario31/`, which is also the layout
the file tree must be staged into.

The target frame is the *last* of the five, matching the challenge convention
that observations run tau = t-W+1 .. t and the label belongs to instant t.

Scenario 31 is intended as held-out evaluation data. Keeping it out of training
is what preserves the "unseen scenario" claim it is being used to test.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

W = 5                 # observations per sample, config.SLIDING_WINDOW
N_GPS = 2             # GPS observations, tau = t-1 .. t
SCENARIO = "scenario31"

_FRAME = re.compile(r"_(\d+)\.[a-z]+$", re.I)


def frame_of(path: str) -> int:
    m = _FRAME.search(str(path))
    if m is None:
        raise ValueError(f"no frame number in {path!r}")
    return int(m.group(1))


def _prefix(path: str) -> str:
    """./unit1/... -> ./scenario31/unit1/..., so the scenario regex matches."""
    p = str(path).lstrip("./")
    return f"./{SCENARIO}/{p}"


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Slide a W-frame window within each sequence, keeping only intact runs.

    A window is emitted only where the W frame indices are strictly
    consecutive. Capture gaps are common in this release, and a window spanning
    one would silently pair observations seconds apart.
    """
    df = df.copy()
    df["frame"] = df["unit1_rgb"].map(frame_of)
    rows = []
    dropped_gap = 0

    for seq, g in df.groupby("seq_index", sort=True):
        g = g.sort_values("frame").reset_index(drop=True)
        for i in range(len(g) - W + 1):
            win = g.iloc[i:i + W]
            frames = win["frame"].to_numpy()
            if frames[-1] - frames[0] != W - 1:      # not strictly consecutive
                dropped_gap += 1
                continue
            last = win.iloc[-1]
            row = {"seq_index": seq, "index": int(last["index"])}
            for j in range(W):
                o = win.iloc[j]
                row[f"unit1_rgb_{j + 1}"] = _prefix(o["unit1_rgb"])
                row[f"unit1_lidar_{j + 1}"] = _prefix(o["unit1_lidar"])
                row[f"unit1_radar_{j + 1}"] = _prefix(o["unit1_radar"])
            # GPS observations are the final N_GPS frames, tau = t-1 .. t
            for j in range(N_GPS):
                o = win.iloc[W - N_GPS + j]
                row[f"unit2_loc_{j + 1}"] = _prefix(o["unit2_loc"])
            # unit1_loc is the base station: static within a scenario
            row["unit1_loc"] = _prefix(last["unit1_loc"])
            row["unit1_beam"] = int(last["unit1_beam"])
            row["unit1_pwr_60ghz"] = _prefix(last["unit1_pwr_60ghz"])
            for c in ("unit1_max_pwr", "time_stamp"):
                if c in last.index:
                    row[c] = last[c]
            rows.append(row)

    out = pd.DataFrame(rows)
    print(f"  sequences          {df.seq_index.nunique()}")
    print(f"  frames in          {len(df)}")
    print(f"  windows out        {len(out)}")
    print(f"  dropped (gaps)     {dropped_gap}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True,
                    help="path to scenario31_dev.csv")
    ap.add_argument("--out", required=True,
                    help="where to write the challenge-format csv")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    need = ["index", "seq_index", "unit1_rgb", "unit1_lidar", "unit1_radar",
            "unit1_loc", "unit2_loc", "unit1_beam", "unit1_pwr_60ghz"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"scenario31_dev.csv is missing {missing}")

    print(f"adapting {a.csv}")
    out = build_windows(df)

    # The beam label must survive the conversion untouched; it is the thing the
    # whole evaluation turns on.
    assert out["unit1_beam"].between(1, 64).all(), "beam index outside 1..64"
    assert out["unit1_beam"].notna().all(), "null beam label"
    assert out["unit1_rgb_1"].str.contains(SCENARIO).all(), "scenario prefix missing"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"  wrote              {a.out}")
    print(f"  beam range         {out.unit1_beam.min()}..{out.unit1_beam.max()}"
          f" ({out.unit1_beam.nunique()} distinct)")


if __name__ == "__main__":
    main()
