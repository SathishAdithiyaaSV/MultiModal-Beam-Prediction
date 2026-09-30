"""Per-sample beam-difficulty measures, derived from the 64-beam power vector.

These quantify how *ambiguous* a sample's optimal beam is, which is the raw
material for the planned analysis of whether cheap modalities suffice on easy
samples and expensive ones are only needed on hard ones.

    margin_db       10*log10(P_best / P_second)     small => ambiguous
    entropy_bits    Shannon entropy of P/sum(P)     0 .. log2(64) = 6
    n_within_3db    beams with P >= P_best / 2      >= 1
    n_within_10pct  beams with P >= 0.9 * P_best    >= 1

**These are computed from the TARGET and must never be fed to a model as an
input feature.** They exist for offline analysis only: to characterise samples,
to correlate with model error, and to decide whether a difficulty-aware gate is
worth building. A gate that consumed them at inference would be reading the
answer.

Caveat measured on this dataset: the whole 64-beam vector spans only ~2.6 dB
(scenario 32) to ~5.9 dB (scenario 34), so `n_within_3db` saturates near 64 for
many samples and carries little information. Prefer `margin_db`,
`entropy_bits` and `n_within_10pct`.

Computed from `beam_pwr.npy`, which the index already stores, so nothing needs
re-preprocessing and the index itself is left untouched.

Run:
  python -m amber.difficulty --index-dir data/processed_amber/index
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BEAMS = 64


def difficulty_metrics(pwr: np.ndarray) -> pd.DataFrame:
    """Per-row difficulty measures for an (N, 64) power matrix.

    NaN bins -- present in 101 development samples whose power files contain
    literal `nan` -- are ignored rather than propagated, so the measures
    describe the finite beams. Rows with no finite bin at all (the unlabelled
    test split) come back as NaN.
    """
    pwr = np.asarray(pwr, dtype=np.float64)
    finite = np.isfinite(pwr)
    p = np.where(finite, pwr, np.nan)
    p = np.where(finite, np.clip(p, 1e-12, None), np.nan)

    n_ok = finite.sum(axis=1)
    usable = n_ok >= 2

    srt = np.sort(p, axis=1)                        # NaNs sort to the end
    best = np.nanmax(p, axis=1, initial=-np.inf)
    best = np.where(n_ok >= 1, best, np.nan)
    second = np.full(len(p), np.nan)
    idx = np.where(usable)[0]
    second[idx] = srt[idx, n_ok[idx] - 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        margin_db = 10.0 * np.log10(best / second)
        q = p / np.nansum(p, axis=1, keepdims=True)
        entropy_bits = -np.nansum(np.where(finite, q * np.log2(q), 0.0), axis=1)
        entropy_bits = np.where(n_ok >= 1, entropy_bits, np.nan)
        worst = np.where(n_ok >= 1, np.nanmin(np.where(finite, p, np.inf), axis=1), np.nan)
        spread_db = 10.0 * np.log10(best / worst)

    n_within_3db = np.where(finite & (p >= (best[:, None] / 10 ** 0.3)), 1, 0).sum(axis=1)
    n_within_10pct = np.where(finite & (p >= 0.9 * best[:, None]), 1, 0).sum(axis=1)

    return pd.DataFrame({
        "margin_db": margin_db,
        "entropy_bits": entropy_bits,
        "n_within_3db": np.where(n_ok >= 1, n_within_3db, -1),
        "n_within_10pct": np.where(n_ok >= 1, n_within_10pct, -1),
        "spread_db": spread_db,
        "n_finite_beams": n_ok,
    })


def load_with_difficulty(index_dir: str | Path) -> pd.DataFrame:
    """`samples.csv` joined to its difficulty measures, via the `pwr_row` column."""
    index_dir = Path(index_dir)
    samples = pd.read_csv(index_dir / "samples.csv")
    pwr = np.load(index_dir / "beam_pwr.npy")
    metrics = difficulty_metrics(pwr)
    # join on pwr_row rather than assuming row order
    joined = samples.join(metrics, on="pwr_row")
    return joined


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index-dir", type=Path, default=Path("data/processed_amber/index"))
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <index-dir>/difficulty.csv")
    a = ap.parse_args()

    joined = load_with_difficulty(a.index_dir)
    cols = ["sample_id", "source", "scenario", "split", "beam", "pwr_row",
            "margin_db", "entropy_bits", "n_within_3db", "n_within_10pct",
            "spread_db", "n_finite_beams"]
    out = a.out or a.index_dir / "difficulty.csv"
    joined[cols].to_csv(out, index=False)
    print(f"wrote {out}  ({len(joined)} rows)\n")

    scored = joined[joined.split.isin(["train", "val", "adaptation"])]
    print("median difficulty by split and scenario:")
    print(scored.groupby(["split", "scenario"])[
        ["margin_db", "entropy_bits", "n_within_3db", "n_within_10pct", "spread_db"]
    ].median().round(3).to_string())
    print("\nNote: n_within_3db saturates near 64 where the power vector is flat; "
          "prefer margin_db, entropy_bits and n_within_10pct.")


if __name__ == "__main__":
    main()
