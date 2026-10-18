"""Error bars for the routing claim, from cached per-sample tables alone.

Every number in this project is a single run, which is the most common thing a
reviewer objects to. For the *model* configurations that is expensive to fix --
each seed is a full retrain. For the *gate* it is nearly free: it is a
2,113-parameter MLP over cached features, so it can be refit many times on a
laptop in seconds.

Two distinct sources of uncertainty are reported, because they answer different
questions and conflating them would overstate the result.

  **Seed variance** -- refit the gate under different initialisations on the
  same data. Answers "is this gate reproducible, or did we get lucky once?"

  **Sampling uncertainty** -- paired bootstrap over the 2,198 evaluation
  samples. Answers "would this ordering survive a different draw of test data?"
  Paired, because every policy is scored on the *same* samples, so the
  comparison is far tighter than two independent confidence intervals would
  suggest.

Neither substitutes for retraining the tiers under different seeds, which
remains open. This bounds the gate, not the backbone.

Usage:
    python analysis/gate_uncertainty.py [--seeds 20] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "results/adaptive_gate/full_run"
OUT = RUN / "uncertainty"

C_LO, C_HI = 0.3869, 48.2398          # GPS, GPS+Camera GFLOPs
BUDGETS = (0.10, 0.25, 0.50)
DROP = ("sample_id", "scenario", "true_beam", "pred_beam", "correct1",
        "correct3", "dba", "exp_correct1", "exp_correct3", "exp_dba",
        "exp_pred_beam", "label_escalate", "gain_dba", "label_cheap_ok")


class Gate(torch.nn.Module):
    """The deployed architecture, reproduced exactly."""

    def __init__(self, d_in, hidden=32, out=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_gate(X, y, objective="regress", epochs=300, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp(min=1e-6)
    Xn = (Xt - mu) / sd
    yt = torch.tensor(y, dtype=torch.float32)
    g = Gate(Xn.shape[1])
    opt = torch.optim.Adam(g.parameters(), lr=lr, weight_decay=1e-4)
    if objective == "classify":
        pw = torch.tensor([(len(yt) - yt.sum()) / yt.sum().clamp(min=1)])
        lossfn = lambda p, t: torch.nn.functional.binary_cross_entropy_with_logits(
            p, t, pos_weight=pw)
    else:
        lossfn = torch.nn.functional.mse_loss
    for _ in range(epochs):
        opt.zero_grad()
        lossfn(g(Xn), yt).backward()
        opt.step()
    g.eval()
    return lambda Z: g((torch.tensor(Z, dtype=torch.float32) - mu) / sd).detach().numpy()


def dba_at(score, f, cd, ed):
    """DBA when the top `f` fraction by `score` is escalated."""
    n = len(cd)
    order = np.argsort(-np.asarray(score, dtype=float))
    esc = np.zeros(n, bool)
    esc[order[:int(round(f * n))]] = True
    return float(np.where(esc, ed, cd).mean())


def esc_to_match(score, cd, ed):
    """Smallest escalated fraction reaching the always-expensive DBA."""
    target = ed.mean()
    for f in np.linspace(0, 1, 101):
        if dba_at(score, f, cd, ed) >= target:
            return float(f)
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    TR = pd.read_csv(RUN / "table_train.csv")
    EV = pd.read_csv(RUN / "table_val.csv")
    feats = [c for c in TR.columns if c not in DROP]
    # Constant columns cannot contribute; dropping them makes the parameter
    # count honest rather than carrying thirteen dead inputs.
    live = [c for c in feats if TR[c].nunique() > 1]
    dead = [c for c in feats if c not in live]
    cd, ed = EV.dba.to_numpy(), EV.exp_dba.to_numpy()
    Xtr, Xev = TR[live].to_numpy(), EV[live].to_numpy()

    print(f"train {len(TR)}  eval {len(EV)}  features {len(live)} live"
          f"{f' ({len(dead)} dead dropped)' if dead else ''}")
    print(f"always-cheap DBA {cd.mean():.4f}   always-expensive {ed.mean():.4f}\n")

    # ---------------------------------------------------------- seed variance
    print(f"refitting the gate under {a.seeds} seeds ...")
    curves, matches = {b: [] for b in BUDGETS}, []
    for s in range(a.seeds):
        f = train_gate(Xtr, TR.gain_dba.to_numpy(), "regress", seed=s)
        sc = f(Xev)
        for b in BUDGETS:
            curves[b].append(dba_at(sc, b, cd, ed))
        matches.append(esc_to_match(sc, cd, ed))
        print(f"  seed {s:>2}  esc@match {matches[-1]:.0%}", end="\r", flush=True)
    print(" " * 40, end="\r")

    conf = -EV.top1_conf.to_numpy()
    rows = []
    print(f"{'budget':<10}{'gate DBA (mean+-sd)':>26}{'confidence':>12}{'gap':>9}")
    for b in BUDGETS:
        v = np.array(curves[b])
        c = dba_at(conf, b, cd, ed)
        print(f"{b:<10.0%}{v.mean():>17.4f} +-{v.std():.4f}{c:>12.4f}"
              f"{v.mean() - c:>+9.4f}")
        rows.append({"quantity": f"DBA @ {b:.0%} escalated", "gate_mean": v.mean(),
                     "gate_sd": v.std(), "gate_min": v.min(), "gate_max": v.max(),
                     "confidence": c, "gap": v.mean() - c})
    m = np.array(matches)
    print(f"\nescalation to match always-expensive: "
          f"{m.mean():.1%} +- {m.std():.1%}  (range {m.min():.0%}-{m.max():.0%})")
    print(f"  -> compute saved {1 - (C_LO + m.mean()*(C_HI-C_LO))/C_HI:.0%} "
          f"+- {m.std()*(C_HI-C_LO)/C_HI:.0%}")
    rows.append({"quantity": "escalation to match", "gate_mean": m.mean(),
                 "gate_sd": m.std(), "gate_min": m.min(), "gate_max": m.max(),
                 "confidence": esc_to_match(conf, cd, ed),
                 "gap": m.mean() - esc_to_match(conf, cd, ed)})

    # ------------------------------------------------- paired bootstrap on EV
    print(f"\npaired bootstrap over evaluation samples ({a.boot} resamples) ...")
    gate_sc = train_gate(Xtr, TR.gain_dba.to_numpy(), "regress", seed=0)(Xev)
    rng = np.random.default_rng(0)
    n = len(EV)
    boot = {b: [] for b in BUDGETS}
    tier = []
    for _ in range(a.boot):
        idx = rng.integers(0, n, n)
        c_, e_ = cd[idx], ed[idx]
        tier.append(e_.mean() - c_.mean())
        for b in BUDGETS:
            boot[b].append(dba_at(gate_sc[idx], b, c_, e_)
                           - dba_at(conf[idx], b, c_, e_))
    print(f"{'budget':<10}{'gate - confidence':>22}{'95% CI':>24}{'  p(gate>conf)'}")
    for b in BUDGETS:
        d = np.array(boot[b])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{b:<10.0%}{d.mean():>+22.4f}   [{lo:+.4f}, {hi:+.4f}]"
              f"{(d > 0).mean():>13.1%}")
        rows.append({"quantity": f"gate - confidence @ {b:.0%} (bootstrap)",
                     "gate_mean": d.mean(), "gate_sd": d.std(),
                     "gate_min": lo, "gate_max": hi,
                     "confidence": np.nan, "gap": (d > 0).mean()})
    t = np.array(tier)
    lo, hi = np.percentile(t, [2.5, 97.5])
    print(f"\nGPS+Camera - GPS DBA: {t.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gate_uncertainty.csv", index=False)
    json.dump({"seeds": a.seeds, "bootstrap": a.boot, "n_eval": n,
               "n_features_live": len(live), "dead_features": dead,
               "esc_to_match_mean": float(m.mean()),
               "esc_to_match_sd": float(m.std()),
               "compute_saved_mean": float(1 - (C_LO + m.mean()*(C_HI-C_LO))/C_HI)},
              open(OUT / "gate_uncertainty.json", "w"), indent=2)
    print(f"\nwrote {OUT}/gate_uncertainty.csv and .json")


if __name__ == "__main__":
    main()
