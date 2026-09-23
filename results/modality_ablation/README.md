# Modality ablation — results

Which modalities are actually worth paying for, relative to the cheap
GPS + Radar configuration? Eight configurations, each trained independently
from scratch under an identical protocol, varying only which modalities the
model may use.

Protocol (identical across all configurations, verified): 10 epochs, batch 16,
lr 1e-4, pool 4, seed 2022, modality dropout 0.0, **historical beams disabled**.
Reported on `val` (2,198 labelled, scenarios 32/33/34), with `adaptation`
(100 labelled, scenarios 31/32/33) as a secondary check.

| | status |
|---|---|
| **part 2** | ✅ complete — `part2/` |
| **part 1** | ⬜ not yet run — GPS, GPS+Radar, GPS+Radar+Camera, full |

**Part 1 is still needed.** It trains the GPS + Radar baseline that every delta
in this experiment is measured against, so the headline question cannot be
answered yet. Run `notebooks/modality_ablation_part1.ipynb`, then merge both
parts with `notebooks/modality_ablation_report.ipynb`.

## Part 2 — validation split

| Configuration | Top-1 | Top-3 | Top-5 | DBA | GFLOPs | Params (M) | Train |
|---|---|---|---|---|---|---|---|
| **GPS + Camera** | **0.4604** | **0.8139** | **0.9295** | **0.8835** | 48.2 | 26.7 | 56 min |
| GPS + Camera + LiDAR | 0.4377 | 0.7962 | 0.9177 | 0.8727 | 70.9 | 38.0 | 83 min |
| GPS + Radar + LiDAR | 0.3640 | 0.6984 | 0.8389 | 0.8025 | 46.2 | 27.9 | 65 min |
| GPS + LiDAR | 0.3449 | 0.6547 | 0.8126 | 0.7770 | 23.1 | 16.5 | 38 min |

Three things stand out.

**Camera is the modality that matters.** GPS + Camera is the best configuration
in this half on every metric, and it does so with two modalities. This
independently confirms the preliminary masking result in
`../amber/run2_scenarios_32_33_34/`, where removing Camera cost 0.0965 Top-1
against 0.0105–0.0159 for every other modality.

**Adding LiDAR to GPS + Camera makes it worse**, by −2.3 pp Top-1 and −1.1 pp
DBA, while costing +22.7 GFLOPs. More modalities is not monotonically better
here, which is itself an argument for selective acquisition rather than always
using everything.

**GPS + Camera beats the full AMBER run** (val Top-1 0.4604 vs 0.4345 for
`run2_scenarios_32_33_34`). Not a like-for-like comparison — that run used
17 epochs of a 20-epoch schedule, modality dropout 0.15 and historical beams
enabled — but it does suggest the full model is not extracting value from its
extra modalities in proportion to their cost.

## The in-domain / out-of-domain inversion

The ranking on the unseen scenario is **exactly reversed**:

| Configuration | val DBA (scenarios 32/33/34, seen) | scenario 31 DBA (unseen) |
|---|---|---|
| GPS + Camera | **0.8828** (best) | **0.0200** (worst) |
| GPS + Camera + LiDAR | 0.8714 | 0.0933 |
| GPS + Radar + LiDAR | 0.8023 | 0.1987 |
| GPS + LiDAR | 0.7783 (worst) | **0.2240** (best) |

Perfectly monotonic across four configurations, in both directions. The natural
reading: camera features are **site-specific** — a model can memorise how a
particular intersection looks — whereas LiDAR and radar encode relative
geometry, which transfers further. Camera therefore buys the most in-domain
accuracy and generalises the worst.

Two caveats. Scenario 31 has n=50, so the individual numbers are noisy; it is
the consistent ordering across all four configurations that carries the signal.
And every configuration is poor on scenario 31 (0.02–0.22 DBA against 0.78–0.88
in-domain), so "LiDAR generalises better" means less catastrophically bad, not
good. Scenario 31 has zero training samples and is the only straight-road
geometry.

**Why this matters for the adaptive-routing direction.** Judged on in-domain
validation alone, Camera is the obvious modality to escalate to. But 48 % of the
official test split is scenario 31, where Camera is the worst choice. So the
right modality to acquire appears to depend on whether the sample resembles the
training distribution — a stronger and more interesting motivation for routing
than confidence gating alone, and one worth testing directly once part 1 lands.

## Contents

```
part2/
├── per_config/<slug>.json      full record per configuration: metrics (overall
│                               and per scenario, val + adaptation), cost,
│                               protocol, effective availability, timing
├── results_part2.csv           the table above
├── config_part2.json           the protocol as run
├── runs/<slug>/                config.json, history.csv, metrics.json, best.pt
└── notebook_as_run.ipynb       the executed Kaggle notebook
```

`per_config/*.json` is the input to `modality_ablation_report.ipynb` — upload
these four plus part 1's four as a Kaggle dataset, or point the report's
`SEARCH_ROOTS` here to run it locally.

Checkpoints are git-ignored. `last.pt` files were deleted: they are resume-only
artifacts of finished runs and accounted for 1.6 GB.

## Note on effective availability

`GPS + Radar + LiDAR` trained with radar available for only **77.7 %** of
training samples, because scenario 34 ships radar for ~40 % of its samples and a
sample needs all five consecutive frames. Every configuration's real
availability is recorded in its `per_config` JSON. The mask can only remove a
modality, never conjure one, so this understates rather than inflates radar's
contribution.
