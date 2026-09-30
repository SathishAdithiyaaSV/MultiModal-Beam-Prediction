# Oracle routing analysis

**Can per-sample modality routing beat always using the best fixed
configuration?** Answered on the validation split (n=2,198) using the eight
ablation checkpoints. No training — per-sample predictions from each
configuration, then arithmetic.

Verified first: the per-sample decomposition reproduces all eight ablation
aggregates to four decimals, which is what makes scoring a routed mixture valid
without re-running any model.

## The two tiers

| tier | configuration | GFLOPs | DBA |
|---|---|---|---|
| cheap | GPS | 0.39 | 0.7298 |
| escalate | GPS + Camera | 48.24 | 0.8835 |

A 125× cost ratio.

## Where the tiers agree (Top-1)

| | share of samples |
|---|---|
| both correct | 23.1 % |
| **cheap only — escalation *hurts*** | **8.9 %** |
| **expensive only — escalation pays** | **22.9 %** |
| neither | **45.0 %** |

## Headroom exists

| | Top-1 | GFLOPs |
|---|---|---|
| always cheap | 0.3203 | 0.39 |
| always expensive | 0.4604 | 48.24 |
| **oracle router** | **0.5496** | **11.36** |

The oracle beats always-expensive by **+0.089 Top-1 at 24 % of its compute**,
escalating 22.9 % of samples. It consults the ground truth, so it is an
unachievable upper bound — but it proves a good router would win substantially.

## The realisable gate captures little of it

Escalating the least-confident samples, using only the cheap model's own output:

| target | escalated | GFLOPs | compute saved |
|---|---|---|---|
| match always-expensive Top-1 | 89.0 % | 42.98 | 11 % |
| match always-expensive DBA | 87.0 % | 42.02 | 13 % |

Escalating ~88 % of samples to save ~12 % of compute is thin. The gap between
this and the oracle is the whole story.

**Why:** the gate signal is weak.

| signal | AUC for "cheap tier already correct" |
|---|---|
| cheap model confidence | **0.6864** |
| n_within_10pct *(offline only)* | 0.6331 |
| margin_db *(offline only)* | 0.5932 |
| entropy_bits *(offline only)* | 0.5913 |
| spread_db *(offline only)* | 0.5458 |

## Two findings that change the plan

**1. The difficulty-aware gate is dead.** Every beam-ambiguity measure scores
*below* raw confidence. They are computed from the target, so they were the
**ceiling** for that approach — and the ceiling sits beneath what confidence
already gives for free. This closes a specific item from the original project
plan.

**2. The gate does beat most fixed configurations, just not the best one.**
Comparing against all eight rather than only GPS + Camera:

| operating point | DBA | GFLOPs | fixed configs dominated |
|---|---|---|---|
| gate @ 25 % escalated | 0.8199 | **12.35** | GPS+Radar, GPS+LiDAR, GPS+Radar+LiDAR |
| gate @ 50 % escalated | 0.8557 | 24.31 | GPS+Radar+LiDAR |
| gate @ 87 % escalated | 0.8835 | 42.02 | 4 of 8 |

At 25 % escalation the gate is **+0.017 DBA over GPS+Radar+LiDAR at 3.7× less
compute**, and **+0.041 DBA over GPS+Radar at 1.9× less**. It reaches operating
points no fixed configuration can, which is the resource-efficiency claim —
even though it does not match the single best configuration at equal accuracy.

## Limitations

- **45 % of samples are wrong under both tiers.** The largest bucket, and it
  caps any routing gain.
- **8.9 % are actively hurt by escalation** — consistent with the camera
  generalisation trap from the ablation.
- Measured on `val`, which is **in-domain by construction**. Camera generalises
  badly to the unseen scenario 31, where GPS alone is better, so a gate tuned
  here may escalate the wrong way on the 48 % of the test split that is
  scenario 31.
- Difficulty measures are target-derived and **offline only** — a deployed gate
  cannot read them.

## What this implies for the next step

The oracle/confidence gap is now a **measured target**: AUC 0.686 against
perfect separation. The honest next move is to *learn* a gate — predict "will
GPS suffice" from the cheap model's features rather than thresholding its top-1
confidence — with the oracle as the known ceiling and difficulty features ruled
out.

## Contents

```
summary.json               headline numbers as run
gate_vs_fixed_configs.csv  the Pareto dominance comparison above
gate_signal_auc.csv        gate signal quality
notebook_as_run.ipynb      the executed Kaggle notebook
```

The run also produced `per_sample_val.csv` (722 KB, 2,198 × 8 configurations),
`confidence_gate_sweep.csv`, 16 per-configuration prediction CSVs and two
figures, all in its Kaggle output. Those are worth downloading into this
directory — `per_sample_val.csv` in particular is the input to any learned-gate
experiment.
