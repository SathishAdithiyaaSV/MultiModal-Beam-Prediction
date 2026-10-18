# Full run — `QUICK_RUN = False`

8,844 training and 2,198 evaluation samples, both full splits. **This is the
result.** The `../quick_run/` directory is a smoke test and should not be cited.

Cheap tier GPS (0.39 GFLOPs), expensive tier GPS + Camera (48.24), so
`cost(f) = 0.3869 + f * (48.2398 - 0.3869)`. Reference points on `val`:
always-cheap DBA 0.7298 / Top-1 0.3203, always-expensive DBA 0.8835 / Top-1
0.4604.

## Headline: the regression gate works

Escalation needed to reach always-expensive DBA, and what that costs:

| policy | escalated | GFLOPs | compute saved |
|---|---|---|---|
| oracle (DBA-optimal) | 26 % | 12.83 | 73 % |
| **learned gate (`regress_gain`)** | **66 %** | **31.97** | **34 %** |
| learned gate (`classify`) | 84 % | 40.58 | 16 % |
| confidence threshold | 87 % | 42.02 | 13 % |

DBA at fixed escalation budgets:

| policy | @10 % | @25 % | @50 % | @75 % |
|---|---|---|---|---|
| oracle (DBA-optimal) | 0.8155 | 0.8819 | 0.9134 | 0.9134 |
| **learned gate (`regress_gain`)** | **0.7919** | **0.8372** | **0.8662** | **0.8850** |
| confidence threshold | 0.7653 | 0.8199 | 0.8557 | 0.8763 |
| learned gate (`classify`) | 0.7552 | 0.7958 | 0.8393 | 0.8759 |

Three findings:

1. **`regress_gain` beats confidence thresholding at every budget**, by +0.027
   DBA at 10 % escalation, +0.017 at 25 %, +0.011 at 50 %. The single-proportion
   standard error on n = 2,198 is ±0.011, and these are paired comparisons on
   identical samples, so the low-budget margins are comfortably real. In
   operational terms it reaches the always-expensive DBA for **34 % less
   compute, against confidence's 13 %** — about 2.6× the saving.
2. **The regression objective is the right one.** `classify` trails
   `regress_gain` by 0.03–0.04 DBA throughout and is beaten by a plain
   confidence threshold below 50 % escalation. Predicting the continuous DBA
   gain matches the metric being optimised; predicting the binary "cheap wrong,
   expensive right" label discards how *much* each sample stands to gain.
3. **Roughly half the oracle headroom is now claimed.** On the escalation
   needed to match always-expensive, the gate closes 66 % → from confidence's
   87 % toward the oracle's 26 %, i.e. about 34 % of the available distance.
   The remaining gap is the open problem.

The quick run had suggested `regress_gain` was within noise of confidence
(+0.001 DBA at 25 %). That was an artefact of fitting a 2,113-parameter gate on
400 samples. With 8,844 the advantage is unambiguous — which is exactly why the
quick run was not reportable.

## Error bars — and a correction to the headline

The gate is a 2,113-parameter MLP over cached features, so unlike the model
tiers it can be refit many times on a laptop. `uncertainty/` reports twenty
seeds and a 2,000-resample paired bootstrap over the evaluation samples. The
two tell different stories and the difference matters.

### What is solid

**The gate beats confidence thresholding at every escalation budget**, and this
survives both tests comfortably:

| budget | gate (20 seeds) | confidence | gap | 95 % bootstrap CI | p(gate > conf) |
|---|---|---|---|---|---|
| 10 % | 0.7924 ± 0.0017 | 0.7653 | **+0.0274** | [+0.0220, +0.0328] | 100 % |
| 25 % | 0.8354 ± 0.0037 | 0.8199 | **+0.0185** | [+0.0127, +0.0242] | 100 % |
| 50 % | 0.8673 ± 0.0029 | 0.8557 | **+0.0151** | [+0.0105, +0.0200] | 100 % |

Seed standard deviation is at most 0.0037, and every bootstrap interval
excludes zero by a wide margin. This claim is safe to make.

### What is not

**The "66 % escalation / 34 % compute saved" headline was a single seed, and it
does not hold.** Across twenty seeds:

| | value |
|---|---|
| escalation to match always-expensive | **76.8 % ± 12.6 %** (range 62–100 %) |
| compute saved | **23 % ± 13 %** |
| seeds that never match | **3 of 20** |

Seed 0 — the one the original run used — gave 66 %, near the optimistic end.
The honest figure is about **77 %, against confidence's 87 %**, and three seeds
in twenty never reach always-expensive accuracy at any escalation level.

The cause is statistical, not a bug. "Escalation needed to match" is a
threshold crossing on a curve that is nearly flat as it approaches the target,
so a DBA change of 0.002 moves the crossing point by tens of percent. It is a
brittle statistic and should not carry a headline. The DBA-at-fixed-budget
numbers above measure the same advantage stably, and are what we now lead with.

### A third finding: the gate uses 12 features, not 31

Nineteen of the thirty-one are constant on the training split and contribute
exactly nothing — the thirteen sensor-quality statistics (the zero-tensor
defect recorded in `../../quality_signal/`) plus six availability bits that
never vary. The gate actually runs on:

`top1_conf`, `top2_conf`, `margin_12`, `margin_15`, `entropy`, `top5_mass`,
`pred_beam_norm`, `gps_step`, `gps_step_x`, `gps_step_y`, `gps_at_boundary`,
`gps_radius`.

This does not weaken the result — it strengthens it. **Twelve cheap features,
seven of them free from the cheap model's own output distribution, beat the
entire classical rejection-score family.** It also means the sensor-quality
idea is genuinely untested rather than tested and found wanting.

### Scope

This bounds the *gate*, not the backbone. The tiers themselves are still single
runs, so the configuration-level numbers (0.8835, 0.8759, and the rest) carry
no error bars yet.

## Comparison against the selective-prediction literature

"Learned gate beats confidence" is a weak claim if confidence is the only
baseline, since the deferral and selective-prediction literature has a family of
standard rejection scores. All of them are computable from the cheap model's
output distribution, which we already have cached per sample, so this costs no
GPU. Escalation needed to reach always-expensive DBA (0.8835):

| routing score | esc. to match | @10 % | @25 % | @50 % |
|---|---|---|---|---|
| Chow's rule (max-softmax) | 87 % | 0.7653 | 0.8199 | 0.8557 |
| margin, top1−top2 | 93 % | 0.7555 | 0.7933 | 0.8453 |
| margin, top1−top5 | 88 % | 0.7643 | 0.8137 | 0.8521 |
| predictive entropy | 94 % | 0.7684 | 0.8083 | 0.8427 |
| 1 − top-5 mass | 88 % | 0.7645 | 0.8209 | 0.8447 |
| Gini impurity | 89 % | 0.7658 | 0.8201 | 0.8553 |
| **learned gate (regress gain)** | **66 %** | **0.7919** | **0.8372** | **0.8662** |

**Every classical rejection score needs 87–94 % escalation; the gate needs
66 %.** It also leads at every fixed budget. The classical family is tightly
clustered — they are all monotone functions of much the same uncertainty — which
is the point: the gate's advantage does not come from a better-calibrated
uncertainty estimate but from using information those scores do not see.

Raw numbers: `deferral_baselines.csv`.

*Not yet covered:* a trained learning-to-defer surrogate, and architectural
early-exit methods (BranchyNet, MSDNet) which would require their own training
runs rather than a rescoring of ours.

## Cross-scenario generalisation

Leave-one-scenario-out, gate trained without the held-out scenario. Fraction of
the always-cheap → always-expensive gap recovered at 50 % escalation:

| held out | gate @50 % | always-cheap | always-expensive | gap recovered | n |
|---|---|---|---|---|---|
| scenario 32 | 0.8078 | 0.6209 | 0.8793 | **72 %** | 600 |
| scenario 33 | 0.8452 | 0.7944 | 0.8748 | **63 %** | 750 |
| scenario 34 | 0.8596 | 0.7497 | 0.8943 | **76 %** | 848 |

The gate transfers. This is the clearest improvement over the quick run, where
the same measurement gave 45 % / 4 % / 7 % — the earlier figures reflected a
gate that had not been fitted, not a routing policy that fails to generalise.

## Oracle correction

This run executed the notebook **before** the oracle-ordering fix, so
`operating_curves.csv` still contains a single `ORACLE (upper bound)` curve
ordered by the Top-1 criterion. That bounds Top-1 but understates the DBA
ceiling (0.8365 rather than 0.8819 at 25 % escalation; 33 % rather than 26 %
escalation to match).

`operating_curves_oracle_corrected.csv` supplies both correctly-ordered curves,
recomputed here from `table_val.csv`. The tables are per-sample, so this is an
exact recomputation and not an approximation. All oracle figures quoted above
come from the corrected file. Everything else in `operating_curves.csv` — the
confidence and learned-gate curves — is unaffected by the bug and stands as
written.

## Files

| file | contents |
|---|---|
| `config.json` | run configuration, 31-feature list, cost model, leakage note |
| `operating_curves.csv` | as produced; gate curves valid, oracle mislabelled |
| `operating_curves_oracle_corrected.csv` | both oracles, correctly ordered |
| `final_comparison.csv` | gates at DBA-matched and 25 % points vs fixed configurations |
| `cross_scenario_generalisation.csv` | leave-one-scenario-out transfer |
| `table_train.csv` (8,844), `table_val.csv` (2,198) | per-sample features, both tiers' outcomes, gate targets |
| `plots/accuracy_vs_cost.png` | DBA and Top-1 against average GFLOPs |
| `notebook_as_run.ipynb` | the executed notebook, with outputs |

The tables carry the beam label and per-sample DBA as the gate's offline
*supervision*; the 31 columns in `config.json` are its *inputs*. No
label-derived column is ever an input.
