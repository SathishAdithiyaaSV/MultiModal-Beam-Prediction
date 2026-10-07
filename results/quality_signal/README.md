# Modality quality as a routing signal (RQ3)

Which inference-available signals predict that the cheap tier already suffices,
and do the ones that need an expensive sensor read add anything?

Produced by [`notebooks/modality_quality_signal.ipynb`](../../notebooks/modality_quality_signal.ipynb).

| run | status |
|---|---|
| [`quick_run_void/`](quick_run_void/) | **void — a bug made the central comparison impossible.** Also `QUICK_RUN = True`. |
| corrected run | **not yet performed.** Blocking next step. |

## The bug: feature group C was never actually tested

The notebook's whole point is to separate two cost models:

- **Group B** — signals available *before* any expensive sensor is read.
- **Group C** — B plus image, LiDAR and radar quality statistics, which are
  legitimate only under a processing-cost model.

Both groups scored **bit-identical AUC (0.5659288057727804, same per-fold
0.515 / 0.519 / 0.664)** despite C having 31 features against B's 18. Two
features sets cannot agree to 16 decimal places by chance.

**Cause.** `AmberDataset` returns zero tensors for any modality outside the
configuration's slug. The cheap tier is GPS-only, so for every sample the
image, LiDAR and radar tensors were all zeros, and all **13** sensor-quality
statistics came out identically 0.0. Constant features contribute exactly
nothing to a standardised logistic fit, so C collapsed onto B exactly. They
were also dropped from `single_feature_auc.csv`, which filters on
`nunique() > 1` — which is why that table lists only 11 features, all of them
`needs_sensor_read = False`.

`restrict_availability` was not at fault; it only ANDs the availability vector
and leaves tensors alone. The zeroing is upstream, in the dataset.

**Fix.** `run_eval` now loads every modality whenever sensor-quality features
are being collected, and restricts the *model* through the availability mask
instead. Masking is numerically equivalent to not loading (verified:
Δlogits = 0.0), so logits are unchanged. Quality features are now taken from
the pre-restriction batch, which also makes `avail_*` report which sensors
genuinely reported rather than which the configuration was permitted to use —
the latter is constant per configuration and so useless to a gate.

A guard now raises if any sensor-quality feature comes out constant, rather
than letting the comparison quietly become vacuous.

## What survives from this run

Only the no-sensor-read signals, and only as a 300-sample indication:

| feature | AUC |
|---|---|
| `top1_conf` | 0.657 |
| `margin_15` | 0.656 |
| `top2_conf` | 0.641 |
| `margin_12` | 0.622 |
| `entropy` | 0.610 |
| `gps_radius` | 0.592 |

Consistent with the oracle study's in-domain confidence AUC of 0.686, and
confirming that confidence-family signals lead. Note that **group B (18
features, 0.566) scores *worse* than group A (confidence alone, 0.650)** on
held-out scenarios — 18 features fitted on 300 samples overfits. Whether that
survives the full split is unknown.

## Impact on the gate result

[`../adaptive_gate/full_run/`](../adaptive_gate/full_run/) has the same 13
constant columns. Its headline result is **unaffected** — constant features
contribute exactly zero, so the gate genuinely achieved 34 % compute saving
against confidence's 13 % using the 18 informative features. But it means the
gate has **never been given sensor-quality information**, so that result is a
floor, not a ceiling. Re-running it with the fix may improve it.

## Next

1. Re-run with the fix and `QUICK_RUN = False`.
2. Re-run the gate with the fix; compare against the current 66 % escalation.
