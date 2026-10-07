# Modality ablation — results

Which modalities are worth paying for? Eight configurations, each trained
independently from scratch under an **identical** protocol, varying only which
modalities the model may use.

Protocol (verified identical across all eight): 10 epochs, batch 16, lr 1e-4,
pool 4, seed 2022, modality dropout 0.0, **historical beams disabled** (not a
DeepSense6G challenge input, and absent from the entire test and adaptation
splits). Reported on `val` (2,198 labelled, scenarios 32/33/34); `adaptation`
(100 labelled, scenarios 31/32/33) is the secondary check and the only source of
scenario-31 numbers. `test` is unlabelled and cannot be scored.

Both parts complete: `part1/` and `part2/`.

## All eight configurations (validation split)

| Configuration | Top-1 | Top-3 | Top-5 | DBA | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| GPS + Camera | 0.4604 | 0.8139 | 0.9295 | 0.8835 | 26.7 | 48.2 | **Pareto**
| GPS + Radar + Camera | 0.4431 | 0.8139 | 0.9327 | 0.8789 | 38.0 | 71.4 |
| GPS + Radar + Camera + LiDAR | 0.4399 | 0.8103 | 0.9286 | 0.8759 | 49.4 | 94.1 |
| GPS + Camera + LiDAR | 0.4377 | 0.7962 | 0.9177 | 0.8727 | 38.0 | 70.9 |
| GPS + Radar + LiDAR | 0.3640 | 0.6984 | 0.8389 | 0.8025 | 27.9 | 46.2 | **Pareto**
| GPS + Radar | 0.3521 | 0.6665 | 0.8103 | 0.7794 | 16.5 | 23.6 | **Pareto**
| GPS + LiDAR | 0.3449 | 0.6547 | 0.8126 | 0.7770 | 16.5 | 23.1 | **Pareto**
| GPS | 0.3203 | 0.6142 | 0.7502 | 0.7298 | 5.2 | 0.4 | **Pareto**

## Four findings

**1. GPS + Camera is the best configuration — and it beats the full model at half
the compute.** DBA 0.8835 / Top-1 0.4604 against the 4-modality model's 0.8759 /
0.4399, for 48.2 against 94.1 GFLOPs. The Pareto frontier *ends* there:

```
  GPS  ->  GPS + LiDAR  ->  GPS + Radar  ->  GPS + Radar + LiDAR  ->  GPS + Camera
```

Every configuration costing more than GPS + Camera has lower DBA, so **the full
multimodal model is Pareto-dominated**.

**2. Radar does not pay for itself.** GPS -> GPS + Radar costs 23.2 GFLOPs for
+0.050 DBA, and adding radar *on top of* camera makes things worse (0.8835 ->
0.8789). GPS + LiDAR (0.7770) and GPS + Radar (0.7794) are within noise of each
other, so radar is no better than LiDAR as a cheap partner.

**3. GPS alone is the real cheap tier: DBA 0.7298 at 0.39 GFLOPs** *(requires
high-precision positioning — see ../modality_robustness/)* — 83 % of the
full model's DBA for **0.4 % of its compute**, a 240x ratio. This is the single
most consequential number in the table for the resource-efficiency direction.

**4. Adding modalities is not monotonic.** 2 modalities (0.8835) > 3 (0.8789) >
4 (0.8759). Extra modalities actively hurt, which is itself an argument for
selective acquisition over always-on fusion.

### Incremental value over GPS + Radar

| Configuration | dTop-1 | dDBA | dGFLOPs | DBA per GFLOP |
|---|---|---|---|---|
| GPS + Camera | +0.1083 | +0.1041 | +24.7 | 0.00422 |
| GPS + Radar + Camera | +0.0910 | +0.0995 | +47.9 | 0.00208 |
| GPS + Radar + Camera + LiDAR | +0.0878 | +0.0965 | +70.5 | 0.00137 |
| GPS + Camera + LiDAR | +0.0856 | +0.0933 | +47.3 | 0.00197 |
| GPS + Radar + LiDAR | +0.0119 | +0.0231 | +22.7 | 0.00102 |
| GPS + LiDAR | -0.0072 | -0.0024 | -0.5 | 0.00459 |
| GPS | -0.0318 | -0.0496 | -23.2 | 0.00214 |

Camera is worth **+0.104 DBA** against LiDAR's **+0.023** — a 4.5x margin,
agreeing with the preliminary masking result in `../amber/` (removing Camera cost
0.0965 Top-1 against 0.0105-0.0159 for everything else). Three independent routes,
one conclusion.

## The camera generalisation trap

The ranking on the **unseen** scenario 31 is almost exactly inverted, and the split
is entirely explained by whether camera is used:

| Configuration | seen (val 32/33/34) | unseen scenario 31 | camera |
|---|---|---|---|
| GPS + Camera | 0.8828 | 0.0200 | yes |
| GPS + Radar + Camera | 0.8771 | 0.0867 | yes |
| GPS + Radar + Camera + LiDAR | 0.8741 | 0.0267 | yes |
| GPS + Camera + LiDAR | 0.8714 | 0.0933 | yes |
| GPS + Radar + LiDAR | 0.8023 | 0.1987 | no |
| GPS + LiDAR | 0.7783 | 0.2240 | no |
| GPS + Radar | 0.7780 | 0.0893 | no |
| GPS | 0.7217 | 0.2640 | no |

Averaged: camera configurations score **0.8763** on seen
scenarios but **0.0567** on the unseen one; camera-free ones
**0.7701** and **0.1940**. **GPS alone is the best
configuration on the unseen scenario** (0.2640), beating every camera configuration
by 3-13x.

The natural reading: camera features are **site-specific** — a model can memorise
how one intersection looks — while GPS, radar and LiDAR encode relative geometry
that transfers further. Camera therefore buys the most in-domain accuracy and
generalises the worst.

Caveats: scenario 31 has n=50, so the ordering rather than individual values carries
the signal; and every configuration is poor there (0.02-0.26 against 0.72-0.88
in-domain), so 'generalises better' means less catastrophically bad. Scenario 31 has
zero training samples and is the only straight-road geometry.

**This matters because 48 % of the official test split is scenario 31**, where camera
is the worst choice.

## What this implies for the adaptive-routing direction

The original framing — cheap Radar + GPS, escalating to Camera or LiDAR — does not
survive: radar is nearly worthless and LiDAR adds little. The replacement is
stronger:

| tier | configuration | GFLOPs | DBA |
|---|---|---|---|
| cheap | GPS | 0.39 | 0.7298 |
| escalate | GPS + Camera | 48.2 | 0.8835 |

A **124x** cost ratio with **+0.154 DBA** of headroom, against roughly 4x in the
original framing. The decision also collapses to a clean binary. Dropping radar and
LiDAR from the design is a *result*, not a retreat.

A second axis falls out of the generalisation trap: escalating to camera looks right
in-domain and wrong out-of-domain, so *when* to escalate may depend on domain
familiarity rather than confidence alone. That is testable directly.

## Contents

```
results_all8.csv              all eight configurations, validation split
deltas_vs_gps_radar.csv       incremental value over the cheap baseline
scenario_results_all8.csv     per scenario, val + adaptation
pareto.json                   frontier and dominated set
part1/  GPS, GPS+Radar, GPS+Radar+Camera, full   (see part1/PROVENANCE.md)
part2/  GPS+Camera, GPS+LiDAR, GPS+Radar+LiDAR, GPS+Camera+LiDAR
```

Per-run `history.csv`, `metrics.json` and `config.json` sit under each part's
`runs/<slug>/`. Checkpoints are git-ignored. Part 2 additionally has the
authoritative `per_config/*.json`; part 1's equivalents are still in its Kaggle
session — see `part1/PROVENANCE.md`.

## Known limitation

`GPS + Radar + LiDAR` and `GPS + Radar` trained with radar available for only
**77.7 %** of training samples, because scenario 34 ships radar for ~40 % of its
samples and a sample needs all five consecutive frames. The availability mask can
only remove a modality, never conjure one, so this **understates** radar's
contribution rather than inflating it. Each configuration's real availability is
recorded in its `per_config` JSON.
