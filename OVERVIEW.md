# Project Overview — Multimodal mmWave Beam Prediction (AMBER)

All results obtained so far, with the caveats that qualify them.

**Task.** Predict the best beam of a 64-beam codebook from a 5-step window of
camera, LiDAR, radar and GPS. DeepSense6G 2022 Multi-Modal, scenarios 31–34.

**Status.** Preprocessing complete. AMBER baseline trained. The eight-configuration
modality ablation is complete and is the main result. A knowledge-distillation
baseline has run on an earlier version of the data. The adaptive-routing
contribution is not yet built — the ablation has just changed what it should be.

---

## 1. Data

Three official releases, 46 GB raw, preprocessed to 11 GB of tensors.

| split | n | scenarios | role |
|---|---|---|---|
| `train` | 8,844 | 32, 33, 34 | fitting |
| `val` | 2,198 | 32, 33, 34 | model selection, all reported metrics |
| `adaptation` | 100 | 31, 32, 33 | secondary check; only source of scenario-31 numbers |
| `test` | 625 | 31, 32, 33, 34 | **unlabelled** — predictions only, cannot be scored |
| `excluded_nan_pwr` | 101 | 32, 33, 34 | corrupt labels, quarantined |

Modality availability on `train` (what the eq. 3 mask sees): image 1.000,
lidar 1.000, radar **0.777**, beam 0.979, gps 1.000. Radar is short because
scenario 34 ships radar for ~40 % of its samples and a sample needs all five
consecutive frames.

**Two structural gaps.** Scenario 31 has **zero training samples** — it appears
only in `adaptation` (50) and `test` (300) — and is the only straight-road
geometry. And **48 % of the test split is scenario 31**, so nearly half of it is
out-of-domain. Closing this needs the standalone scenario-31 release (7,012
frames) the AMBER paper used.

## 2. Preprocessing

Per AMBER's specification: radar → 2-channel range-angle/range-velocity tensor
under one joint min-max (no clutter removal); LiDAR → BEV count histogram capped
at 5 per cell; camera → 256×256 cache, per-image standardisation; GPS → Cartesian
metres relative to the base station, normalised on `train` only; plus the 4 past
beam indices and a 5-element availability mask. Detail in [README.md](README.md).

## 3. AMBER baseline

| | run 1 | run 2 |
|---|---|---|
| trained on | scenarios 32, 33 | scenarios 32, 33, **34** |
| train / val | 5,544 / 1,350 | 8,844 / 2,198 |
| epochs | 20 of 20 | 17 of 20 |
| Top-1 | 0.3852 | **0.4345** |
| Top-3 | 0.7230 | **0.7862** |
| Top-5 | — | **0.9113** |
| DBA | 0.8338 | **0.8611** |

Adding scenario 34 gained **+6.8 pp Top-1, +9.0 pp Top-3, +4.7 pp DBA** — it also
introduced the crossroad geometry the model had never seen.

**Error structure (run 2).** Median absolute beam error is **1**; 43.4 % exact,
78.6 % within ±1 beam, 93.9 % within ±3. The model lands on or next to the right
beam almost always, which is what the high DBA reflects.

For reference the paper reports Top-1 0.6415 / Top-3 0.8907 / DBA 0.9294. **Not
like-for-like:** it trains on all four scenarios including scenario 31's separate
7,012-frame release, uses a random 80/20 split that leaks temporally adjacent
frames, and keeps historical beams enabled.

## 4. Modality ablation — the main result

Eight configurations, each trained **independently from scratch** under a verified
identical protocol (10 epochs, batch 16, lr 1e-4, pool 4, seed 2022, modality
dropout 0, **historical beams disabled**), varying only which modalities the model
may use. 8.2 GPU-hours total.

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

### Four findings

**1. GPS + Camera is the best configuration, and beats the full model at half the
compute.** DBA 0.8835 / Top-1 0.4604 against the 4-modality model's 0.8759 /
0.4399, for 48.2 against 94.1 GFLOPs. The Pareto frontier *ends* there:

```
  GPS  →  GPS + LiDAR  →  GPS + Radar  →  GPS + Radar + LiDAR  →  GPS + Camera
```

Everything costing more than GPS + Camera has lower DBA, so **the full multimodal
model is Pareto-dominated by a two-modality one**.

**2. Radar does not pay for itself.** GPS → GPS + Radar costs 23.2 GFLOPs for
+0.050 DBA, and adding radar *on top of* camera makes things worse (0.8835 →
0.8789). GPS + LiDAR (0.7770) and GPS + Radar (0.7794) are within noise, so radar
is no better than LiDAR as a cheap partner.

**3. GPS alone is the real cheap tier: DBA 0.7298 at 0.39 GFLOPs** — 83 % of the
full model's DBA for **0.4 % of its compute**, a 240× ratio. The most consequential
number in the table.

**4. Adding modalities is not monotonic.** Two (0.8835) > three (0.8789) > four
(0.8759).

### Incremental value over GPS + Radar

| Configuration | ΔTop-1 | ΔDBA | ΔGFLOPs | DBA per GFLOP |
|---|---|---|---|---|
| GPS + Camera | +0.1083 | +0.1041 | +24.7 | 0.00422 |
| GPS + Radar + Camera | +0.0910 | +0.0995 | +47.9 | 0.00208 |
| GPS + Radar + Camera + LiDAR | +0.0878 | +0.0965 | +70.5 | 0.00137 |
| GPS + Camera + LiDAR | +0.0856 | +0.0933 | +47.3 | 0.00197 |
| GPS + Radar + LiDAR | +0.0119 | +0.0231 | +22.7 | 0.00102 |
| GPS + LiDAR | -0.0072 | -0.0024 | -0.5 | 0.00459 |
| GPS | -0.0318 | -0.0496 | -23.2 | 0.00214 |

Camera is worth **+0.104 DBA** against LiDAR's **+0.023** — a 4.5× margin.

### Corroboration

A cheaper measurement agrees. Masking one modality at a time on the *single* run-2
model (`results/amber/run2_scenarios_32_33_34/preliminary_masking_ablation_val.csv`)
gives: removing Camera −0.0965 Top-1, LiDAR −0.0159, GPS −0.0155, Radar −0.0132,
BeamIdx −0.0105. Camera dominates by ~6×. That measures something different —
how much one trained model leans on each input, not what a model trained without
it achieves — so the agreement across three independent routes is meaningful.

## 5. The camera generalisation trap

The ranking on the **unseen** scenario 31 is almost exactly inverted, and the split
is entirely explained by whether camera is used.

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

Averaged: camera configurations score **0.8763** on seen scenarios and
**0.0567** on the unseen one; camera-free ones **0.7701** and
**0.1940**. **GPS alone is the best configuration on scenario 31**
(0.2640), beating every camera configuration by 3–13×.

Reading: camera features are **site-specific** — a model can memorise how one
intersection looks — while GPS, radar and LiDAR encode relative geometry that
transfers. Camera buys the most in-domain accuracy and generalises the worst.

**Caveats.** Scenario 31 is n=50, so the consistent ordering across all eight
configurations is the signal, not the individual values. And every configuration is
poor there (0.02–0.26 against 0.72–0.88 in-domain), so "generalises better" means
less catastrophically bad. Validating this properly needs scenario 31's full release.

## 6. Knowledge-distillation baseline (reference paper 4)

Teacher / radar-only student / no-KD control, in `trainedNotebooks/trained_kd.ipynb`.

| model | Top-1 | Top-5 | Top-10 | MPR |
|---|---|---|---|---|
| Teacher (multimodal) | 0.2867 | 0.8015 | 0.9044 | 0.9312 |
| Student, without KD (radar only) | 0.1244 | 0.3459 | 0.5163 | 0.7225 |
| Student, with KD (radar only) | 0.1207 | 0.3778 | 0.5637 | 0.7411 |

**Three reasons these are not comparable to §4.** It ran on the **earlier index**
(train 5,544 / val 1,350 — before scenario 34), it is a self-contained
implementation with its own dataset, model and loop at 128×128 rather than reusing
`amber/`, and it reports MPR rather than DBA.

On its own terms: **KD did not improve Top-1** (0.1207 against 0.1244) but did
improve Top-5, Top-10 and MPR. And radar-only reaches 0.1244 Top-1 where §4's
GPS-only reaches 0.3203 — consistent with radar being the weak modality, and a
reason to reconsider whether a radar-only student is the right cheap comparator.

## 7. Data-integrity findings

Original diagnostic work, not available from the papers.

**Scenario 34 arrived incomplete** — no radar, vehicle GPS or power files on first
download, and only 1,007 of 4,439 LiDAR files. Re-downloaded; labels now verify
(`argmax(power)+1 == unit1_beam` for all 4,191). Radar is still short 800 files,
leaving 40 % of its samples with usable radar.

**101 power vectors contain NaN, and their official labels are wrong.** For every
one, `unit1_beam` equals the index of the *first NaN* rather than the argmax of the
finite bins — the labels were generated with `np.argmax` on NaN-containing vectors.
This also explains why the obvious check `argmax(power)+1 == unit1_beam` passes at
100 %: both sides carry the same bug. Anyone training off the official CSV without
this filter trains on 101 corrupt labels.

**The power vectors are flatter than the standard metric assumes.** The whole
64-beam vector spans only ~2.6 dB (scenario 32) to ~5.9 dB (scenario 34), so the
conventional "beams within 3 dB of best" ambiguity measure saturates at 64 and
carries almost no information. `amber/difficulty.py` therefore also provides
`margin_db`, `entropy_bits` and `n_within_10pct`.

**Two files are truncated** rather than missing (`lidar_data_2163.ply`,
`radar_data_3998.npy`); both are reported and skipped rather than aborting a run.

## 8. What this implies for the research direction

The original framing — a cheap Radar + GPS tier escalating to Camera or LiDAR —
**does not survive the ablation**: radar is nearly worthless and LiDAR adds little.
The replacement is stronger:

| tier | configuration | GFLOPs | DBA |
|---|---|---|---|
| cheap | GPS | 0.39 | 0.7298 |
| escalate | GPS + Camera | 48.2 | 0.8835 |

A **124× cost ratio** with **+0.154 DBA** of headroom, against roughly 4× as
originally framed, and the decision collapses to a clean binary. Dropping radar and
LiDAR from the design is a *result*, not a retreat.

A second axis falls out of §5: escalating to camera is right in-domain and wrong
out-of-domain, so *when* to escalate may depend on domain familiarity rather than
confidence alone.

**The make-or-break question is still open:** can per-sample routing beat simply
always using GPS + Camera? If not, there is no contribution. The next experiment
answers it without any training — for each val sample, which configurations got it
right, crossed with the difficulty measures, giving the oracle ceiling on how often
GPS alone would have sufficed.

## 9. Where things stand

| Step | Status |
|---|---|
| Data, preprocessing, sanity checks | done |
| Baseline 4 — AMBER | done (§3) |
| Modality ablation, 8 configurations | done (§4) |
| Baseline 5 — KD radar-only student | done, on stale data and its own protocol (§6) |
| Baseline 1 — official GPS-only LSTM | not started |
| Baselines 2, 3 — LSTM, TII Transformer | dropped when the project went AMBER-only |
| Oracle routing analysis | next; prerequisites landed |
| Adaptive cost-aware gate | gated on the oracle analysis |
| Difficulty / marginal-utility analysis | tooling ready (`amber/difficulty.py`) |

## 10. Reproducing

```bash
python3 -m venv ~/.venvs/beamprep
~/.venvs/beamprep/bin/pip install -r requirements.txt
PY=~/.venvs/beamprep/bin/python bash preprocessing_amber/run_preprocessing.sh
```

Then on Kaggle: `notebooks/amber_baseline.ipynb` for §3,
`notebooks/modality_ablation_part{1,2}.ipynb` plus
`notebooks/modality_ablation_report.ipynb` for §4.

Full detail: [README.md](README.md) for preprocessing and the data findings,
[results/amber/README.md](results/amber/README.md) for the baseline runs,
[results/modality_ablation/README.md](results/modality_ablation/README.md) for the
ablation.
