# CONTEXT.md — full project handoff

Everything a new agent or collaborator needs to continue this work. Written
2026-10-07. Read this first, then `OVERVIEW.md` (narrative) and
`PRESENTATION.md` (slide-shaped results).

---

## 1. The project in one paragraph

Predict the optimal mmWave beam out of 64 from multimodal sensor data, using
**DeepSense6G 2022, scenarios 31–34 only**. Each sample is 5 camera frames,
5 LiDAR sweeps, 5 radar frames and 2 GPS positions. The baseline architecture is
**AMBER** (Adaptive Multimodal Beam prediction with mask transformers),
reimplemented from the paper in `amber/`. The research goal is a
**sample-adaptive, cost-aware modality routing mechanism**: run a cheap model on
most samples, escalate only the ones that need an expensive one.

### Hard constraints — do not violate

- **Never use the 2023 V2V dataset.** 2022 scenarios 31–34 only.
- **Ground-truth beam power must never be an inference input.** The routing gate
  must not see the beam label, the 64-dim power vector, a future beam, oracle
  correctness, any target-derived difficulty measure, or test labels. These are
  admissible *only* as offline supervision when fitting the gate.
- **Historical beam indices are disabled** (`beam` modality exists in code but
  is not used as an input). They are absent from 100 % of the test split, so
  using them would inflate results with a signal that vanishes at inference.
- **Commits are authored solely by the user.** Never add AI co-author trailers.
- **Never push without being explicitly told to.** Commit freely.
- Do not run heavy training locally. Training happens on Kaggle/Colab GPUs.

---

## 2. Repository layout

```
.
├── CONTEXT.md             <- this file
├── OVERVIEW.md            <- high-level narrative of everything done
├── PRESENTATION.md        <- slide-shaped results, 18 slides
├── README.md              <- setup, commands, API reference
├── amber/                 <- the model package (16 modules)
├── preprocessing_amber/   <- raw DeepSense -> tensors + index
├── notebooks/             <- Kaggle notebooks + their generator scripts
├── results/               <- all experimental outputs, organised per experiment
├── tests/test_amber.py    <- regression tests
├── references/            <- papers
└── trainedNotebooks/      <- one stale KD notebook, own protocol, low priority
```

### `amber/` — the model

| module | contents |
|---|---|
| `config.py` | `MODALITIES = ("image","lidar","radar","beam","gps")`; `ModelConfig` (`embed_dim=256`, `n_heads=8`, `pool_hw=(4,4)`, `skip_unavailable_encoders`), `TrainConfig` (`batch_size=16`, `num_workers=4`) |
| `dataset.py` | `AmberDataset(index_dir, split, cfg, root=None, modalities=None, degradations=None)`. Loads per-modality tensors, applies the eq. (3) availability mask, eq. (10) per-image standardisation, optional degradations |
| `encoders.py` | per-modality CNN encoders |
| `embeddings.py` | modality/positional embeddings |
| `transformer.py` | masked transformer blocks, eqs. (21)–(31) |
| `cma.py` | cross-modal attention + contrastive objective, eqs. (32)–(34) |
| `losses.py` | focal loss with Gaussian soft labels, eq. (35) |
| `metrics.py` | Top-K accuracy and DBA, eqs. (37)–(39) |
| `model.py` | assembles the above |
| `train.py` | CLI. Flags: `--modalities`, `--skip-unavailable-encoders`, `--beam-dropout`, `--resume`. Writes `enabled_modalities` and `model_config` into the checkpoint |
| `predict.py` | inference CLI, `--modalities`, outputs suffixed per configuration |
| `ablation.py` | `parse_modalities`, `enabled_mask`, `restrict_availability`, `active_parameters`, `measure_latency`, `measure_gflops`, `config_label` |
| `difficulty.py` | offline target-derived difficulty measures (`margin_db`, `entropy_bits`, `n_within_3db`, `n_within_10pct`, `spread_db`). **Offline analysis only — never gate inputs** |
| `degrade.py` | faithful modality degradations, plus a `REJECTED` dict documenting what cannot be done faithfully and why |
| `quality.py` | 31 inference-available gate features, grouped A/B/C by cost model |

### `preprocessing_amber/` — raw → tensors

`config.py` (auto-discovers sources, `AMBER_DATA_ROOT` override) · `ply_io.py`
(custom PLY reader) · `audit_dataset.py` · `radar_ra_rv.py` (eqs. 5–8) ·
`lidar_bev.py` (eq. 11) · `image_cache.py` · `build_index.py` ·
`run_preprocessing.sh` (runs the whole chain).

Run: `bash preprocessing_amber/run_preprocessing.sh`

### `notebooks/` — everything runs on Kaggle

Notebooks are **generated**, never hand-edited. Edit the `_build_*.py` script
and re-run it.

| builder | produces |
|---|---|
| `_build_kaggle_nb.py` | `amber_baseline.ipynb` |
| `_build_ablation_nb.py` | `modality_ablation_part{1,2}.ipynb` |
| `_build_report_nb.py` | `modality_ablation_report.ipynb` |
| `_build_oracle_nb.py` | `oracle_routing.ipynb` |
| `_build_adaptive_nbs.py` | `modality_robustness.ipynb`, `modality_quality_signal.ipynb`, `learned_adaptive_gate.ipynb` |

**Known trap:** generating notebooks with nested triple-quoted heredocs fails.
Use `'''` inside cell bodies when the outer string is `"""`. This bit three
times; it is documented in the builder docstrings.

Helpers to paste into a Kaggle cell when inputs misbehave:
`kaggle_input_report.py`, `kaggle_missing_files_report.py`.

---

## 3. The data

| split | n | scenarios | role |
|---|---|---|---|
| `train` | 8,844 | 32, 33, 34 | fitting |
| `val` | 2,198 | 32, 33, 34 | model selection, all reported metrics |
| `adaptation` | 100 | 31, 32, 33 | **only source of scenario-31 numbers** (n=50 for sc. 31) |
| `test` | 625 | 31, 32, 33, 34 | **unlabelled** — predictions only, cannot be scored |
| `excluded_nan_pwr` | 101 | 32, 33, 34 | corrupt labels, quarantined |

Modality availability on `train`: image 1.000, lidar 1.000, **radar 0.777**,
beam 0.979, gps 1.000. Radar is short because scenario 34 ships radar for only
~40 % of its samples and a sample needs all five consecutive frames.

**`source` vs `split` are different things.** `source` is the physical release
the sample came from; `split` is its permitted use. Do not conflate them.

**Splits are block-based** — contiguous 50-frame blocks — to prevent temporally
adjacent frames leaking between train and val. This is stricter than the random
split the AMBER paper used, which is part of why absolute numbers are lower.

### Two structural gaps

1. **Scenario 31 has zero training samples.** It appears only in `adaptation`
   (n=50) and `test`. All generalisation claims rest on those 50 samples.
   **The standalone scenario-31 release has 7,012 samples — downloading it is
   the single highest-value action available.**
2. **48 % of the official test split is scenario 31**, i.e. nearly half of it is
   out of domain.

### Three data defects we found (all real, all in the public release)

1. **Truncated files.** `lidar_data_2163.ply` and `radar_data_3998.npy` are
   truncated. Handled by `TruncatedPLYError` and equivalent tolerance in
   `radar_ra_rv.py` — report and skip, never crash the scenario.
2. **101 corrupt power vectors.** The official `unit1_beam` label equals the
   *first-NaN index*. The standard integrity check `argmax(power)+1 ==
   unit1_beam` passes at 100 % **because both sides share the bug**. These 101
   samples are quarantined in `excluded_nan_pwr`.
3. Power vectors must be loaded **per row**, not all-or-nothing; one incomplete
   scenario otherwise blanks labels for the whole source.

---

## 4. Results so far

All metrics on `val` (n=2,198) unless stated. **DBA** = Distance-Based Accuracy,
Δ=5, decomposes exactly per sample.

### 4.1 Modality ablation — 8 configurations (`results/modality_ablation/`)

| configuration | DBA | GFLOPs |
|---|---|---|
| **GPS + Camera** | **0.8835** | 48.24 |
| GPS + Radar + Camera + LiDAR (full) | 0.8759 | 94.10 |
| GPS + Radar + LiDAR | 0.8025 | 46.23 |
| GPS + Radar | 0.7794 | 23.57 |
| GPS + LiDAR | 0.7770 | 23.05 |
| **GPS alone** | **0.7298** | **0.39** |

Findings:
- **The best model is the second-cheapest.** GPS + Camera beats the full
  four-modality model at half the compute. The Pareto frontier ends there.
- **GPS alone gets 83 % of full DBA at 0.4 % of the compute** (240× cheaper).
- **Radar does not pay** (+0.050 DBA for 23.2 GFLOPs, and it *hurts* when added
  to camera). This killed the original "Radar+GPS cheap tier" premise.
- **The camera generalisation trap.** Camera configurations average 0.8764 on
  seen scenarios but **0.0567** on unseen scenario 31; camera-free ones 0.7701 →
  0.1940. **GPS alone is the best configuration on scenario 31.** The modality
  that buys the most in-domain accuracy generalises the worst. *(n=50 — the
  consistent ordering across all eight configurations carries this, not the
  individual values.)*

### 4.2 Oracle routing (`results/oracle_routing/`)

Per-sample ceiling, no training: oracle router **Top-1 0.5496 at 11.36 GFLOPs**
vs always-expensive 0.4604 at 48.24 — **+0.089 Top-1 at 24 % of the compute**.

But a **confidence threshold needs 87–89 % escalation to save 11–13 %** of
compute. Confidence gives AUC 0.686 for "the cheap tier already suffices".
45 % of samples are wrong under both tiers; 8.9 % are actively *hurt* by
escalating.

**Difficulty-aware gating is ruled out.** Every target-derived difficulty
measure scores *below* raw confidence (0.633, 0.593, 0.591 vs 0.686) — and since
they are computed from the target they were the *ceiling* for that approach.
Do not build it.

### 4.3 Learned adaptive gate (`results/adaptive_gate/full_run/`) — **the newest result**

Cheap tier GPS (0.39 GFLOPs), expensive tier GPS + Camera (48.24),
`cost(f) = 0.3869 + f·(48.2398 − 0.3869)`. Gate is a 2,113-parameter MLP over
31 inference-available features, trained on `train` (8,844), evaluated on `val`
(2,198).

Escalation needed to reach always-expensive DBA (0.8835):

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
| **`regress_gain`** | **0.7919** | **0.8372** | **0.8662** | **0.8850** |
| confidence threshold | 0.7653 | 0.8199 | 0.8557 | 0.8763 |
| `classify` | 0.7552 | 0.7958 | 0.8393 | 0.8759 |

- **The learned gate beats confidence thresholding at every budget** (+0.027 DBA
  at 10 % escalation, +0.017 at 25 %), saving **34 % of compute vs 13 %**.
- **The regression objective is the right one.** Predicting the continuous DBA
  gain matches the metric; predicting the binary "cheap wrong, expensive right"
  label discards *how much* each sample gains. `classify` loses to plain
  confidence below 50 % escalation — drop that arm.
- **Cross-scenario transfer works**: leave-one-scenario-out at 50 % escalation
  recovers 72 % / 63 % / 76 % of the cheap→expensive gap on scenarios 32/33/34.

---

## 5. Bugs already found and fixed — do not reintroduce

| bug | fix |
|---|---|
| Open3D has no Python 3.14 wheels | custom `ply_io.py` + `scipy.cKDTree` (~1000× faster) |
| `all()` short-circuit in the audit skipped the second split | use a list |
| `_load_maps` loaded unconditionally — would crash once scenario 34 entered | mask-driven zeros + shape probing, regression-tested |
| All 8 ablation configs reported identical 1.593 GFLOPs | `load_checkpoint` did not enable encoder skipping; `--skip-unavailable-encoders` now recorded in the checkpoint. Cost now spans 0.39–94.1 GFLOPs |
| Bare `runs/` in `.gitignore` matched at any depth, silently excluding per-run logs | anchored to `/runs/` |
| Checkpoints 189 MB / 567 MB exceed GitHub's 100 MB limit | `results/**/*.pt` gitignored — **checkpoints are not in the repo** |
| Kaggle picked the wrong `samples.csv` when a results dump was attached | `find_index` scores candidates by co-located tensor sources (3/3 vs 0/3) |
| Non-idempotent Kaggle staging (`/kaggle/working` persists; `.exists()` follows symlinks) | validate and re-stage |
| Output buffering made a healthy run look stalled for 40 min | `python -u` |
| LiDAR binomial thinning guarded on a running total (retained 67 % at keep=0.5) | index return slots; now exact to 1 % |
| **Oracle ordering**: a single `ORACLE` curve ordered by the Top-1 criterion was used as the DBA ceiling | DBA is maximised by ordering on per-sample DBA *gain*. Builder now emits `ORACLE (Top-1 optimal)` and `ORACLE (DBA optimal)` separately, each plotted only on its own metric's panel |

**The oracle bug is worth understanding**: the tell was `regress_gain` scoring
0.8271 against an "oracle" of 0.8236 — a gate beating its own ceiling, which is
impossible. Both completed runs predate the fix, so their `operating_curves.csv`
still carries the mislabelled curve;
`full_run/operating_curves_oracle_corrected.csv` is an exact recomputation from
the per-sample tables.

---

## 6. Methodology notes that took real work to establish

- **`skip_unavailable_encoders` is numerically equivalent** (Δlogits = 0.0) and
  is what makes FLOPs accounting honest.
- **Camera brightness and contrast degradations are provable no-ops.** Eq. (10)
  standardises each image by its own mean and std, which is affine-invariant:
  `x → a·x + b` becomes `(a·x + b − a·mean − b)/(a·std) = (x − mean)/std`.
  Verified numerically to 4e-7. Any lighting experiment on this architecture
  measures nothing.
- **Radar degradation is not implementable faithfully.** The stored tensor is a
  post-2D-FFT magnitude pair under a joint min-max; map-domain noise has no
  clean pre-image in the IQ cube. Radar unreliability is studied through the
  dataset's *own* missingness instead (scenario 34 is ~40 % radar).
- **LiDAR range truncation is near-vacuous** — the BEV is already cropped to the
  ±50 m ROI holding 99 % of points.
- **LiDAR point dropout is exact**, not approximate: BEV cell value is
  `count/5` with integer count, so binomial thinning of counts *is* "each return
  independently lost".
- **Noise and blur are indistinguishable by sharpness alone**, so `quality.py`
  adds a two-scale `img_hf_ratio` (blur drives it 0.30→0.005, noise 0.30→12.6).
- **`img_contrast` was removed** — eq. (10) pins it at 1.0, so it carries no
  information.
- **Processing-cost vs sensing-cost** decides which gate features are
  admissible. Under a *processing*-cost model (what we measure — forward
  GFLOPs), computing a cheap image statistic and then deciding whether to run
  ResNet34 is a genuine saving. Under a *sensing*-cost model it is not, because
  the camera must be read first. `quality.FEATURE_GROUPS` splits features into
  A/B (no expensive sensor read) and C (image/LiDAR statistics) so results can
  be read either way.

---

## 7. State of the work

| | |
|---|---|
| ✅ | Preprocessing, audit, three data defects found |
| ✅ | AMBER baseline implemented and trained |
| ✅ | Modality ablation, 8 configurations, Pareto frontier |
| ✅ | Oracle routing — headroom real, confidence gate weak |
| ✅ | Faithful degradations + 31 inference-available quality features |
| ✅ | **Learned adaptive gate — beats confidence, full run complete** |
| ❌ | Difficulty-aware gating — **ruled out**, do not build |
| 🔄 | `modality_robustness.ipynb` — **built, not yet run** |
| 🔄 | `modality_quality_signal.ipynb` — **built, not yet run** |
| ⬜ | Scenario 31 full release (7,012 samples) |
| ⬜ | Multiple seeds / error bars — everything is currently a single run |
| ⬜ | Validate the AMBER reimplementation against the paper's reported numbers |
| ⬜ | Official GPS-only LSTM baseline |

### Immediate next actions, in priority order

1. **Run the two built notebooks** (`modality_robustness`,
   `modality_quality_signal`) on Kaggle. Set `QUICK_RUN = False`. They need the
   preprocessed dataset plus the 8 checkpoints attached as Kaggle inputs.
2. **Get the standalone scenario-31 release (7,012 samples).** The camera
   generalisation trap — the most interesting finding — currently rests on
   n=50. This converts an anecdote into a headline result.
3. **Add seeds.** 3–5 per configuration, report mean ± std. Some Pareto
   orderings (0.8835 vs 0.8759) may not survive this; better to find out
   internally. This is the cheapest fix with the largest credibility return.
4. **Resolve the absolute-accuracy question.** Top-1 0.4604 is low for 64-beam
   DeepSense. Either the block-based leak-free split is genuinely harder than
   what others report on (a defensible position that must be *argued*), or the
   AMBER reimplementation underperforms. This is the highest-priority unknown —
   if it is the latter, every comparison rests on a weak base.
5. **Benchmark the gate against the real literature** — cascades, early-exit
   (BranchyNet, MSDNet), learning-to-defer — not only against a confidence
   threshold. "Learned gate beats confidence" is known there; the novelty here
   is the multimodal sensor-cost framing.

### Publication read

Two papers are available. The **stronger** one is *not* the routing paper — it
is "most modalities in multimodal beam prediction don't pay for themselves, and
the one that pays most in-domain generalises worst", with the routing work as
its constructive final section. That framing is contrarian, actionable, and cuts
against the field's push toward more modalities.

Realistic: workshop near-certain; a solid conference paper (ICC/GLOBECOM, IEEE
WCL) around a coin flip now and clearly better than even after items 2–3; a top
journal (TWC/JSAC) a real but not likely target, gated mostly on scenario 31 and
the absolute-accuracy question. Top-tier ML venues are a venue mismatch, not a
quality problem — don't spend effort there.

---

## 8. How to run things

```bash
# preprocessing (local, CPU, ~hours)
bash preprocessing_amber/run_preprocessing.sh

# regenerate notebooks after editing a builder
python notebooks/_build_adaptive_nbs.py

# tests
pytest

# training (do this on a GPU, not locally)
python -u -m amber.train --modalities gps,image --skip-unavailable-encoders
```

**On Kaggle**, attach the preprocessed dataset and the checkpoint dataset, then
run all cells. If inputs look wrong, paste `notebooks/kaggle_input_report.py`
into a cell — it prints the input tree and a READY/NOT-READY verdict.

Checkpoint slugs are modality lists joined by `_`, e.g. `gps`, `gps_image`,
`gps_radar_image_lidar`. Discovery handles both `runs/<slug>/best.pt` and
`<slug>.pt`.

**Checkpoints are not in this repo** (GitHub's 100 MB limit). They live in the
Kaggle dataset.
