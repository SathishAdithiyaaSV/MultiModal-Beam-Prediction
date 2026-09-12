# Multimodal mmWave Beam Prediction — AMBER

Preprocessing **and model** for the **DeepSense6G 2022 Multi-Modal Beam
Prediction** dataset, restricted to **scenarios 31–34**, implementing

> **AMBER: An Adaptive Multimodal Mask Transformer for Beam Prediction with
> Missing Modalities** — Wen, Shi, Li, Zhao, Zhao, Wang

**New here?** Read [OVERVIEW.md](OVERVIEW.md) first — a high-level tour of the
data, the pipeline and what to show when presenting. This file is the detailed
reference.

Task: predict the best beam of a **64-beam codebook** from a window of 5 camera
frames, 5 LiDAR sweeps, 5 radar cubes, 2 GPS positions and 4 past beam indices.
The label is the `argmax` of the 64-dim mmWave power vector at the current
instant.

> Scope: scenarios 31–34 of the 2022 release only. The 2023 V2V dataset and all
> other DeepSense scenarios are out of scope; `preprocessing_amber/config.py`
> hardcodes the scenario list and nothing reaches outside `data/raw/`.

---

## 1. Repository layout

```
.
├── OVERVIEW.md                   # start here
├── README.md                     # this file
├── requirements.txt
├── data/
│   ├── raw/                                  # DeepSense6G releases, git-ignored
│   │   ├── development/                      # "Multi_Modal" dev set (scen 32,33,34)
│   │   │   ├── ml_challenge_dev_multi_modal.csv
│   │   │   └── scenario3X/unit1/{camera_data,lidar_data,radar_data,mmWave_data,GPS_data}
│   │   │                    /unit2/GPS_data
│   │   ├── adaptation/                       # "Adaptation_dataset_multi_modal" (31,32,33)
│   │   └── test/                             # "Multi_Modal_Test"  [add when available]
│   └── processed_amber/                      # generated, git-ignored
│       ├── <source>/<scenario>/radar_ra_rv/  (2,256,256) float32
│       ├── <source>/<scenario>/lidar_bev/    (1,256,256) float32
│       ├── <source>/<scenario>/camera/       256x256 RGB
│       └── index/{samples.csv,beam_pwr.npy,gps_norm.json,image_stats.csv,
│                 split_summary.csv,meta.json}
├── preprocessing_amber/
│   ├── config.py                 # single source of truth: paths, sources, constants
│   ├── ply_io.py                 # numpy PLY reader/writer (replaces Open3D)
│   ├── audit_dataset.py          # raw-data verification
│   ├── radar_ra_rv.py            # 2-channel range-angle / range-velocity tensor
│   ├── lidar_bev.py              # BEV count histogram, capped at 5/cell
│   ├── image_cache.py            # resized frame cache + per-image statistics
│   ├── build_index.py            # sequences, GPS, beam history, mask, splits
│   └── run_preprocessing.sh      # end-to-end driver
├── amber/                        # the model
│   ├── config.py                 # hyperparameters; [Table II] vs [UNSPECIFIED]
│   ├── encoders.py               # modality encoders, eqs. (9)-(13)
│   ├── embeddings.py             # positional embeddings + weight indicator, (16)-(20)
│   ├── transformer.py            # masked MSA / MCA blocks, (21)-(31)
│   ├── cma.py                    # class-former alignment, (32)-(34)
│   ├── model.py                  # assembly + prediction head
│   ├── losses.py                 # focal w/ soft labels, (35)-(36)
│   ├── metrics.py                # Top-K and DBA, (37)-(39)
│   ├── dataset.py                # torch Dataset over the processed index
│   └── train.py                  # training / evaluation entry point
├── tests/test_amber.py           # 24 invariants checked against the paper
├── pytest.ini
└── references/                   # reference-paper index (PDFs are git-ignored)
```

The raw directories were renamed from `Training_dataset` / `Testing_dataset` to
`development` / `adaptation`, because that is what they are: the second holds
`ml_challenge_data_adaptation_multi_modal.csv`, i.e. the 100-sample
**adaptation** set, not a test set. See §5 for how the real test set slots in.

---

## 2. Raw data format (verified on disk, not assumed)

| Modality | Path | Shape / format |
|---|---|---|
| Camera | `unit1/camera_data/image_<f>.jpg` | 960×540 RGB JPEG |
| Radar | `unit1/radar_data/radar_data_<f>.npy` | `(4, 256, 250)` complex64 — 4 RX antennas × 256 samples/chirp × 250 chirps |
| LiDAR | `unit1/lidar_data/lidar_data_<f>.ply` | ASCII PLY, ~16k–18k points, `double x,y,z` + `ushort intensity` |
| GPS (BS) | `unit1/GPS_data/gps_location.txt` | one static `lat`/`lon` per scenario |
| GPS (vehicle) | `unit2/GPS_data/GPS_location_<f>.txt` | per-frame `lat`/`lon` |
| Power | `unit1/mmWave_data/mmWave_power_<f>.txt` | 64 floats — the beam power vector |
| Label | `unit1_beam` column | **1-indexed** beam, `== argmax(power) + 1` |

Frame stride inside a 5-observation window is **2** in the development set and
**1** in the adaptation set. Camera/radar/LiDAR share frame indices, and
`unit2_loc_1/2` align with observations 1–2. All asserted by `audit_dataset.py`.

---

## 3. Environment

`open3d` has **no wheels for Python 3.14** (the interpreter here), and the
project directory name contains a `:`, which `venv` refuses to create inside.
So the venv lives outside the project and Open3D/torchvision are not used:

```bash
python3 -m venv ~/.venvs/beamprep
~/.venvs/beamprep/bin/pip install -r requirements.txt
```

| Would-be dependency | Replacement | Note |
|---|---|---|
| `open3d.io.read/write_point_cloud` | [ply_io.py](preprocessing_amber/ply_io.py) | keeps x/y/z only, exactly as Open3D's `PointCloud` does |
| `torchvision.transforms.functional` | `PIL` | AMBER applies no photometric augmentation anyway |

`torch` is needed only later, for training — not for preprocessing.

---

## 4. Commands

```bash
cd "/Users/rakshith/Desktop/pe_5g:6g"
PY=~/.venvs/beamprep/bin/python bash preprocessing_amber/run_preprocessing.sh
```

Or step by step:

```bash
PY=~/.venvs/beamprep/bin/python

# 0. verify the raw data: existence, shapes, temporal alignment, labels
$PY preprocessing_amber/audit_dataset.py            # --quick skips the .ply scan

# 1. radar -> (2,256,256) jointly-normalised RA/RV tensors      (eqs. 5-8)
$PY preprocessing_amber/radar_ra_rv.py --source all

# 2. lidar -> (1,256,256) BEV histograms, capped at 5/cell      (eq. 11)
$PY preprocessing_amber/lidar_bev.py --source all

# 3. camera -> resized frame cache + per-image mean/std         (eq. 10)
$PY preprocessing_amber/image_cache.py --source all

# 4. index: W=5 sequences, GPS Cartesian, beam history, mask, splits
$PY preprocessing_amber/build_index.py
```

Every stage is idempotent and skips existing outputs, so re-running the driver
after a partial run only does the missing work.

Flags: `--source {development,adaptation,test,all}`, `--scenario scenario33`,
`--n-jobs N`, `--overwrite`; plus `--nfft`, `--grid VG HG`, `--x-range`,
`--y-range`, `--z-range`, `--max-per-cell`, `--image-size W H` (`0 0` = native),
`--split-mode`, `--train-frac`, `--block`, `--seed`.

---

## 5. Source vs split — read this before reporting any number

The word "test" is easy to overload here, so the index keeps two separate
columns.

**`source`** is the physical DeepSense6G release a row came from.
**`split`** is what the row may be used for.

| source (on disk) | → split | Role |
|---|---|---|
| `development` | `train` | fit model parameters |
| `development` | `val` | model selection, early stopping, hyperparameters |
| `adaptation` | `adaptation` | official 100-sample labelled set, **held out whole**; a secondary in-domain generalisation check |
| `test` | `test` | **official challenge test release only** |

Nothing derived from the development set is ever labelled `test`. So any number
reported on the `test` split is on official held-out data, with no ambiguity
about its provenance.

### Adding the official test set

Drop the release in as `data/raw/test/` (its scenario folders plus its index
CSV) and re-run the four commands in §4. Nothing else changes:

- sources are discovered from disk, so `test` is simply empty until it exists;
- the CSV name is auto-detected among the names the release has shipped under;
- the official test release is **unlabelled**. That is handled: rows with no
  power vector get `beam = -1`, all-NaN rows in `beam_pwr.npy`, and are still
  fully indexed so their tensors stay addressable for inference.

Verified end to end against a staged unlabelled source: the rows landed in
`test`, carried `beam = -1`, and zero of them leaked into `train`/`val`.

### Current split sizes

`--split-mode block --train-frac 0.8 --seed 2022`:

| split | n | scenarios |
|---|---|---|
| `train` | 5,544 | 32, 33 |
| `val` | 1,350 | 32, 33 |
| `adaptation` | 100 | 31, 32, 33 |
| `test` | 0 | *awaiting the official release* |
| `excluded_nan_pwr` | 58 | 32, 33 — corrupt labels, see §7.2 |
| `no_target` | 4,191 | 34 — no power files shipped, see §7.1 |

**6,994 usable samples.** Filter with
`~split.isin(['no_target','excluded_nan_pwr'])`.

### Why blocks, not random samples

Consecutive DeepSense6G frames overlap in time and are strongly correlated, so a
per-sample random split leaks validation data into training. Contiguous
**50-frame blocks** are therefore assigned whole. Verified: zero frame overlap
between `train` and `val`.

AMBER states the data is "randomly divided into 80 % training and 20 % testing
subsets, each corresponding to an independent vehicle pass-by event" — two
clauses that conflict, and pass-by events are not recoverable from this release
(trajectory-jump detection finds only 2–4 macro recording sessions per
scenario). `--split-mode` therefore offers:

- `block` (**default**) — contiguous 50-frame blocks. What actually controls leakage.
- `session` — whole recording sessions (frame gap > 10 starts a new one). Closest
  to "pass-by event", but only 2–4 groups per scenario.
- `random` — per-sample, faithful to the paper's wording. **Leaks.** Use only for
  like-for-like comparison against AMBER's published numbers, and say so.

---

## 6. What each stage computes

**Radar** — AMBER eqs. (5)–(8). Given `XR[t]` of shape `(4, 256, 250)`:

- range-angle: `RA = (Σ_a |FFT2D(XR[:,:,a])|)ᵀ` → `(256, NFFT)`
- range-velocity: `RV = Σ_a |FFT2D(XR[a,:,:])|` → `(256, NFFT)`
- the angle axis (4 antennas) and velocity axis (250 chirps) are zero-padded to
  `NFFT` first, so both maps share one spatial size
- the two are stacked as **2 channels of one tensor** and normalised with a
  **single joint** min-max — not one per map

Note there is **no static clutter removal**: eq. (5) has no mean-subtraction
term.

**LiDAR** — AMBER eq. (11). The point cloud is projected to a `Vg × Hg`
bird's-eye-view histogram, counts **capped at 5 per cell** "to reduce outlier
effects", then normalised. No background model. Measured: 99.2 % of points fall
inside the ROI, ~3.8 % of cells are occupied, and ~600 cells hit the cap.

*Deviation:* AMBER says only "the histogram is normalized". We divide by the cap
of 5 rather than per-frame min-max, because the cap already fixes the range and
a per-frame min-max would rescale every frame by its own densest cell,
destroying comparability across frames and scenarios.

**Camera** — AMBER eq. (10) standardises each image by its own mean and std.
That is one line in the data loader, and storing float tensors would cost ~54 GB,
so this stage only caches resized uint8 JPEGs and records per-frame mean/std in
`index/image_stats.csv`.

**GPS** — AMBER eq. (12). lat/lon → Cartesian **metres relative to the BS** via
a local equirectangular projection (sub-centimetre over a <100 m scene, and no
UTM zone bookkeeping), then min-max normalised. The min/max are fit on the
**`train` split only** and written to `index/gps_norm.json`, so val/adaptation/
test are transformed with train statistics. **Apply that file — do not
re-normalise per split.**

**Beam history** — AMBER eq. (13). The `W−1 = 4` **past** beam indices
(τ = t−4 … t−1), derived from the per-frame `mmWave_power_<f>.txt` files, which
exist independently of the CSV's single `unit1_pwr_60ghz` column.

This is side information, not leakage: it uses only τ ≤ t−1 while the target is
at t. Asserted on 400 random samples that no beam-history frame is ≥ the target
frame, and that observation 5's frame equals the target power frame.

**Availability mask** — AMBER eq. (3). `m = [mI, mL, mR, mB, mG]` per sample,
from what is actually on disk. Samples are never dropped for a missing modality;
AMBER is designed to handle that, and the mask is what its missing-modality
attention consumes.

### Index columns

| Group | Columns | Notes |
|---|---|---|
| Identity | `sample_id`, `source`, `scenario`, `frame`, `frame_target` | `frame` = observation 1; `frame_target` = instant t |
| Modality paths | `image_k`, `lidar_bev_k`, `radar_k`, k=1…5 | repo-relative; **k=5 is the current instant** |
| GPS | `ue_x_1/2`, `ue_y_1/2`, `bs_lat`, `bs_lon` | metres from the BS; `NaN` where unavailable |
| Beam history | `beam_hist_1..4`, `n_beam_hist_available` | **`-1` = unavailable**, not beam 0 |
| Target | `beam`, `beam_official`, `pwr_n_nan` | `beam` is 0-indexed; `-1` = no target |
| Mask | `m_image`, `m_lidar`, `m_radar`, `m_beam`, `m_gps` | AMBER's `m` |
| Bookkeeping | `pwr_row`, `split` | `pwr_row` indexes `beam_pwr.npy` — use the column, don't assume row order |

`beam_pwr.npy` is `(N, 64)` float32, the power vector at t. Per the project
plan it is for **offline difficulty analysis only** — never an inference input,
since it directly reveals the target.

---

## 7. Data-integrity findings

All found by `audit_dataset.py`; none caused by this pipeline.

### 7.1 Scenario 34 is incomplete as downloaded

For `development/scenario34` (4,191 samples in the CSV):

| Modality | Status |
|---|---|
| Camera | complete (4,439 files) |
| **Radar** | **entirely missing** (no `radar_data/`) |
| **LiDAR** | **1,007 of 4,439 files** — missing for ~77 % of samples |
| **GPS `unit2`** | **entirely missing** (no `unit2/`) |
| **mmWave power** | **entirely missing** (no `mmWave_data/`) |

No power vectors means no label, so scenario 34 is routed to the `no_target`
split — 4,191 samples, ~38 % of the development set. One further file,
`lidar_data_2163.ply`, is **truncated** (header declares 18,865 vertices, body
holds 10,433 values); it is reported and skipped rather than aborting the run.

**Action: re-download scenario 34.**

Note also **scenario 31 appears only in the adaptation set** (50 samples). Even
once scenario 34 is fixed, there is no scenario-31 training data in this release.

### 7.2 58 power vectors contain NaN, and their official labels are wrong

58 development samples (20 in scenario 32, 38 in scenario 33) have literal `nan`
tokens in their power file — 44 with 1 NaN, 6 with 2, 2 with 3, 6 with 9.

For **all 58**, the official `unit1_beam` equals the index of the *first NaN*,
never the argmax of the finite bins. The labels were produced by `np.argmax` on
a NaN-containing vector, which returns the first NaN's position. This is also
why the naive check `argmax(power) + 1 == unit1_beam` passes at 100 % — both
sides carry the same bug.

The pipeline relabels them with `nanargmax`, keeps the official value in
`beam_official`, records `pwr_n_nan`, and routes them to `excluded_nan_pwr`.
**Any baseline trained straight off the official CSV trains on 58 corrupt
labels.**

### 7.3 The power vectors are flatter than the 3-dB metric assumes

Best-to-worst spread across the whole 64-beam vector:

| Scenario | median | p5 | p95 |
|---|---|---|---|
| 32 | 2.60 dB | 0.76 | 6.29 |
| 33 | 5.60 dB | 1.44 | 6.65 |
| 31 (adaptation) | 4.41 dB | 3.17 | 6.46 |

The top-1 vs top-2 margin has median **0.06 dB**. So the conventional "beams
within 3 dB of the best" ambiguity measure **saturates at 64** for most
scenario-32 samples — it reports that every beam is equally good. Prefer
continuous measures (top-1/top-2 margin in dB, power-distribution entropy) or a
tighter threshold such as "within 10 % of best power", which has a median of 5.

Worth confirming against the DeepSense6G documentation before building the
beam-difficulty analysis, since it changes what "ambiguous sample" can mean.

### 7.4 Beam history is unavailable on the entire adaptation set

0 of 100 adaptation samples have usable beam history: that release ships only
the target frame's power file, not the preceding frames. Development is fine at
~96 %.

AMBER's mask handles it, but `m_beam = 0` there, so adaptation-split results are
**not comparable** to development-split results that include beam history —
especially since AMBER's own ablation makes beam history its strongest cheap
signal (see below).

---

## 8. Choices AMBER leaves unspecified

All exposed as CLI flags and recorded in `index/meta.json`.

| Symbol | Default | Reasoning |
|---|---|---|
| `NFFT` | 256 | Angle axis (4 antennas) and velocity axis (250 chirps) are both zero-padded to `NFFT`, so `NFFT ≥ 250`; 256 is the smallest power of two that qualifies and matches `SR = 256` |
| `Vg × Hg` | 256 × 256 | Matches the radar map size, so all 2D encoders see one spatial grid |
| BEV ROI | x, y ∈ [−50, 50] m | Measured over 150 frames/scenario: 99 % of points fall inside ±47 m; retains 99.2 % |
| z filter | none | AMBER describes none |
| Image size | 256 × 256 | Paper gives no input resolution; `--image-size 0 0` keeps native 960×540 |

---

## 9. Reproducing AMBER's published numbers — two blockers

AMBER's Table I reports **7,012 samples for scenario 31** and evaluates per
scenario on S31–S34. This copy has 50 scenario-31 samples (adaptation only, no
training data) and a scenario 34 with no radar/GPS/power. S32/S33/S34 frame
counts match exactly (3,235 / 3,981 / 4,439), which shows AMBER used the **full
per-scenario DeepSense6G releases**, not the challenge dev+adaptation CSVs.

Fetch the full scenario 31 and 34 releases before attempting to match its
per-scenario tables.

### One number to keep in view

AMBER's own ablation (its Table V): **BeamIdx + GPS alone reaches 58.81 % Top-1
at 0.077 GFLOPs**, against 64.15 % for the full model at 47.27 GFLOPs — 92 % of
the accuracy for 0.16 % of the compute. That is the bar any cost-aware modality
routing has to clear.

---

## 10. The model

`amber/` implements the AMBER architecture. One module per concern, each
docstring citing the equations it implements.

### Architecture

```
 image (W,3,H,W)  ─ ResNet34 ─┐
 lidar (W,1,H,W)  ─ ResNet18 ─┤  adaptive pool (VA,HA) → spatial flatten → MLP
 radar (W,2,H,W)  ─ ResNet18 ─┤                                    eqs. (9)-(11)
 beam  (W-1, 2)   ─ 3-layer MLP ┤                                  eq.  (13)
 gps   (2, 2)     ─ 3-layer MLP ┘                                  eq.  (12)
                                 │
        + sinusoidal spatial / temporal embeddings          eqs. (16)-(17)
        × modality-weight indicator  α = softmax(w/τ)       eqs. (18)-(19)
                                 │
   ┌─────────────────────────────▼──────────────────────────────┐
   │ modality-specific block — masked self-attention            │
   │ mask M[j,i] = 1 iff i == j  (block-diagonal)  eqs. (21)-(27)│
   └─────────────────────────────┬──────────────────────────────┘
                                 │ Zbar, Z'
   ┌─────────────────────────────▼──────────────────────────────┐
   │ modality-fusion block — masked cross-attention             │
   │ learnable fusion token queries the available modalities    │
   │ mask M[F,i] = 1 iff modality i available      eqs. (28)-(31)│
   └──────────┬──────────────────────────────┬──────────────────┘
              │ Zbar_F                       │ Z'_F
     prediction head (64 logits)      CMA class-formers → contrastive
        Sec. III-C                    loss, TRAINING ONLY  eqs. (32)-(34)
```

Token budget matches the paper exactly: with `VA = HA = 4`, each of image /
lidar / radar contributes 16 tokens, beam 4 and GPS 2, so the sequence is 54
long and `N = 2 × 54 = 108`, satisfying `N = 2(3·VA·HA + W + 1)`.

Objective, eq. (36): `L = 10·L_focal + 0.2·L_contrastive + 0.2·L_2`, where the
focal term uses **Gaussian soft labels** over the codebook — neighbouring beams
point in neighbouring directions, so a near miss costs less than a distant one,
which is also what DBA measures.

**Missing modalities are handled in three places**, which is what makes the
model robust to arbitrary availability patterns: the input tensor is zeroed, the
eq. (30) fusion mask blocks attention to it, and it is excluded from both the
contrastive average and the eq. (20) penalty.

### Training

```bash
PY=~/.venvs/beamprep/bin/python

# full run: AdamW, cosine schedule w/ 5 warm-up steps, 20 epochs   [Table II]
$PY -m amber.train --name amber-full

# quick smoke test on a handful of samples
$PY -m amber.train --name smoke --epochs 2 --batch-size 4     --limit-train 32 --limit-eval 32 --no-pretrained

# score an existing checkpoint
$PY -m amber.train --eval-only --checkpoint runs/amber-full/best.pt
```

Results go to `runs/<name>/`: `config.json` (every hyperparameter, including the
unspecified ones), `history.csv`, `metrics.json`, `best.pt`. Metrics are Top-1 /
Top-3 / Top-5 and DBA, reported **overall and per scenario**.

Model selection is on `val` Top-1 only. `adaptation` and `test` are scored but
never selected on.

Useful flags: `--pool` (VA = HA), `--temporal-pool {concat,mean,tokens}`,
`--modality-dropout`, `--no-pretrained`, `--device {auto,cpu,cuda,mps}`,
`--limit-train/--limit-eval`.

### Tests

```bash
~/.venvs/beamprep/bin/python -m pytest -q          # 23 fast invariants
~/.venvs/beamprep/bin/python -m pytest -q -m slow  # + the overfit check (~2 min)
```

The suite checks the paper's claims rather than just that the code runs: the
token count equals `N`, the eq. (23) mask is strictly within-modality, the
eq. (30) mask drops exactly the missing modalities, `alpha` is a distribution,
the eq. (20) penalty ignores missing modalities, focal loss orders
correct < near-miss < distant-miss, DBA equals its closed form for an off-by-one
prediction, all 32 missing-modality patterns stay finite, the CMA is active in
training and skipped at eval, and gradients reach every trainable parameter.

The slow test confirms the model can drive 8 samples to 100 % Top-1 — the
end-to-end check that the encoder → mask → fusion → head chain is wired
correctly.

---

## 11. One ambiguity in the paper worth knowing about

AMBER states two things about the temporal window that cannot both hold
literally:

- the first conv takes "two-channel radar input" and eq. (9) encodes a single
  instant `XR[t]`, implying one forward pass per frame;
- eq. (15) gives `ξ_I, ξ_L, ξ_R ∈ R^{VA·HA × C}` with no `W` factor, and
  `N = 2(3·VA·HA + W + 1)` where the `W+1 = 6` accounts only for the beam (4)
  and GPS (2) tokens.

So the `W` per-frame feature maps must collapse into `VA·HA` tokens, but the
paper never says how, and its temporal positional embedding is described only
for the beam and GPS modalities.

`--temporal-pool` selects the reconciliation. The default `concat` encodes each
frame at the stated channel count, tiles the `W` feature maps along the width
axis, then applies a single adaptive pool — keeping both stated facts true,
preserving temporal structure, and yielding exactly `VA·HA` tokens. `mean`
averages over time; `tokens` keeps `W·VA·HA` tokens but then `N` no longer
matches the paper. Report which you used.

---

## 12. Next steps

Official GPS-only LSTM baseline as a label/evaluation sanity check → train
AMBER → modality ablations → the difficulty and marginal-utility analysis →
only then the adaptive cost-aware gate.
