# Project Overview — Multimodal mmWave Beam Prediction

High-level summary of the work completed so far: what the raw dataset contains,
what preprocessing was built, what the processed datasets hold, and what to put
in front of an audience.

**Status: preprocessing complete and verified. No models trained yet.**

- Task: predict the best beam of a **64-beam codebook** from multimodal sensing
- Dataset: **DeepSense6G 2022 Multi-Modal Beam Prediction**, scenarios **31–34** only
- Two independent preprocessing pipelines built, one per planned baseline
- Three data-integrity defects found in the public release (see §5)

---

## 1. The problem in one paragraph

A 60 GHz base station with 16 antennas and a 64-beam codebook serves a moving
vehicle. Choosing the right beam by exhaustive search is expensive, so instead we
predict it from sensors the BS already has: a camera, a LiDAR, a radar, and the
vehicle's GPS relayed over a sub-6 GHz control link. Each sample is a short
window of history — 5 camera frames, 5 LiDAR sweeps, 5 radar cubes, 2 GPS
positions — and the label is the beam that actually maximised received power.
The eventual research question is whether the expensive sensors (camera, LiDAR)
can be activated only when they are actually needed, holding accuracy while
cutting average sensing cost.

---

## 2. What is in the raw dataset

Two official releases, 25 GB total. They were originally misnamed on disk
(`Training_dataset` / `Testing_dataset`) and were renamed to what they actually
are — the second one is the small *adaptation* set, not a test set.

| On disk | Official name | Scenarios | Samples | Size |
|---|---|---|---|---|
| `data/raw/development/` | `Multi_Modal` dev set | 32, 33, 34 | 11,143 | 24 GB |
| `data/raw/adaptation/` | `Adaptation_dataset_multi_modal` | 31, 32, 33 | 100 | 1.5 GB |

The challenge's unlabelled `Multi_Modal_Test` split is **not** part of this copy,
so all train/val/test splits are cut from the development set.

### Per-modality format (verified on disk, not assumed)

| Modality | File | Content |
|---|---|---|
| Camera | `image_<f>.jpg` | 960×540 RGB |
| Radar | `radar_data_<f>.npy` | `(4, 256, 250)` complex64 — 4 RX antennas × 256 samples/chirp × 250 chirps |
| LiDAR | `lidar_data_<f>.ply` | ASCII point cloud, 16k–18k points, `x,y,z` + intensity |
| GPS (BS) | `gps_location.txt` | one static lat/lon per scenario |
| GPS (vehicle) | `GPS_location_<f>.txt` | per-frame lat/lon |
| Power | `mmWave_power_<f>.txt` | **64 floats — the beam power vector** |
| Label | `unit1_beam` column | 1-indexed beam = `argmax(power) + 1` |

### Raw file counts

| Scenario | Camera | LiDAR | Radar | Power | Vehicle GPS |
|---|---|---|---|---|---|
| dev / 32 | 3,235 | 3,235 | 3,235 | 3,115 | 3,145 |
| dev / 33 | 3,981 | 3,981 | 3,981 | 3,837 | 3,873 |
| dev / 34 | 4,439 | **1,007** | **0** | **0** | **0** |
| adapt / 31 | 243 | 243 | 243 | 50 | 100 |
| adapt / 32 | 118 | 118 | 118 | 25 | 49 |
| adapt / 33 | 125 | 125 | 125 | 25 | 50 |

Two structural facts worth internalising: **scenario 34 is badly incomplete**,
and **scenario 31 exists only in the 100-sample adaptation set** — there is no
scenario-31 training data anywhere in this release.

---

## 3. What preprocessing was built

Raw sensor data is unusable as-is: radar is a complex IQ cube, LiDAR is an
unordered point list, and neither has a fixed shape a CNN can consume. Two
pipelines convert them, each matching a different baseline paper. They are kept
separate because their tensors are **mutually incompatible**, not because of
duplication.

| | `preprocessing/` | `preprocessing_amber/` |
|---|---|---|
| Follows | TII challenge solution (ref. paper 2) | AMBER (ref. paper 1) |
| Feeds | Baseline 3 | Baseline 4 (main baseline) |
| Output | `data/processed/` (4.9 GB) | `data/processed_amber/` (6.2 GB) |

### Per-modality transforms

| Modality | TII pipeline | AMBER pipeline |
|---|---|---|
| **Radar** | 1D range FFT → **subtract chirp mean (clutter removal)** → 1D angle FFT. Two files, each normalised **independently** | true **2D FFT**, **no clutter removal**. Range-angle + range-velocity stacked as **2 channels of one tensor**, **single joint** min-max |
| | → 2 × `(256,256)` float32 | → `(2,256,256)` float32 |
| **LiDAR** | Estimate the **static background** per scenario, subtract it with a KD-tree. Keeps the vehicle, drops the street | **BEV point-count histogram**, capped at 5 points/cell, **no background model** |
| | → filtered `.ply`, ~2.7 % of points survive | → `(1,256,256)` float32 |
| **Camera** | 7 photometric augmentation variants | **no augmentation**, per-image mean/std at load time |
| **GPS** | raw lat/lon | **Cartesian metres relative to the BS**, min-max normalised on train only |
| **Beam history** | not used | **4 past beam indices become an input modality** |
| **Missing data** | not modelled | 5-element availability mask per sample |

### Engineering notes

- **Open3D was dropped.** It has no wheels for Python 3.14, so its two call
  sites were replaced by a small PLY reader/writer plus `scipy.cKDTree`. The
  KD-tree path is also ~1000× faster than the original per-point Python loop.
- **torchvision was dropped** from preprocessing; `PIL.ImageEnhance` does the
  same photometric operations. `torch` is only needed later, for training.
- Every stage is **resumable and idempotent** — re-running only does missing work.
- One driver script per pipeline: `run_preprocessing.sh`.

---

## 4. What is in the processed datasets

Both pipelines emit per-frame tensors plus a single **index CSV** that is the
one thing training code should read. The index resolves every modality path,
carries the label, and defines the splits — so no dataloader has to re-derive
sequence structure or re-parse the official CSVs.

### `data/processed/` — TII pipeline

```
lidar_background/scenario3X_background.ply     the 4 estimated static backgrounds
<split>/<scenario>/radar_ang|radar_vel/*.npy   (256,256) float32 in [0,1]
<split>/<scenario>/lidar/*.ply                 background-filtered point clouds
<split>_aug/...                                augmented adaptation split
index/samples.csv                              7,052 rows x 40 cols
index/beam_pwr.npy                             (7052, 64) float32 power vectors
```

Splits: `train` 4,794 / `val` 1,000 / `test` 1,100 / `adaptation` 100 /
`excluded_nan_pwr` 58. **6,894 usable samples.**

This index also carries **beam-ambiguity metrics** for the planned difficulty
analysis, computed from the power vector and therefore for **offline use only**:
`margin_db`, `entropy_bits`, `n_within_3db`, `n_within_10pct`.

### `data/processed_amber/` — AMBER pipeline

```
<split>/<scenario>/radar_ra_rv/*.npy     (2,256,256) float32 in [0,1]
<split>/<scenario>/lidar_bev/*.npy       (1,256,256) float32 in [0,1]
<split>/<scenario>/camera/*.jpg          256x256 RGB cache
index/samples.csv                        11,243 rows x 40 cols
index/beam_pwr.npy                       (11243, 64) float32
index/gps_norm.json                      train-fit GPS min/max
index/image_stats.csv                    per-frame channel mean/std
```

Splits: `train` 5,544 / `test` 1,350 / `adaptation` 100 /
`excluded_nan_pwr` 58 / `no_target` 4,191. **6,894 usable samples** — verified
to be the *identical* sample set with *identical* labels to the TII index, just
split 80/20 instead of 70/15/15. So any accuracy gap between Baseline 3 and
Baseline 4 is attributable to representation and architecture, not to different
data.

### Index columns, grouped

| Group | Columns | Purpose |
|---|---|---|
| Identity | `sample_id`, `split_src`, `scenario`, `frame` | joins, per-scenario metrics |
| Modality paths | 15 cols: `image_k`, `lidar_bev_k`, `radar_k` for k=1…5 | the W=5 window; k=5 is the current instant |
| GPS | `ue_x_1/2`, `ue_y_1/2`, `bs_lat`, `bs_lon` | vehicle position in metres from the BS |
| Beam history | `beam_hist_1..4`, `n_beam_hist_available` | AMBER input modality; `-1` = unavailable |
| Target | `beam`, `beam_official`, `pwr_n_nan` | the label, plus an audit trail |
| Availability mask | `m_image`, `m_lidar`, `m_radar`, `m_beam`, `m_gps` | drives missing-modality training |
| Bookkeeping | `pwr_row`, `split` | row into `beam_pwr.npy`; split assignment |

### Split policy — the one methodological choice that matters

Consecutive DeepSense6G frames overlap in time and are strongly correlated, so a
**per-sample random split leaks the test set into training**. Both pipelines
therefore assign **contiguous 50-frame blocks** whole to train/val/test.

Verified: **zero frame overlap** between splits. Residual correlation sits only
at block boundaries — 72/500 scenario-32 and 51/600 scenario-33 test frames lie
within 4 frames of a training frame.

AMBER's paper says "randomly divided into 80/20, each corresponding to an
independent vehicle pass-by event" — two clauses that conflict, and pass-by
events are not recoverable here (trajectory-jump detection finds only 2–4 macro
sessions per scenario). So `--split-mode` offers `block` (default),
`session`, and `random` (faithful to the paper's wording, but leaks). **Always
state which mode produced a number.**

---

## 5. Three defects found in the public dataset

None are caused by our code; all were found by `audit_dataset.py`.

**1 — Scenario 34 is unusable as downloaded.** Radar, vehicle GPS and the
mmWave power files are entirely absent; only 1,007 of 4,439 LiDAR clouds are
present, and one of those is truncated. No power vector means no label. It is
excluded by default: 4,191 samples lost, ~38 % of the development set.
*Fix: re-download scenario 34.*

**2 — 58 samples have corrupt labels.** Their power files contain literal `nan`
values, and for **all 58** the official `unit1_beam` equals the index of the
*first NaN* rather than the argmax of the real values. The official labels were
generated with `np.argmax` on NaN-containing vectors. This also explains why the
obvious sanity check `argmax(power)+1 == unit1_beam` passes at 100 % — both
sides carry the same bug. Anyone training straight off the official CSV trains on
58 garbage labels. We relabel and quarantine them.

**3 — The power vectors are far flatter than expected.** The whole 64-beam
vector spans only **2.6 dB** (scenario 32) to **5.6 dB** (scenario 33) between
best and worst beam, and the top-1 vs top-2 margin has a median of **0.06 dB**.
Consequence: the conventional "beams within 3 dB of the best" ambiguity measure
**saturates at 64** for most scenario-32 samples — it says every beam is
equally good. We added `n_within_10pct` (median 5), which stays discriminative.
This directly reshapes what "hard sample" can mean in the planned analysis.

---

## 6. What to show when presenting

A suggested order. The through-line is: *the data is real and messy, we
understood it, and we built the foundation correctly.*

### Slide 1 — The task
One diagram: BS with camera/LiDAR/radar, moving vehicle, 64 beams. State the
input (5+5+5 observations + 2 GPS) and output (1 of 64 beams). One sentence on
why it matters: beam search is expensive, sensors are already there.

### Slide 2 — The dataset
The scenario table from §2 with sample counts. Show **one real sample**: the
camera frame, the LiDAR point cloud, the radar cube, and the 64-bar power vector
with the argmax highlighted. This single slide proves you understand the data.

### Slide 3 — Preprocessing, visually
**Show before/after images, not equations.** Four panels:
- radar cube → range-angle + range-velocity heatmaps
- raw LiDAR cloud → background-filtered cloud (TII) *and* → BEV histogram (AMBER)
- the camera frame
- GPS track plotted in metres relative to the BS

Then one line on why two pipelines exist: different baselines demand
incompatible tensors.

### Slide 4 — Why the two pipelines differ
The comparison table from §3. Emphasise the two substantive divergences:
**clutter removal vs none** in radar, and **background subtraction vs BEV
histogram** in LiDAR. These are modelling assumptions, not implementation
details.

### Slide 5 — Data-integrity findings *(the strongest slide)*
All three defects from §5. This is original diagnostic work that a reader
cannot get from the papers. Lead with the corrupt-label finding — "the official
labels are wrong for 58 samples, and the standard sanity check cannot detect it"
is a genuinely notable result. Pair the flatness finding with a **histogram of
best-to-worst beam spread** and a note that it invalidates the textbook 3 dB
ambiguity metric.

### Slide 6 — Splits and leakage control
A small timeline graphic: consecutive frames overlap, so random splitting leaks.
Show the block-split scheme and the verified zero-overlap result. Mention that
AMBER's stated split is self-contradictory and that we support both modes. This
is the slide that signals methodological care.

### Slide 7 — What is ready, and what is next
Table of both processed datasets: sizes, sample counts, split counts, tensor
shapes. Then the roadmap: GPS-only baseline → TII Transformer → AMBER →
modality ablations → difficulty analysis → the cost-aware gate.

### One number to plant early
AMBER's own ablation (its Table V): **BeamIdx + GPS alone reaches 58.81 % Top-1
at 0.077 GFLOPs**, against 64.15 % for the full model at 47.27 GFLOPs. That is
92 % of the accuracy for 0.16 % of the compute. Show it during the motivation,
because it defines the bar the adaptive-routing idea must clear, and it makes
the cost-efficiency premise concrete rather than speculative.

### What NOT to claim yet
No accuracy numbers of our own — nothing has been trained. And no novelty claim
for adaptive modality selection until the literature comparison in the project
plan is done; the honest framing today is *"foundation built, defects found,
baselines next."*

---

## 7. Reproducing everything

```bash
python3 -m venv ~/.venvs/beamprep
~/.venvs/beamprep/bin/pip install -r requirements.txt

PY=~/.venvs/beamprep/bin/python bash preprocessing/run_preprocessing.sh
PY=~/.venvs/beamprep/bin/python bash preprocessing_amber/run_preprocessing.sh
```

Detail lives in [README.md](README.md) (raw formats, TII pipeline, defects) and
[preprocessing_amber/README.md](preprocessing_amber/README.md) (AMBER pipeline,
paper-unspecified hyperparameters, reproduction blockers).

Note: the project directory name contains a `:`, which prevents `venv` from
being created inside it — hence the venv in `~/.venvs/`. Renaming the folder to
`pe_5g_6g` would remove that wrinkle.
