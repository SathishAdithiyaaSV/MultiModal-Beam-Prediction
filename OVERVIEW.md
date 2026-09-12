# Project Overview — Multimodal mmWave Beam Prediction (AMBER)

High-level summary of the work so far: what the raw dataset contains, what
preprocessing was built, what the processed dataset holds, and what to put in
front of an audience.

**Status: preprocessing and the AMBER model are implemented and verified.
No training run completed yet.**

- Task: predict the best beam of a **64-beam codebook** from multimodal sensing
- Dataset: **DeepSense6G 2022 Multi-Modal Beam Prediction**, scenarios **31–34**
- Preprocessing and architecture both follow **AMBER** (Wen et al.)
- Three defects found in the public data release (see §6)

---

## 1. The problem in one paragraph

A 60 GHz base station with 16 antennas and a 64-beam codebook serves a moving
vehicle. Picking the right beam by exhaustive search is expensive, so instead we
predict it from sensors the BS already has: a camera, a LiDAR, a radar, and the
vehicle's GPS relayed over a sub-6 GHz control link. Each sample is a short
window of history — 5 camera frames, 5 LiDAR sweeps, 5 radar cubes, 2 GPS
positions, plus the 4 previously-used beams — and the label is the beam that
actually maximised received power. The eventual research question is whether the
expensive sensors (camera, LiDAR) can be activated only when genuinely needed,
holding accuracy while cutting average sensing cost.

---

## 2. What is in the raw dataset

Two official releases, 25 GB. They were misnamed on disk (`Training_dataset` /
`Testing_dataset`) and were renamed to what they actually are — the second is
the small *adaptation* set, not a test set.

| On disk | Official name | Scenarios | Samples | Size |
|---|---|---|---|---|
| `data/raw/development/` | `Multi_Modal` dev set | 32, 33, 34 | 11,143 | 24 GB |
| `data/raw/adaptation/` | `Adaptation_dataset_multi_modal` | 31, 32, 33 | 100 | 1.5 GB |
| `data/raw/test/` | `Multi_Modal_Test` | — | — | *to be added* |

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
unordered point list, and neither has a fixed shape a CNN can consume. The
pipeline in `preprocessing_amber/` converts all five modalities into exactly the
representations AMBER's encoders expect.

| Modality | Transform | Output | AMBER eq. |
|---|---|---|---|
| **Radar** | 2D FFT → range-angle and range-velocity maps, stacked as **2 channels of one tensor** with a **single joint** min-max. **No clutter removal.** | `(2,256,256)` float32 | 5–8 |
| **LiDAR** | **Bird's-eye-view point-count histogram**, capped at 5 points/cell. No background model. | `(1,256,256)` float32 | 11 |
| **Camera** | Resized cache; per-image mean/std standardisation at load time. No augmentation. | 256×256 RGB | 10 |
| **GPS** | lat/lon → **Cartesian metres relative to the BS**, min-max normalised on `train` only | 4 floats | 12 |
| **Beam history** | the **4 past beam indices** become an input modality | 4 ints | 13 |
| **Availability** | 5-element mask `m = [mI, mL, mR, mB, mG]` from what is on disk | 5 bits | 3 |

Stages: `audit_dataset.py` → `radar_ra_rv.py` → `lidar_bev.py` →
`image_cache.py` → `build_index.py`, all driven by `run_preprocessing.sh`.

### Engineering notes

- **Open3D was dropped.** No wheels for Python 3.14, so its two call sites became
  a small PLY reader/writer plus `scipy.cKDTree`.
- **torchvision was dropped** — AMBER applies no photometric augmentation, so PIL
  suffices. `torch` is needed only for training.
- Every stage is **resumable and idempotent**; re-running does only missing work.
- Sources are **discovered from disk**, so the pipeline runs unchanged before and
  after the official test set is added.

---

## 4. What is in the processed dataset

`data/processed_amber/`, 6.2 GB. Per-frame tensors plus a single **index CSV**
that is the only thing training code needs to read — it resolves every modality
path, carries the label, and defines the splits.

```
<source>/<scenario>/radar_ra_rv/*.npy     (2,256,256) float32 in [0,1]
<source>/<scenario>/lidar_bev/*.npy       (1,256,256) float32 in [0,1]
<source>/<scenario>/camera/*.jpg          256x256 RGB
index/samples.csv                         11,243 rows x 41 cols
index/beam_pwr.npy                        (11243, 64) float32
index/gps_norm.json                       train-fit GPS min/max  <- apply this
index/image_stats.csv                     per-frame channel mean/std
index/split_summary.csv, meta.json
```

### Index columns, grouped

| Group | Columns | Purpose |
|---|---|---|
| Identity | `sample_id`, `source`, `scenario`, `frame`, `frame_target` | joins, per-scenario metrics |
| Modality paths | 15 cols: `image_k`, `lidar_bev_k`, `radar_k` for k=1…5 | the W=5 window; **k=5 is the current instant** |
| GPS | `ue_x_1/2`, `ue_y_1/2`, `bs_lat`, `bs_lon` | vehicle position in metres from the BS |
| Beam history | `beam_hist_1..4`, `n_beam_hist_available` | AMBER input modality; **`-1` = unavailable** |
| Target | `beam`, `beam_official`, `pwr_n_nan` | 0-indexed label, plus an audit trail |
| Mask | `m_image`, `m_lidar`, `m_radar`, `m_beam`, `m_gps` | drives missing-modality training |
| Bookkeeping | `pwr_row`, `split` | row into `beam_pwr.npy`; split assignment |

### Source vs split — the thing to get right

`source` = which physical release a row came from. `split` = what it may be used
for. Two columns, so no name ever means two things:

| source | → split | n | Role |
|---|---|---|---|
| `development` | `train` | 5,544 | fit parameters |
| `development` | `val` | 1,350 | model selection, early stopping |
| `adaptation` | `adaptation` | 100 | official labelled set, held out whole |
| `test` | `test` | 0 | **official test release only** — awaiting data |
| `development` | `excluded_nan_pwr` | 58 | corrupt labels (§6) |
| `development` | `no_target` | 4,191 | scenario 34, no power files (§6) |

**6,994 usable samples.** Nothing derived from development is ever called
`test`, so any number on the `test` split is unambiguously on official held-out
data.

Adding the official test set is a drop-in: put it at `data/raw/test/` and re-run
the pipeline. It is **unlabelled**, which is handled — those rows get
`beam = -1` and stay fully indexed for inference. Verified end to end against a
staged unlabelled source.

### Why blocks, not random samples

Consecutive DeepSense6G frames overlap in time and are strongly correlated, so a
per-sample random split leaks validation data into training. Contiguous
**50-frame blocks** are assigned whole instead; verified zero frame overlap
between `train` and `val`.

AMBER's paper says "randomly divided into 80/20, each corresponding to an
independent vehicle pass-by event" — two clauses that conflict, and pass-by
events aren't recoverable here (only 2–4 macro sessions per scenario exist). So
`--split-mode` offers `block` (default), `session`, and `random` (faithful to
the wording, but leaks). **Always state which mode produced a number.**

---

## 5. The model

`amber/` implements the AMBER architecture, one module per concern:

| Stage | What it does | Paper |
|---|---|---|
| **Encoders** | ResNet34 (image), ResNet18 (lidar, radar), 3-layer MLPs (GPS, beam history) → tokens in a common 256-dim space | eqs. 9–13 |
| **Embeddings** | sinusoidal spatial + temporal position, then a learnable per-modality weight `α = softmax(w/τ)` | eqs. 16–19 |
| **Modality-specific block** | self-attention **masked to stay inside each modality**, so nothing leaks between sensors yet | eqs. 21–27 |
| **Fusion block** | a learnable fusion token cross-attends to **only the modalities that are actually present** | eqs. 28–31 |
| **CMA** | class queries per modality aligned to the fusion query by a contrastive loss — **training only**, a regulariser | eqs. 32–34 |
| **Head + loss** | 64 logits; focal loss with **Gaussian soft labels** over the codebook, plus contrastive and L2 terms | eqs. 35–36 |

Two design points worth understanding:

**Missing modalities are handled in three places at once** — the input is
zeroed, the fusion mask blocks attention to it, and it drops out of the
contrastive and regularisation terms. That is what makes the model work under
arbitrary sensor availability, and it is the hook the eventual cost-aware
routing idea plugs into.

**The loss knows beams are ordered.** Predicting beam 31 when the truth is 30
costs much less than predicting beam 5, because adjacent beams point in
adjacent directions. This matches what the DBA metric rewards.

Verified: the token budget reproduces the paper's `N = 108` exactly, all 32
missing-modality patterns produce finite outputs, gradients reach every
parameter, and the model drives 8 samples to 100 % Top-1 — the end-to-end proof
the wiring is right. 24 tests in `tests/test_amber.py` check the paper's claims,
not just that the code runs.

```bash
PY=~/.venvs/beamprep/bin/python
$PY -m amber.train --name amber-full        # 20 epochs, AdamW, cosine schedule
$PY -m pytest -q                            # the invariant suite
```

---

## 6. Three defects found in the public dataset

None caused by our code; all found by `audit_dataset.py`.

**1 — Scenario 34 is unusable as downloaded.** Radar, vehicle GPS and the power
files are entirely absent; only 1,007 of 4,439 LiDAR clouds are present, and one
of those is truncated. No power vector means no label. 4,191 samples lost, ~38 %
of the development set. *Fix: re-download scenario 34.*

**2 — 58 samples have corrupt labels.** Their power files contain literal `nan`
values, and for **all 58** the official `unit1_beam` equals the index of the
*first NaN* rather than the argmax of the real values — the labels were
generated with `np.argmax` on NaN-containing vectors. This also explains why the
obvious sanity check `argmax(power)+1 == unit1_beam` passes at 100 %: both sides
carry the same bug. Anyone training straight off the official CSV trains on 58
garbage labels. We relabel and quarantine them.

**3 — The power vectors are far flatter than expected.** The whole 64-beam
vector spans only **2.6 dB** (scenario 32) to **5.6 dB** (scenario 33) between
best and worst beam, and the top-1 vs top-2 margin has a median of **0.06 dB**.
Consequence: the conventional "beams within 3 dB of the best" ambiguity measure
**saturates at 64** for most scenario-32 samples — it says every beam is equally
good. This directly reshapes what "hard sample" can mean in the planned
difficulty analysis; prefer continuous margin/entropy measures or a tighter
threshold.

A fourth, milder issue: **beam history is unavailable for all 100 adaptation
samples** (that release ships only the target frame's power file). AMBER's mask
handles it, but adaptation results aren't comparable to development results that
include beam history.

---

## 7. What to show when presenting

Suggested order. The through-line: *the data is real and messy, we understood
it, and the foundation is correct.*

### Slide 1 — The task
One diagram: BS with camera/LiDAR/radar, moving vehicle, 64 beams. State the
input (5+5+5 observations, 2 GPS, 4 past beams) and output (1 of 64 beams). One
sentence on why: beam search is expensive, the sensors are already there.

### Slide 2 — The dataset
The scenario table from §2 with sample counts. Show **one real sample**: camera
frame, LiDAR point cloud, radar cube, and the 64-bar power vector with the
argmax highlighted. This single slide proves you understand the data.

### Slide 3 — Preprocessing, visually
**Show before/after images, not equations.** Four panels:
- radar cube → range-angle + range-velocity heatmaps
- raw LiDAR cloud → BEV histogram
- the camera frame at 256×256
- GPS track plotted in metres relative to the BS

One line per modality on what changed and why.

### Slide 4 — The five modalities AMBER fuses
Table from §3, with the availability mask called out. The point to land: AMBER
treats *missing modalities as normal*, which is what makes the cost-aware
routing idea implementable on top of it rather than a separate architecture.

### Slide 5 — Data-integrity findings *(the strongest slide)*
All three defects from §6. This is original diagnostic work a reader cannot get
from the papers. **Lead with the corrupt labels** — "the official labels are
wrong for 58 samples, and the standard sanity check cannot detect it because
both sides share the bug" is a genuinely notable result. Pair the flatness
finding with a **histogram of best-to-worst beam spread** and note that it
invalidates the textbook 3 dB ambiguity metric.

### Slide 6 — Splits and leakage control
A timeline graphic: consecutive frames overlap, so random splitting leaks. Show
the block scheme, the verified zero-overlap result, and the source-vs-split
table. Mention that AMBER's stated split is self-contradictory and that we
support both modes. This is the slide that signals methodological care.

### Slide 7 — The model
The architecture diagram from §5, and the one idea that carries the project:
AMBER treats **missing modalities as normal**, masking them out of attention
rather than imputing them. Say that the implementation is verified against the
paper's own equations (token count, mask semantics, loss form) by a test suite,
and that it can overfit a tiny batch — the standard evidence of correct wiring.

### Slide 8 — What is ready, what is next
Sizes, sample counts, split counts, tensor shapes, parameter count. Then the
roadmap: GPS-only sanity baseline → train AMBER → modality ablations →
difficulty analysis → the cost-aware gate.

### One number to plant early
AMBER's own ablation (its Table V): **BeamIdx + GPS alone reaches 58.81 % Top-1
at 0.077 GFLOPs**, against 64.15 % for the full model at 47.27 GFLOPs. That is
92 % of the accuracy for 0.16 % of the compute. Show it during the motivation —
it defines the bar the adaptive-routing idea must clear, and makes the
cost-efficiency premise concrete rather than speculative.

### What NOT to claim yet
No accuracy numbers of our own — the model is implemented and verified, but no
training run has been completed. And no novelty claim
for adaptive modality selection until the literature comparison in the project
plan is done. The honest framing today is *"foundation built, defects found,
baselines next."*

---

## 8. Reproducing everything

```bash
python3 -m venv ~/.venvs/beamprep
~/.venvs/beamprep/bin/pip install -r requirements.txt

PY=~/.venvs/beamprep/bin/python bash preprocessing_amber/run_preprocessing.sh
~/.venvs/beamprep/bin/python -m amber.train --name amber-full
```

Full detail — raw formats, per-stage maths, the five hyperparameters AMBER
leaves unspecified, and the scenario-31/34 reproduction blockers — is in
[README.md](README.md).

Note: the project directory name contains a `:`, which prevents `venv` from
being created inside it, hence the venv in `~/.venvs/`. Renaming the folder to
`pe_5g_6g` would remove that wrinkle.
