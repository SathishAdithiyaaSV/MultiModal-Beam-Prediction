# AMBER-style preprocessing

A second, independent preprocessing pipeline implementing the representations
described in **AMBER: An Adaptive Multimodal Mask Transformer for Beam
Prediction with Missing Modalities** (Wen, Shi, Li, Zhao, Zhao, Wang), for
DeepSense6G scenarios 31–34.

This is deliberately **not** a modification of [../preprocessing/](../preprocessing/)
(the TII challenge pipeline). AMBER's per-modality representations differ enough
that the two produce incompatible tensors, so they live side by side and write
to separate output trees. Use this one for the AMBER baseline (Baseline 4) and
the TII one for Baseline 3.

---

## 1. What AMBER does differently

| Modality | TII pipeline (`../preprocessing`) | AMBER pipeline (here) |
|---|---|---|
| **Radar** | 1D range FFT → **subtract per-antenna chirp mean (clutter removal)** → 1D angle FFT. Two separate files, each min-max normalised **independently** | True **2D FFT**, **no clutter removal**. RA and RV stacked as **2 channels of one tensor** with a **single joint** min-max (eqs. 5–8) |
| **LiDAR** | Static background estimated per scenario, subtracted with a KD-tree; emits filtered `.ply` point clouds (~2.7 % of points survive) | **BEV point-count histogram**, counts **capped at 5 per cell**, no background model at all (eq. 11) |
| **Camera** | 7 photometric augmentation variants per frame | **No augmentation.** Per-image mean/std standardisation only (eq. 10) |
| **GPS** | raw lat/lon passed through | lat/lon → **Cartesian metres relative to the BS**, then min-max normalised (eq. 12) |
| **Beam history** | not used | **4 past beam indices are an input modality** (eq. 13) |
| **Availability** | not modelled | 5-element mask `m = [mI, mL, mR, mB, mG]` per sample (eq. 3) |

Output shapes: radar `(2, 256, 256)`, LiDAR BEV `(1, 256, 256)`, camera
`256×256` RGB JPEG — all three ready for the ResNet18/ResNet18/ResNet34
encoders AMBER specifies.

### Beam history is side information, not leakage

AMBER feeds the `W−1 = 4` **past** beam indices (τ = t−4 … t−1) while predicting
the beam at t. The pipeline derives them from the per-frame
`mmWave_power_<f>.txt` files, which exist independently of the CSV's single
`unit1_pwr_60ghz` column. Causality is asserted: verified on 400 random samples
that no beam-history frame is ≥ the target frame, and that observation 5's frame
equals the target power frame.

This matters because AMBER's own ablation (Table V) shows `BeamIdx+GPS` alone
reaches **58.81 % Top-1** at 0.077 GFLOPs, versus 64.15 % for the full model at
47.27 GFLOPs. Most of AMBER's accuracy comes from the two cheapest modalities —
worth keeping in mind for the cost-aware routing idea, since it sets a very
strong and very cheap floor.

---

## 2. Commands

```bash
cd "/Users/rakshith/Desktop/pe_5g:6g"
PY=~/.venvs/beamprep/bin/python bash preprocessing_amber/run_preprocessing.sh
```

Or per stage:

```bash
PY=~/.venvs/beamprep/bin/python

# 1. radar -> (2,256,256) joint-normalised RA/RV tensors      (eqs. 5-8)
$PY preprocessing_amber/radar_ra_rv.py --split all

# 2. lidar -> (1,256,256) BEV histograms, capped at 5/cell    (eq. 11)
$PY preprocessing_amber/lidar_bev.py --split all

# 3. camera -> resized frame cache + per-image mean/std       (eq. 10)
$PY preprocessing_amber/image_cache.py --split all

# 4. index: W=5 sequences, GPS Cartesian, beam history,
#    availability mask, 80/20 split
$PY preprocessing_amber/build_index.py
```

No extra dependencies beyond [../requirements.txt](../requirements.txt); `ply_io`
is imported from the sibling pipeline rather than duplicated.

Notable flags: `--nfft`, `--grid VG HG`, `--x-range`, `--y-range`, `--z-range`,
`--max-per-cell`, `--image-size W H` (`0 0` keeps native resolution),
`--split-mode {block,session,random}`, `--train-frac`, `--seed`. Every stage
skips existing outputs unless `--overwrite`.

---

## 3. Outputs

```
data/processed_amber/
├── <split>/<scenario>/radar_ra_rv/radar_data_<f>.npy   (2,256,256) f32 in [0,1]
├── <split>/<scenario>/lidar_bev/lidar_data_<f>.npy     (1,256,256) f32 in [0,1]
├── <split>/<scenario>/camera/image_<f>.jpg             256x256 RGB
└── index/
    ├── samples.csv        one row per sample: 15 modality paths, GPS metres,
    │                      beam_hist_1..4, target beam, m_* mask, split
    ├── beam_pwr.npy       (11243, 64) float32 target power vectors
    ├── gps_norm.json      train-fit min/max for the GPS Cartesian columns
    ├── image_stats.csv    per-frame channel mean/std
    ├── split_summary.csv  per split x scenario counts and mask rates
    └── meta.json          AMBER hyperparameters + paper-unspecified choices
```

~6.2 GB total. Verified: all 15 modality paths resolve for 400 random samples,
`beam == nanargmax(power)`, and zero train/test frame overlap.

| split | n | note |
|---|---|---|
| `train` | 5544 | scenarios 32, 33 |
| `test` | 1350 | scenarios 32, 33 |
| `adaptation` | 100 | scenarios 31, 32, 33 — held out whole |
| `excluded_nan_pwr` | 58 | corrupt power vectors (see [../README.md](../README.md) §7.2) |
| `no_target` | 4191 | scenario 34 — no power files shipped (§7.1) |

---

## 4. Choices the paper does not specify

All five are exposed as CLI flags and recorded in `meta.json`.

| Symbol | Our default | Reasoning |
|---|---|---|
| `NFFT` | 256 | The angle axis (4 antennas) and velocity axis (250 chirps) are both zero-padded to `NFFT`, so `NFFT ≥ 250`; 256 is the smallest power of two that qualifies and matches `SR = 256`, giving square tensors |
| `Vg × Hg` | 256 × 256 | Matches the radar map size, so all 2D encoders see the same spatial grid |
| BEV ROI | x, y ∈ [−50, 50] m | Measured over 150 frames per scenario: 99 % of points fall inside ±47 m. Retains 99.2 % of points |
| z filter | none | AMBER describes none |
| Image size | 256 × 256 | Paper gives no input resolution. `--image-size 0 0` keeps the native 960×540 |

---

## 5. Two deviations, both deliberate

**BEV normalisation.** AMBER says only that "the histogram is normalized". We
divide by the cap of 5 rather than per-frame min-max, because the cap already
fixes the range and a per-frame min-max would rescale each frame by its own
densest cell — destroying comparability across frames and scenarios.

**Split policy.** AMBER states the dataset is "randomly divided into 80 %
training and 20 % testing subsets, each corresponding to an independent vehicle
pass-by event". Those two clauses conflict, and pass-by events are not
recoverable from this release: UE-trajectory jump detection finds only 2–4
macro recording sessions per scenario, far too few for a stable 80/20 split.
So `--split-mode` offers three options:

- `block` (**default**) — contiguous 50-frame blocks assigned whole, 80/20.
  Consecutive DeepSense6G frames overlap in time, so this is what actually
  controls leakage.
- `session` — whole recording sessions (frame-index gap > 10 starts a new one).
  Closest to "pass-by event", but coarse: 2–4 groups per scenario.
- `random` — per-sample, faithful to the paper's wording. **Leaks**: temporally
  adjacent frames end up on both sides of the split, so numbers from this mode
  are optimistic and not comparable to the TII pipeline's.

Report which mode you used. If you need to match AMBER's published numbers,
`random` is the like-for-like setting; for the thesis, `block` is defensible.

---

## 6. Blocker for reproducing AMBER's headline table

AMBER's Table I reports **7012 samples for scenario 31** and evaluates per
scenario on S31–S34. This dataset copy has:

- **scenario 31**: 243 frames / 50 samples, adaptation split only — 3 % of what
  AMBER used, and no training data at all
- **scenario 34**: no radar, GPS or power files (see [../README.md](../README.md) §7.1)

So per-scenario S31 and S34 numbers cannot be reproduced as published. AMBER
used the **full per-scenario DeepSense6G releases**, not the challenge
`dev + adaptation` CSVs — S32/S33/S34 frame counts match (3235 / 3981 / 4439),
but S31's 7012 is a separate download. Fetch the full scenario 31 and 34
releases before attempting to match Table III/IV.

Also note the beam-history modality is **unavailable on the entire adaptation
split** (0/100 samples): that release ships only the target frame's power file,
not the preceding frames. Since AMBER is built for missing modalities, the mask
handles it — but `m_beam = 0` there, so adaptation-split results are not
comparable to development-split results that include beam history.
