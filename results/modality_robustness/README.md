# Modality robustness (RQ1 + RQ2) — degradation and missing modalities

Full run, `QUICK_RUN = False`, all 2,198 `val` samples at every point.
Produced by [`notebooks/modality_robustness.ipynb`](../../notebooks/modality_robustness.ipynb).

Four findings, two of which change how earlier results should be read.

---

## 1. Camera is fragile — modest degradation makes it worse than free GPS

The reference is GPS alone: **DBA 0.7298 at 0.39 GFLOPs**. GPS + Camera starts
at 0.8835 for 48.24 GFLOPs. The question is how much degradation it takes before
that 124× compute premium buys *nothing*.

| degradation | crosses GPS-alone at | in plain terms |
|---|---|---|
| Gaussian blur | **σ ≈ 1.3 px** | mild defocus on a 256×256 frame |
| additive noise | **σ ≈ 0.084** | ~8 % intensity noise; a dim night scene |
| occlusion | **≈ 17.5 % of frame** | a dirt patch or a raindrop |
| resolution | **≈ 0.42 downscale** | roughly halving linear resolution |

Full curves (DBA):

| severity → | intact | | | | |
|---|---|---|---|---|---|
| blur σ | 0 → 0.8835 | 1 → 0.7874 | 2 → 0.6076 | 4 → 0.4655 | 8 → 0.3197 |
| noise σ | 0 → 0.8835 | 0.05 → 0.8485 | 0.10 → 0.6761 | 0.20 → 0.2951 | 0.40 → 0.1960 |
| occlusion | 0 → 0.8835 | 0.05 → 0.8627 | 0.15 → 0.7892 | 0.35 → 0.3185 | 0.60 → 0.2171 |
| resolution | 1.0 → 0.8835 | 0.5 → 0.7786 | 0.25 → 0.6327 | 0.125 → 0.4720 | 0.0625 → 0.3187 |

**Why this matters for routing.** These are the thresholds a cost-aware router
should trigger on, and they are *not* extreme. A camera with σ=2 blur is still
a perfectly normal-looking image, and at that point running it costs 124× more
than GPS and performs **worse**. The sensor-quality features in
`amber/quality.py` exist precisely to detect this regime before paying for the
encoder.

---

## 2. The GPS-only cheap tier depends on GPS precision far beyond consumer GNSS

This qualifies one of the project's headline claims.

| GPS error | GPS alone | GPS + Camera |
|---|---|---|
| 0 m | 0.7298 | 0.8835 |
| 1 m | 0.6680 | 0.8827 |
| 2 m | 0.6054 | 0.8801 |
| **5 m** | **0.4367** | **0.8743** |
| 10 m | 0.2990 | 0.8645 |
| 20 m | 0.1610 | 0.8360 |

**Consumer GNSS is 1–5 m.** At 5 m the GPS-only tier falls from 0.7298 to
**0.4367** — it loses 40 % of its DBA. The "GPS alone gets 83 % of full accuracy
at 0.4 % of the compute" result is **true for this dataset's high-precision
positioning and should not be quoted without that qualification.** DeepSense
GPS is far more precise than a phone's.

**The converse is just as interesting: camera makes the system GPS-robust.**
GPS + Camera barely moves across the whole sweep — 0.8835 → 0.8360 at 20 m of
error. Camera is not only the accuracy modality, it is the *redundancy*
modality. That is a second, independent argument for it, and it partly offsets
the generalisation trap.

---

## 3. AMBER's availability masking does **not** give graceful degradation

RQ2 asked whether masking a modality at inference matches training a model for
that subset. Masked full model vs independently trained, DBA:

| modalities | masked | trained | gap |
|---|---|---|---|
| gps + radar | 0.3746 | 0.7794 | **−0.4048** |
| gps + lidar | 0.4117 | 0.7770 | **−0.3654** |
| gps + radar + lidar | 0.4386 | 0.8025 | **−0.3639** |
| gps + image | 0.8703 | 0.8835 | −0.0132 |
| gps + radar + image | 0.8756 | 0.8789 | −0.0033 |
| gps + radar + image + lidar | 0.8759 | 0.8759 | 0.0000 |
| gps + image + lidar | 0.8747 | 0.8727 | +0.0020 |

**The pattern is exact: masking is fine whenever camera survives, and
catastrophic whenever it does not.** Lose the camera and the full model drops to
0.37–0.44 DBA, while a model actually *trained* without a camera reaches
0.78–0.80. A gap of 0.40 DBA.

The eq. (3) availability mechanism is presented as handling missing modalities.
It handles *redundant* missing modalities. It does not survive losing the one
modality the model actually learned to use — and for a reliability argument,
that is the case that matters.

**Practical consequence for routing:** you cannot build a cheap tier by masking
the expensive model. Tiers must be independently trained. That is what we do,
so our results are unaffected — but it closes off the cheaper alternative and is
worth stating.

### Which modalities the full model actually uses

Dropping one modality from the full model, DBA 0.8759 intact:

| dropped | DBA | change |
|---|---|---|
| lidar | 0.8756 | −0.0003 |
| radar | 0.8747 | −0.0012 |
| gps | 0.8545 | −0.0214 |
| **image** | **0.4386** | **−0.4373** |

LiDAR and radar are, within noise, **doing nothing at all**. This independently
confirms the ablation's conclusion by a completely different route.

---

## 4. Natural missingness is confounded — do not treat it as random

Samples where radar is genuinely absent score **higher**, not lower:

| radar present | n | Top-1 | DBA |
|---|---|---|---|
| no | 453 | 0.4570 | **0.9046** |
| yes | 1,745 | 0.4355 | 0.8684 |

Radar missingness is concentrated in scenario 34 and so correlates with
scene difficulty rather than being random. **Natural missingness cannot be used
as a proxy for synthetic masking**, and any robustness claim built on it would
be measuring scenario composition instead of sensor loss.

---

## Files

```
full_run/
├── degradation/
│   ├── config.json              sweep definitions
│   ├── degradation_val.csv      37 rows: slug, kind, severity, Top-1/3/5, DBA, n, seconds
│   └── plots/degradation_curves.png
└── missing_modality/
    ├── masked_full_model_val.csv      every subset masked out of the full model
    ├── masking_vs_training_val.csv    masked vs independently trained
    └── natural_missingness_val.csv    radar present vs naturally absent
```

Degradations were chosen for faithfulness to the stored representation; several
were rejected with reasons recorded in `amber/degrade.py` (camera brightness and
contrast are provable no-ops under eq. 10; radar map-domain noise has no
physical pre-image; LiDAR range truncation is near-vacuous).
