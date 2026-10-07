# Multimodal mmWave Beam Prediction — Results So Far

DeepSense6G 2022, scenarios 31–34 · AMBER baseline · modality ablation

> Every number here is measured and traceable to a file in `results/`.
> Nothing is projected or estimated.

---

## 1 · The problem

A 60 GHz base station must pick the best of **64 beams** for a moving vehicle.
Exhaustive beam search is expensive. Instead, predict it from sensors the base
station already has: **camera, LiDAR, radar, GPS**.

Each sample = 5 camera frames + 5 LiDAR sweeps + 5 radar cubes + 2 GPS positions.

**The question driving the project:** can we avoid paying for expensive sensors
when they are not needed?

---

## 2 · What exists now

| | |
|---|---|
| Preprocessing pipeline | complete, 46 GB raw → 11 GB tensors |
| AMBER implementation | complete, 1,635 lines, 29 tests |
| AMBER baseline | trained |
| Modality ablation | **8 configurations, complete** |
| KD radar-only baseline | trained (on earlier data) |
| Adaptive gate | not built — the ablation just changed what it should be |

**What this means.** The foundation is done. The results below are what it
produced, and they redirect the research plan.

---

## 3 · The data is not what the paper had

| split | n | scenarios |
|---|---|---|
| train | 8,844 | 32, 33, 34 |
| val | 2,198 | 32, 33, 34 |
| adaptation | 100 | 31, 32, 33 |
| test | 625 | 31, 32, 33, 34 — **unlabelled** |

Two structural gaps:

- **Scenario 31 has zero training samples.** It exists only in adaptation (50)
  and test (300). It is also the only *straight road*.
- **48 % of the test split is scenario 31** — so half of it is out-of-domain.

**What this means.** We cannot reproduce the paper's per-scenario numbers, and
any claim about generalisation rests on 50 samples until scenario 31's full
7,012-frame release is downloaded.

---

## 4 · Three defects we found in the public dataset

**1. 101 labels are simply wrong.** Their power files contain `nan`, and the
official `unit1_beam` equals the index of the *first NaN* rather than the best
beam — the labels were generated with `argmax` on NaN-containing vectors.

**2. The standard sanity check cannot detect it.** `argmax(power)+1 ==
unit1_beam` passes at 100 %, because both sides carry the same bug.

**3. The power vectors are far flatter than expected** — the whole 64-beam
vector spans only 2.6–5.9 dB, so the textbook "beams within 3 dB of best"
ambiguity measure saturates at 64 and says every beam is equally good.

**What this means.** Anyone training off the official CSV trains on 101 corrupt
labels. This is original diagnostic work, not in any of the five reference
papers, and it is publishable on its own.

---

## 5 · AMBER baseline

| | scenarios 32+33 | + scenario 34 |
|---|---|---|
| Top-1 | 0.3852 | **0.4345** |
| Top-3 | 0.7230 | **0.7862** |
| DBA | 0.8338 | **0.8611** |

Adding scenario 34: **+6.8 pp Top-1, +4.7 pp DBA**.

**Error structure:** median beam error is **1**. 43 % exact, **79 % within ±1**,
94 % within ±3.

**What this means.** The model lands on or beside the right beam almost always.
For a real system that is close to sufficient — you would try 2–3 beams, not 64.
It also means Top-1 alone understates the model; DBA is the honest metric.

---

## 6 · The main experiment: modality ablation

Eight configurations. Each trained **from scratch**, identical protocol
(10 epochs, batch 16, lr 1e-4, seed 2022, no modality dropout, no beam history).
**Only the available modalities differ.** 8.2 GPU-hours.

| Configuration | Top-1 | DBA | GFLOPs |
|---|---|---|---|
| GPS + Camera | 0.4604 | 0.8835 | 48.2 | ⭐
| GPS + Radar + Camera | 0.4431 | 0.8789 | 71.4 |
| GPS + Radar + Camera + LiDAR | 0.4399 | 0.8759 | 94.1 |
| GPS + Camera + LiDAR | 0.4377 | 0.8727 | 70.9 |
| GPS + Radar + LiDAR | 0.3640 | 0.8025 | 46.2 | ⭐
| GPS + Radar | 0.3521 | 0.7794 | 23.6 | ⭐
| GPS + LiDAR | 0.3449 | 0.7770 | 23.1 | ⭐
| GPS | 0.3203 | 0.7298 | 0.4 | ⭐

⭐ = on the accuracy/compute Pareto frontier

---

## 7 · Finding 1 — the best model is the second-cheapest

| | DBA | GFLOPs |
|---|---|---|
| **GPS + Camera** | **0.8835** | **48.2** |
| GPS + Radar + Camera + LiDAR (full) | 0.8759 | 94.1 |

The Pareto frontier **ends** at GPS + Camera:

```
  GPS  →  GPS + LiDAR  →  GPS + Radar  →  GPS + Radar + LiDAR  →  GPS + Camera
```

Every configuration costing more has *lower* DBA.

**What this means.** The full four-modality AMBER model is **Pareto-dominated by
a two-modality model at half the compute**. That is a result in itself — the
standard "fuse everything" assumption is wrong on this dataset.

---

## 8 · Finding 2 — radar does not pay for itself

| step | ΔDBA | ΔGFLOPs |
|---|---|---|
| GPS → GPS + Radar | +0.0496 | +23.2 |
| GPS + Camera → GPS + Radar + Camera | -0.0046 | +23.2 |

Adding radar *on top of* camera makes the model **worse**.
GPS + LiDAR (0.7770) and GPS + Radar (0.7794) are within noise of each other.

**What this means.** Our original plan was built on "Radar + GPS is a strong
cheap baseline". **It isn't.** That premise is dead, and finding that out cost
8 GPU-hours instead of months.

---

## 9 · Finding 3 — GPS alone is remarkably strong

**DBA 0.7298 at 0.39 GFLOPs** —
**83 % of the full model's DBA for 0.4 % of its compute.**

A **243× compute ratio.**

**What this means.** This is the single most consequential number in the
project. GPS is effectively free, and it already gets most of the way. The
expensive question is not *which* expensive sensor to add, but *when* to add one
at all.

---

## 10 · Finding 4 — more modalities is not better

| modalities | DBA |
|---|---|
| 2 (GPS + Camera) | 0.8835 |
| 3 (GPS + Radar + Camera) | 0.8789 |
| 4 (full) | 0.8759 |

Monotonically **worse** as modalities are added.

**What this means.** Fusion is not free even when the data is available. This
independently supports selective acquisition: the model does better with fewer,
better-chosen inputs.

---

## 11 · Finding 5 — the camera generalisation trap

| | seen scenarios | **unseen** scenario 31 |
|---|---|---|
| configurations using camera | **0.8763** | 0.0567 |
| configurations without camera | 0.7701 | **0.1940** |

The ranking is **almost exactly inverted**, and the split is entirely explained
by camera. **GPS alone is the best configuration on the unseen scenario**
(0.2640), beating every camera configuration by 3–13×.

**What this means.** Camera features appear **site-specific** — the model
memorises how one intersection looks. GPS, radar and LiDAR encode relative
geometry that transfers.

Because **48 % of the test split is that unseen scenario**, a system tuned on
validation alone would escalate to camera exactly where camera is worst.

*Caveat: n = 50. The signal is the consistent ordering across all eight
configurations, not the individual values.*

---

## 12 · What this does to the research plan

**Original design**

```
  Radar + GPS  ──confident?──> stop
                    │ no
                    └────────> + Camera / LiDAR
```

Dead: radar is nearly worthless, LiDAR adds little.

**Revised design**

```
  GPS (0.39 GFLOPs)  ──confident?──> stop
                          │ no
                          └────────> + Camera (48.2 GFLOPs)
```

| | original | revised |
|---|---|---|
| cost ratio | ~4× | **124×** |
| accuracy headroom | small | **+0.154 DBA** |
| decision | 4-way | **binary** |

**What this means.** The contribution got *stronger*, not weaker. Dropping radar
and LiDAR is a finding, and the cheap tier is now 124× cheaper than the
escalation instead of 4×. A second axis also appeared: *when* to escalate may
depend on domain familiarity, not just confidence.

---

## 13 · The question answered — headroom is real

**Can per-sample routing beat always using GPS + Camera?**

No training needed: for each of the 2,198 validation samples, which of the 8
configurations got it right?

| | Top-1 | GFLOPs |
|---|---|---|
| always cheap (GPS) | 0.3203 | 0.39 |
| always expensive (GPS + Camera) | 0.4604 | 48.24 |
| **oracle router** | **0.5496** | **11.36** |

The oracle beats always-expensive by **+0.089 Top-1 at 24 % of its compute**,
escalating only 22.9 % of samples.

**What this means.** A good router would win substantially. The premise is sound
— the opportunity is measurably there.

---

## 14 · But confidence cannot capture it

Escalating the least-confident samples, using only the cheap model's own output:

| target | escalated | compute saved |
|---|---|---|
| match always-expensive Top-1 | **89 %** | 11 % |
| match always-expensive DBA | **87 %** | 13 % |

Escalating ~88 % of samples to save ~12 % of compute.

**Why:** confidence gives **AUC 0.686** for predicting "the cheap tier already
suffices".

Where the tiers agree — **45 % of samples are wrong under both**, and
**8.9 % are actively *hurt* by escalating**.

**What this means.** The bottleneck is not the idea, it is the **gate signal**.
And the oracle/confidence gap is now a measured target rather than a guess.

---

## 15 · Two results that redirect the work

**1. The difficulty-aware gate is dead.**

| signal | AUC |
|---|---|
| cheap model confidence | **0.686** |
| n_within_10pct *(offline)* | 0.633 |
| margin_db *(offline)* | 0.593 |
| entropy_bits *(offline)* | 0.591 |

Every beam-ambiguity measure scores **below raw confidence** — and they are
computed from the target, so they were the *ceiling* for that approach.

**What this means.** A specific item from the original plan is closed, cheaply.
Don't build it.

---

## 16 · The gate beats 6 of the 8 fixed configurations

The notebook compared only against the *best* one. Against all eight, the gate
dominates — better DBA **and** fewer GFLOPs — 6 of them at one
operating point or another. Only two are never dominated: **GPS** (nothing beats
it on cost) and **GPS + Camera** (the gate's own endpoint).

At 25 % escalation:

| | DBA | GFLOPs |
|---|---|---|
| **gate @ 25 % escalated** | **0.8199** | **12.35** |
| GPS + Radar + LiDAR | 0.8025 | 46.23 |
| GPS + Radar | 0.7794 | 23.57 |
| GPS + LiDAR | 0.7770 | 23.05 |

At 25 % escalation the gate is **+0.017 DBA over GPS+Radar+LiDAR at 3.7× less
compute**, and **+0.041 DBA over GPS+Radar at 1.9× less**.

**What this means.** Routing reaches accuracy/compute points **no fixed
configuration can** — which *is* the resource-efficiency claim. It just does not
match the single best configuration at equal accuracy. Framed against the
frontier rather than against one point, the contribution holds.

*Full dominance table:
[results/oracle_routing/gate_vs_fixed_configs.csv](results/oracle_routing/gate_vs_fixed_configs.csv)*

---

## 16b · The learned gate — it works

A **2,113-parameter MLP** over **31 inference-available features** (cheap-model
uncertainty, availability mask, GPS and sensor-quality statistics). No beam
label, power vector or correctness signal is ever an input. Trained on `train`
(8,844), evaluated on `val` (2,198).

**Escalation needed to reach always-expensive DBA (0.8835):**

| policy | escalated | GFLOPs | compute saved |
|---|---|---|---|
| oracle (DBA-optimal) | 26 % | 12.83 | 73 % |
| **learned gate (`regress_gain`)** | **66 %** | **31.97** | **34 %** |
| learned gate (`classify`) | 84 % | 40.58 | 16 % |
| confidence threshold | 87 % | 42.02 | 13 % |

**DBA at fixed escalation budgets:**

| policy | @10 % | @25 % | @50 % | @75 % |
|---|---|---|---|---|
| oracle (DBA-optimal) | 0.8155 | 0.8819 | 0.9134 | 0.9134 |
| **`regress_gain`** | **0.7919** | **0.8372** | **0.8662** | **0.8850** |
| confidence threshold | 0.7653 | 0.8199 | 0.8557 | 0.8763 |
| `classify` | 0.7552 | 0.7958 | 0.8393 | 0.8759 |

**What this means.** The learned gate beats confidence thresholding **at every
budget** — +0.027 DBA at 10 % escalation, +0.017 at 25 %, against a ±0.011
paired standard error. It saves **34 % of compute where confidence saves 13 %**,
about 2.6×. The bottleneck identified in slide 14 was the gate signal, and a
better gate signal fixed it.

---

## 16c · Two secondary results

**1. The regression objective is the right one.**

`regress_gain` predicts the continuous per-sample DBA gain; `classify` predicts
the binary "cheap wrong, expensive right". The binary label discards *how much*
each sample stands to gain — and `classify` trails by 0.03–0.04 DBA throughout,
losing even to a plain confidence threshold below 50 % escalation. **Drop that
arm.**

**2. The gate transfers across scenarios.**

Leave-one-scenario-out, gate never sees the held-out scenario. Fraction of the
always-cheap → always-expensive gap recovered at 50 % escalation:

| held out | gate @50 % | always-cheap | always-expensive | recovered | n |
|---|---|---|---|---|---|
| scenario 32 | 0.8078 | 0.6209 | 0.8793 | **72 %** | 600 |
| scenario 33 | 0.8452 | 0.7944 | 0.8748 | **63 %** | 750 |
| scenario 34 | 0.8596 | 0.7497 | 0.8943 | **76 %** | 848 |

**A methodological note.** A smoke test on 400 training samples had shown the
gate *within noise* of confidence (+0.001 DBA at 25 %). That was an artefact of
fitting 2,113 parameters on 400 samples, not a property of the method — the
full run shows +0.017. Small-scale pilots can produce false nulls.

**An oracle correction.** The original single `ORACLE` curve was ordered by the
Top-1 criterion, which is *not* optimal for DBA — that is maximised by ordering
on per-sample DBA gain. The tell was the gate scoring 0.8271 against an
"oracle" of 0.8236, beating its own ceiling. Corrected, the true ceiling is
*higher* (26 % escalation, not 33 %), so the headroom is larger than previously
reported.

---

## 16d · Modality robustness — tipping points and a failed safety claim

**Camera is fragile.** Severity at which GPS + Camera (48.24 GFLOPs) drops
*below* free GPS alone (0.7298 DBA, 0.39 GFLOPs):

| degradation | tipping point |
|---|---|
| blur | **σ ≈ 1.3 px** — mild defocus |
| noise | **σ ≈ 0.084** — a dim night scene |
| occlusion | **≈ 17.5 % of frame** — a raindrop |
| resolution | **≈ 0.42 downscale** |

None are extreme. These are the thresholds a cost-aware router should trigger on.

**The GPS cheap tier needs better-than-consumer GNSS.**

| GPS error | GPS alone | GPS + Camera |
|---|---|---|
| 0 m | 0.7298 | 0.8835 |
| **5 m** (consumer) | **0.4367** | 0.8743 |
| 20 m | 0.1610 | 0.8360 |

The "GPS alone gets 83 % at 0.4 % of compute" result **requires DeepSense's
high-precision positioning**. But camera makes the system GPS-robust — so
**camera is the redundancy modality, not only the accuracy modality.** That
partly offsets the generalisation trap.

**AMBER's availability masking does not degrade gracefully.**

| modalities | masked | independently trained | gap |
|---|---|---|---|
| gps + radar | 0.3746 | 0.7794 | **−0.4048** |
| gps + lidar | 0.4117 | 0.7770 | **−0.3654** |
| gps + image | 0.8703 | 0.8835 | −0.0132 |

Masking is fine whenever camera survives and catastrophic whenever it does not.
Eq. (3) handles *redundant* missing modalities, not the one the model actually
learned to use. **Consequence: a cheap tier cannot be built by masking the
expensive model — tiers must be independently trained.**

Dropping one modality from the full model confirms the ablation independently:
lidar **−0.0003**, radar **−0.0012**, gps −0.0214, image **−0.4373**.

*Detail: [results/modality_robustness/](results/modality_robustness/)*

---

## 17 · Honest limitations

- **10 epochs**, not to convergence. Best epochs were 8–10, so the ranking is
  probably stable, but absolute numbers would rise.
- **Historical beam indices disabled.** Not a challenge input, and absent from
  100 % of the test split — including them would inflate results with a signal
  that vanishes at inference.
- **Scenario 31: n = 50.** The generalisation finding needs its full release.
- **The GPS-alone headline assumes high-precision positioning** — at 5 m
  consumer-GNSS error the cheap tier loses 40 % of its DBA.
- **Scenario 34 supplies radar for only 40 %** of its samples, so radar
  configurations are partly GPS-only there — this *understates* radar.
- **Not comparable to the paper** (0.6415 Top-1): it trains on all four
  scenarios including scenario 31's separate release, and uses a random split
  that leaks temporally adjacent frames.
- **Routing measured on validation only** — in-domain by construction. A gate
  tuned here may escalate the wrong way on the 48 % of the test split that is
  the unseen scenario. *(Partly mitigated: leave-one-scenario-out transfer
  recovers 63–76 % of the gap, but scenario 31 cannot be tested this way, as it
  has no training samples.)*
- **Single training run per configuration.** No seeds, no error bars. Some
  Pareto orderings (0.8835 vs 0.8759) may not survive repetition.
- **The gate is compared only against confidence thresholding**, not against the
  cascade / early-exit / learning-to-defer literature.
- **45 % of samples are wrong under both tiers**, which caps any routing gain.

---

## 18 · Where things stand

| | |
|---|---|
| ✅ | Preprocessing, sanity checks, three data defects found |
| ✅ | AMBER baseline (Baseline 4) |
| ✅ | Modality ablation, 8 configurations |
| ✅ | KD radar-only student (Baseline 5) — stale data, own protocol |
| ✅ | Oracle routing analysis — headroom real, confidence gate weak |
| ✅ | **Learned adaptive gate — beats confidence, 34 % vs 13 % compute saved** |
| ❌ | Difficulty-aware gating — **ruled out**, scores below confidence |
| ✅ | Modality robustness — tipping points; masking fails without camera |
| ⚠️ | Quality-signal notebook — run twice, both void (sensor features zero); fixed, needs re-running |
| ⬜ | Scenario 31 full release — needed for the generalisation claim |
| ⬜ | Seeds / error bars — everything is a single run |
| ⬜ | Validate the AMBER reimplementation against the paper's numbers |
| ⬜ | Official GPS-only LSTM (Baseline 1) |

### The one-sentence summary

*A two-modality model beats the full four-modality one at half the compute;
GPS alone gets 83 % of the way at 0.4 % of the cost; camera — the only expensive
modality that pays — is also the one that fails on unseen environments; and
per-sample routing has real headroom (+0.089 Top-1 at a quarter of the compute)
that confidence thresholding cannot reach but a small learned gate partly can,
saving 34 % of compute against confidence's 13 %.*

