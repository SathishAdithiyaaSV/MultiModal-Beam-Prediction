# Adaptive Modality Routing for mmWave Beam Prediction

### Progress report — 8 October 2026

Predicting the optimal beam out of 64 from multimodal vehicle sensing.
DeepSense6G 2022, scenarios 31–34. Each sample carries 5 camera frames, 5 LiDAR
sweeps, 5 radar frames and 2 GPS positions. Baseline architecture: AMBER,
reimplemented from the paper.

All figures below are measured on the validation split, **n = 2,198**.
*DBA* = distance-based accuracy (Δ = 5), the task's standard tolerant metric.

---

## Summary

Five results, in the order they build on each other.

| # | Result |
|---|---|
| 1 | **A two-modality model beats the full four-modality one at half the compute.** GPS + Camera reaches DBA 0.8835; all four modalities reach 0.8759. |
| 2 | **GPS alone reaches 83 % of full accuracy for 0.4 % of the compute** — a 240× cost gap, and the opportunity the rest of the work exploits. |
| 3 | **The camera is the only modality that pays for itself, and it is also the one that fails on unseen environments** — and it degrades below free GPS under mild blur. |
| 4 | **Perfect per-sample routing would beat the always-expensive model at a quarter of its compute.** The headroom is real and measured. |
| 5 | **A small learned gate captures a third of that headroom** — 34 % of compute saved at matched accuracy, against 13 % for the standard confidence baseline. |

One analysis is still outstanding; it is identified in *Work remaining*.

---

## 1 · The setup

Four configurations matter for everything that follows.

| configuration | DBA | GFLOPs |
|---|---|---|
| GPS + Camera | **0.8835** | 48.24 |
| all four modalities | 0.8759 | 94.10 |
| GPS + Radar + LiDAR | 0.8025 | 46.23 |
| GPS alone | 0.7298 | **0.39** |

Two things stand out immediately.

**The best model is the second-cheapest.** Adding radar and LiDAR to GPS +
Camera costs an extra 46 GFLOPs and makes the model slightly *worse*. The
Pareto frontier ends at two modalities.

**GPS alone is remarkably strong for its price.** 83 % of the full model's
accuracy at 0.4 % of its compute. That 240× gap is the entire opportunity: if we
could identify *which* samples genuinely need the camera, we would pay for it
only on those.

This sets the two tiers used throughout: **cheap = GPS, expensive = GPS +
Camera.**

---

## 2 · Which modalities actually contribute

Two independent measurements agree, which is worth stating because they share no
code path.

**By ablation** — training each subset separately. Radar adds +0.050 DBA for
23 GFLOPs, and *reduces* accuracy when added to camera.

**By masking** — removing one modality from the trained full model:

| modality removed | DBA | change |
|---|---|---|
| LiDAR | 0.8756 | −0.0003 |
| Radar | 0.8747 | −0.0012 |
| GPS | 0.8545 | −0.0214 |
| **Camera** | **0.4386** | **−0.4373** |

**LiDAR and radar are, within noise, contributing nothing.** The model is almost
entirely a camera model with a GPS correction.

---

## 3 · The camera generalisation trap

Scenario 31 never appears in training. Evaluated there, the ranking inverts
almost exactly:

| | seen scenarios | unseen scenario 31 |
|---|---|---|
| camera configurations | 0.8764 | **0.0567** |
| camera-free configurations | 0.7701 | **0.1940** |

**GPS alone becomes the best of all eight configurations on the unseen
scenario.** The modality that buys the most in-domain accuracy generalises the
worst.

This is not a footnote: **48 % of the official test split is scenario 31.**

*The caveat, stated plainly:* scenario 31 gives us only 50 labelled samples. The
consistent ordering across all eight configurations is what carries this
result — not the individual values. Obtaining scenario 31's full release
(7,012 samples) is our highest-priority data task.

---

## 4 · How fragile is the camera?

We swept each modality from intact to severely degraded, asking a specific
question: **at what point does the camera stop being worth its 48 GFLOPs?** The
comparison is against free GPS at DBA 0.7298.

| degradation | camera falls below GPS at | in plain terms |
|---|---|---|
| blur | **σ ≈ 1.3 px** | mild defocus |
| additive noise | **σ ≈ 0.084** | ~8 % noise — a dim night scene |
| occlusion | **≈ 17.5 % of frame** | a dirt patch or raindrop |
| resolution loss | **≈ 0.42 downscale** | roughly half linear resolution |

**None of these are extreme.** A frame blurred at σ = 2 still looks perfectly
normal, and at that point the camera costs 124× more than GPS and performs
*worse*. These thresholds are precisely what a cost-aware router should trigger
on.

### The cheap tier has a dependency we had not accounted for

| GPS error | GPS alone | GPS + Camera |
|---|---|---|
| 0 m | 0.7298 | 0.8835 |
| 1 m | 0.6680 | 0.8827 |
| **5 m** | **0.4367** | 0.8743 |
| 20 m | 0.1610 | 0.8360 |

Consumer GNSS is accurate to 1–5 m. **At 5 m the GPS-only tier loses 40 % of its
DBA.** Our "GPS alone gets 83 % of full accuracy" result holds for DeepSense's
high-precision positioning and must be quoted with that qualification. We would
rather surface this ourselves than have it raised in review.

**The converse is a genuinely useful finding.** GPS + Camera barely moves across
the entire sweep — still 0.8360 at *20 m* of positional error. **The camera is
the redundancy modality as well as the accuracy modality.** That is a second,
independent argument for it, and it partly offsets the generalisation trap.

---

## 5 · A negative result about the architecture

AMBER's availability-mask mechanism is presented as handling missing modalities.
We tested that directly: mask a modality at inference, versus train a model for
that subset properly.

| modalities | masked | properly trained | gap |
|---|---|---|---|
| GPS + Radar | 0.3746 | 0.7794 | **−0.4048** |
| GPS + LiDAR | 0.4117 | 0.7770 | **−0.3654** |
| GPS + Radar + LiDAR | 0.4386 | 0.8025 | **−0.3639** |
| GPS + Camera | 0.8703 | 0.8835 | −0.0132 |
| GPS + Radar + Camera | 0.8756 | 0.8789 | −0.0033 |

**The pattern is exact: masking works whenever the camera survives, and
collapses whenever it does not.** The mechanism handles *redundant* missing
modalities. It does not survive losing the one modality the model actually
learned to use — which is the case that matters for any reliability claim.

**Practical consequence:** a cheap tier cannot be built by masking the expensive
model. Tiers must be independently trained. That is what we do, so our routing
results are unaffected — but it closes off the cheaper alternative, and it is a
meaningful caveat on the published method.

---

## 6 · Is per-sample routing worth pursuing?

We measured the ceiling before building anything. For each validation sample, we
routed perfectly using hindsight:

| | Top-1 | GFLOPs |
|---|---|---|
| always cheap (GPS) | 0.3203 | 0.39 |
| always expensive (GPS + Camera) | 0.4604 | 48.24 |
| **perfect routing** | **0.5496** | **11.36** |

**Perfect routing beats the always-expensive model by +0.089 Top-1 at a quarter
of its compute**, escalating only 23 % of samples. The opportunity is real.

**But the obvious approach fails.** Escalating the least-confident samples
requires **87 % escalation to save 13 % of compute** — nine samples in ten sent
to the expensive model, for an eighth of the compute back.

We also closed a planned line of work cheaply: every beam-ambiguity difficulty
measure scored *below* raw confidence (0.633, 0.593, 0.591 against 0.686). Since
those are computed from the ground truth, they represented the *ceiling* for
that approach. **Difficulty-based routing is ruled out and we are not building
it.**

The bottleneck was the routing signal, not the idea.

---

## 7 · The learned gate

A **2,113-parameter network** over **31 features available at inference time** —
the cheap model's uncertainty, the sensor availability mask, GPS statistics, and
sensor-quality measures. The ground-truth beam is never an input; it is used only
to construct the training target.

Two formulations were compared: predicting *whether* escalation helps, and
predicting *how much* it helps.

**Escalation required to match always-expensive accuracy:**

| policy | escalated | GFLOPs | compute saved |
|---|---|---|---|
| perfect routing (ceiling) | 26 % | 12.83 | 73 % |
| **learned gate — predict gain** | **66 %** | **31.97** | **34 %** |
| learned gate — predict yes/no | 84 % | 40.58 | 16 % |
| confidence threshold | 87 % | 42.02 | 13 % |

**Accuracy at fixed compute budgets:**

| policy | 10 % escalated | 25 % | 50 % |
|---|---|---|---|
| perfect routing | 0.8155 | 0.8819 | 0.9134 |
| **learned gate — predict gain** | **0.7919** | **0.8372** | **0.8662** |
| confidence threshold | 0.7653 | 0.8199 | 0.8557 |
| learned gate — predict yes/no | 0.7552 | 0.7958 | 0.8393 |

Three points:

1. **The gate beats confidence thresholding at every budget** — +0.027 DBA at
   10 % escalation, +0.017 at 25 %, against a ±0.011 paired standard error. It
   saves **34 % of compute where confidence saves 13 %**, roughly 2.6×.
2. **Predicting the magnitude of the gain clearly beats predicting a yes/no
   label**, by 0.03–0.04 DBA throughout. The binary formulation discards how
   *much* each sample stands to gain, and loses even to plain confidence below
   50 % escalation. We are dropping it.
3. **The gate transfers.** Trained without a given scenario and tested on it, it
   recovers **72 % / 63 % / 76 %** of the cheap-to-expensive gap on scenarios
   32 / 33 / 34.

### A caution on feature richness

Measured *within* known scenarios, richer features help — that is how the gate
wins. Measured *across* held-out scenarios, plain confidence alone (AUC 0.6952)
beats all eighteen richer features (0.6527).

Both are true, and together they say something sharper than either alone: **the
extra features encode environment-specific structure.** This is the camera
generalisation trap reappearing — this time in the router's features rather than
the model's modalities.

---

## 8 · What this adds up to

**The original plan did not survive contact with the data, and what replaced it
is better.** We had intended a Radar + GPS cheap tier escalating on predicted
difficulty. Both halves failed on evidence: radar does not pay for itself, and
difficulty measures score below plain confidence. The replacement — a GPS cheap
tier with a learned gain-regression gate — outperforms both.

**We have a quantified efficiency result:** 34 % compute saved at matched
accuracy against 13 % for the standard baseline, with demonstrated
cross-scenario transfer.

**But our strongest contribution may not be the routing work.** The evidence
points at a more uncomfortable and more interesting claim: *most modalities in
multimodal beam prediction do not pay for themselves, and the one that does
pay generalises worst, degrades under mild sensor noise, and is the single point
of failure the architecture's robustness mechanism cannot absorb.* That cuts
against the field's push toward adding modalities.

**Proposed framing:** lead with the modality-economics and generalisation
result; present the routing work as its constructive answer.

---

## 9 · Limitations we would state in any write-up

- **Single training run per configuration** — no seeds, no error bars. Some
  close orderings (0.8835 vs 0.8759) may not survive repetition.
- **Scenario 31 provides 50 labelled samples.** The generalisation claim rests
  on the consistent ordering across configurations, not on individual values.
- **The GPS-alone headline assumes high-precision positioning**, as shown above.
- **Our Top-1 is 0.4604; the AMBER paper reports 0.6415.** We believe this is
  our stricter protocol — block-based splits that prevent temporally adjacent
  frames leaking between train and test, no beam-history input, and no
  scenario-31 training data. This needs to be demonstrated rather than asserted.
- **45 % of samples are wrong under both tiers**, which caps any routing gain.
- **Routing is measured in-domain.** Transfer is tested by holding out
  scenarios, but scenario 31 cannot be tested that way, having no training data.
- **Models trained 10 epochs**, not to convergence. Best epochs were 8–10, so
  rankings are likely stable; absolute numbers would rise.

---

## 10 · Work remaining

**Outstanding analysis.** One experiment is incomplete: separating routing
signals by *what they cost to obtain*. Some signals are free before any expensive
sensor is read; others require reading the camera first. Which set we are allowed
to use depends on whether the dominant cost is *processing* the sensor data or
*acquiring* it — and the two give different answers. The first two attempts at
this analysis were invalid due to a defect in feature extraction, now fixed and
guarded. It needs one ~25-minute re-run.

**Before any submission, in priority order:**

1. **Obtain scenario 31's full release** (7,012 samples). Our most interesting
   finding currently rests on 50 samples. This is the single highest-value
   action available.
2. **Add seeds and error bars.** Mechanical, cheap, and removes the most common
   reviewer objection. Better that we find a fragile ordering than a referee.
3. **Resolve the absolute-accuracy question** against the AMBER paper. If our
   protocol explains the gap, that is defensible and interesting; if the
   reimplementation underperforms, every comparison rests on a weak base. This
   is the highest-priority unknown.
4. **Benchmark the gate against the adaptive-inference literature** — cascades,
   early-exit methods, learning-to-defer — rather than only against a confidence
   threshold. Our novelty is the multimodal sensor-cost framing, and we should
   say so explicitly.

---

## 11 · Decisions we would like to take

1. **Framing** — do we lead with modality economics and generalisation, or with
   adaptive routing?
2. **Scenario 31** — is the full release obtainable, and who takes it on?
3. **Cost model** — do we treat *processing* or *sensing* as the dominant cost?
   This determines which routing signals are admissible, and therefore what the
   headline efficiency number is.
4. **Target venue** — this shapes how much of items 1–4 above is required.
