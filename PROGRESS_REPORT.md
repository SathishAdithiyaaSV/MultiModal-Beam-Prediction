# Adaptive Modality Routing for mmWave Beam Prediction

### Progress report — 11 November 2026

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
| 5 | **A random split on this dataset leaks 100 % of validation samples**, which explains the gap to published accuracy figures. |
| 6 | **A small learned gate beats every standard routing rule** — +0.027 DBA at 10 % escalation, +0.019 at 25 %, bootstrap CIs excluding zero, stable across twenty seeds. |
| 7 | **A full-expert gate improves on fixed experts at tight compute budgets**, but sensor probes do not improve clean DBA; this is a same-validation-split result, not an independent test. |

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

| policy | escalated | compute saved |
|---|---|---|
| perfect routing (ceiling) | 26 % | 73 % |
| **learned gate — predict gain** | **77 % ± 13 %** | **23 % ± 13 %** |
| confidence threshold | 87 % | 13 % |

*Measured over twenty seeds. A single seed gave 66 %, but this statistic is
unstable — see below.*

**Accuracy at fixed compute budgets:**

| policy | 10 % escalated | 25 % | 50 % |
|---|---|---|---|
| perfect routing | 0.8155 | 0.8819 | 0.9134 |
| **learned gate — predict gain** | **0.7919** | **0.8372** | **0.8662** |
| confidence threshold | 0.7653 | 0.8199 | 0.8557 |
| learned gate — predict yes/no | 0.7552 | 0.7958 | 0.8393 |

**Accuracy at fixed budgets is where the result is solid.** Twenty seeds and a
2,000-resample paired bootstrap:

| budget | gate (20 seeds) | confidence | gap | 95 % CI | p(gate > conf) |
|---|---|---|---|---|---|
| 10 % | 0.7924 ± 0.0017 | 0.7653 | **+0.0274** | [+0.0220, +0.0328] | 100 % |
| 25 % | 0.8354 ± 0.0037 | 0.8199 | **+0.0185** | [+0.0127, +0.0242] | 100 % |
| 50 % | 0.8673 ± 0.0029 | 0.8557 | **+0.0151** | [+0.0105, +0.0200] | 100 % |

Three points:

1. **The gate beats confidence thresholding at every budget**, with seed
   standard deviation under 0.004 and every bootstrap interval excluding zero
   by a wide margin. This is the claim we stand behind.
2. **Predicting the magnitude of the gain clearly beats predicting a yes/no
   label**, by 0.03–0.04 DBA throughout. The binary formulation discards how
   *much* each sample stands to gain, and loses even to plain confidence below
   50 % escalation. We are dropping it.
3. **The gate transfers.** Trained without a given scenario and tested on it, it
   recovers **72 % / 63 % / 76 %** of the cheap-to-expensive gap on scenarios
   32 / 33 / 34.

### A correction we are making to ourselves

An earlier version of this report led with "34 % of compute saved against
confidence's 13 %". That came from a **single seed**. Across twenty, escalation
to match is **76.8 % ± 12.6 %** — and three seeds never match at all. The
honest saving is **23 % ± 13 %**.

The cause is statistical rather than a defect. "Escalation needed to match" is
a threshold crossing on a curve that is nearly flat near the target, so a DBA
shift of 0.002 moves it by tens of percent. It is too brittle to carry a
headline. The fixed-budget numbers measure the same advantage stably, and are
what we now lead with.

**The earlier two-tier gate turns out to use 12 features, not 31** — nineteen
are constant and contribute nothing. That strengthens rather than weakens its
finding: twelve cheap signals beat the entire classical rejection-score family.
The newer eight-expert experiment separately uses 13 sensor-health probes;
these do not improve clean-validation DBA over the no-probe variant.

### It beats the whole uncertainty family, not just confidence

The deferral literature offers a standard set of rejection scores. All of them
land in the same place:

| routing score | escalation to match |
|---|---|
| Chow's rule (max-softmax) | 87 % |
| margin, top1−top2 | 93 % |
| margin, top1−top5 | 88 % |
| predictive entropy | 94 % |
| 1 − top-5 mass | 88 % |
| Gini impurity | 89 % |
| **learned gate** | **66 %** |

They cluster tightly, because they are largely monotone functions of the same
uncertainty. **The gate's advantage does not come from better-calibrated
uncertainty — it comes from using information these scores cannot see.**

### A caution on feature richness

Measured *within* known scenarios, richer features help — that is how the gate
wins. Measured *across* held-out scenarios, plain confidence alone (AUC 0.6952)
beats all eighteen richer features (0.6527).

Both are true, and together they say something sharper than either alone: **the
extra features encode environment-specific structure.** This is the camera
generalisation trap reappearing — this time in the router's features rather than
the model's modalities.

---

## 7b · The newer gate chooses among all eight experts

The executed
[`notebooks/modality_gate.ipynb`](notebooks/modality_gate.ipynb) expands routing
from the earlier GPS/GPS+Camera cascade to GPS plus all seven multimodal
experts. It ran 2,198 validation samples, five temporal folds and five seeds,
with a 2,183-parameter regressor over 27 non-constant features (13 sensor-health
probes).

DBA at estimated budgets of 5 / 12 / 24 / 48 GFLOPs:

| policy | 5 | 12 | 24 | 48 |
|---|---:|---:|---:|---:|
| learned gate, all modalities + probes | 0.7810 | 0.8229 | 0.8476 | 0.8638 |
| **learned gate, no probes** | **0.7911** | **0.8302** | **0.8583** | **0.8713** |
| confidence → GPS + Camera | 0.7608 | 0.8181 | 0.8530 | **0.8840** |
| best fixed expert | 0.7298 | 0.7298 | 0.7794 | 0.8025 |

At 5–24 GFLOPs, learned expert selection beats the best fixed expert under
the same estimated budget. At 48 GFLOPs, the confidence cascade does better.
The no-probe gate leads the learned variants at every clean budget shown.

With camera blur σ=2 injected in half of the samples, the probe gate sends
42.7 % of clean samples but only 10.6 % of faulty samples to a camera expert.
This confirms a routing response to this synthetic fault, not a net accuracy
benefit: on the faulty subset, DBA is 0.7900 with probes and 0.8018 without.
The estimated GFLOPs budget excludes sensor acquisition/read costs even though
sensor probes require sensor tensors.

**Evaluation caveat:** the gate's predictions are cross-fitted, but the
cross-validation folds, budget sweep and reported metrics all come from the
same validation split. Independent temporal-block or scenario evaluation is
needed before treating these numbers as generalisation performance. The online
router's 36.3 ms vs 47.0 ms measurement is notebook-level, not an end-to-end
sensing-cost measurement.

Full output inventory and limitations:
[`results/modality_gate/README.md`](results/modality_gate/README.md).

---

## 8 · What this adds up to

**The original plan did not survive contact with the data, and what replaced it
is better.** We had intended a Radar + GPS cheap tier escalating on predicted
difficulty. Both halves failed on evidence: radar does not pay for itself, and
difficulty measures score below plain confidence. The replacement — a GPS cheap
tier with a learned gain-regression gate — outperforms both.

**We have a quantified efficiency result:** the earlier two-tier gate beats confidence
thresholding and all six standard rejection scores at every escalation
budget, with bootstrap intervals excluding zero and seed variation under 0.004,
plus demonstrated cross-scenario transfer. The newer full-expert gate adds
promising fixed-expert improvements at tight estimated budgets, but needs
independent evaluation and a sensing-aware cost model.

**But our strongest contribution may not be the routing work.** The evidence
points at a more uncomfortable and more interesting claim: *most modalities in
multimodal beam prediction do not pay for themselves, and the one that does
pay generalises worst, degrades under mild sensor noise, and is the single point
of failure the architecture's robustness mechanism cannot absorb.* That cuts
against the field's push toward adding modalities.

**Proposed framing:** lead with the modality-economics and generalisation
result; present the routing work as its constructive answer.

---

## 10 · Why our accuracy is lower than the published figure

We report Top-1 **0.4604**; the AMBER paper reports **0.6415**. We can now
explain that with measurements rather than assertion — and it turns out to be a
contribution in its own right.

**The mechanism.** Each sample is five consecutive observations. Two samples
whose targets differ by fewer than five frames **share actual input frames** —
the same image, LiDAR sweep and radar cube on both sides of the split.

**What a random split does**, over 11,042 samples and 20 draws:

| | block split (ours) | random split |
|---|---|---|
| val samples sharing input frames with train | **13.6 %** | **100.0 %** |
| val samples within 1 frame of train | 3.4 % | 95.3 % |
| median frames to nearest train sample | **16** | **1** |

**Under a random split, every validation sample shares input frames with
training data.** And the label barely moves — a sample one frame away carries
the same beam **39.4 %** of the time, against 1.6 % chance, a 25× lift.

So a randomly-split evaluation on DeepSense6G is not chiefly measuring
generalisation to new moments; it is substantially measuring recall of frames
already seen. Our numbers are lower because they are measured on genuinely
held-out time. **Random splitting is common on this dataset, so this is worth
reporting as a methodological finding.**

**Where our own split falls short.** 13.6 % of our validation samples still
overlap training windows — block boundaries create adjacent pairs. Far better
than 100 %, but not zero, and we should not claim a clean split while one in
seven overlaps. A **guard band** dropping samples within 4 frames of a boundary
would take it to exactly 0 % for about 8 % of the data. Worth doing before
submission, reported both ways.

*Detail: [results/split_leakage/](results/split_leakage/)*

---

## 9 · Limitations we would state in any write-up

- **Single training run per configuration** — no seeds, no error bars. Some
  close orderings (0.8835 vs 0.8759) may not survive repetition. The *gate* now
  has twenty seeds and bootstrap intervals; the *backbone* does not.
- **Scenario 31 provides 50 labelled samples.** The generalisation claim rests
  on the consistent ordering across configurations, not on individual values.
- **The GPS-alone headline assumes high-precision positioning**, as shown above.
- **Our Top-1 is 0.4604 against the paper's 0.6415** — now explained and
  measured, not asserted (section 10). Our own split still leaks 13.6 %, fixable
  with a guard band.
- **45 % of samples are wrong under both tiers**, which caps any routing gain.
- **Routing is measured in-domain.** Transfer is tested by holding out
  scenarios, but scenario 31 cannot be tested that way, having no training data.
- **The eight-expert router's reported budgets omit sensor acquisition/read
  costs**, and its gate selection and budget sweep use the same validation split.
- **Models trained 10 epochs**, not to convergence. Best epochs were 8–10, so
  rankings are likely stable; absolute numbers would rise.

---

## 11 · Work remaining

**Outstanding analysis.** The dedicated `modality_quality_signal` feature-group
experiment has not yet been rerun with the extraction fix. The newer
eight-expert gate does include sensor-health probes, but it does not replace
that controlled comparison: the no-probe variant performs better on clean
validation DBA, and the reported compute budget does not charge for sensor
acquisition. Signals that require reading camera/LiDAR/radar are admissible
under a processing-cost model, not necessarily under a sensing-cost model.

**Before any submission, in priority order:**

1. **Obtain scenario 31's full release** (7,012 samples). Our most interesting
   finding currently rests on 50 samples. This is the single highest-value
   action available.
2. **Add backbone seeds and error bars.** Some close ablation orderings may
   not survive repetition; the gate seeds are already complete for the
   earlier two-tier experiment.
3. **Reduce residual split leakage.** The temporal analysis explains the
   published accuracy gap, but 13.6 % of validation samples still share input
   frames with training samples. Evaluate with a guard band and report both.
4. **Independently evaluate the eight-expert router.** Its five-fold
   cross-fitted predictions and budget sweep use the same validation split.
   Evaluate on held-out temporal blocks or scenarios, and include sensor
   acquisition/read costs if claiming end-to-end savings.
5. **Extend the baseline comparison.** The earlier two-tier gate is measured
   against six standard rejection scores and beats all of them. Still to add:
   a trained learning-to-defer surrogate and architectural early-exit methods
   (BranchyNet, MSDNet), which need their own training runs.

---

## 12 · Decisions we would like to take

1. **Framing** — do we lead with modality economics and generalisation, or with
   adaptive routing?
2. **Scenario 31** — is the full release obtainable, and who takes it on?
3. **Cost model** — do we treat *processing* or *sensing* as the dominant cost?
   This determines which routing signals are admissible, and therefore what the
   headline efficiency number is.
4. **Target venue** — this shapes how much of items 1–4 above is required.
