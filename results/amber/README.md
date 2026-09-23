# AMBER baseline — run results

Baseline 4 of the project: AMBER (Wen et al.) on DeepSense6G scenarios 31–34.
One directory per training run, named for the scenarios it trained on.

Checkpoints (`*.pt`) are **git-ignored** — `best.pt` is ~189 MB and `last.pt`
~567 MB, both over GitHub's 100 MB per-file limit. Everything needed to read a
result (metrics, history, figures, predictions) is tracked.

| | run1_scenarios_32_33 | run2_scenarios_32_33_34 |
|---|---|---|
| trained on | scenarios 32, 33 | scenarios 32, 33, **34** |
| train / val samples | 5,544 / 1,350 | 8,844 / 2,198 |
| epochs | 20 of 20 | **17 of 20** (see `run_info.json`) |
| best epoch | 13 | 12 |
| val Top-1 | 0.3852 | **0.4345** |
| val Top-3 | 0.7230 | **0.7862** |
| val DBA | 0.8338 | **0.8612** |
| train time | 2.46 h | 3.15 h |

**run2 supersedes run1.** Adding scenario 34 — which also introduced the
crossroad geometry the model had never seen — gained +6.8 pp Top-1, +9.0 pp
Top-3 and +4.7 pp DBA. run1 is kept because it is the controlled
2-scenario-versus-3-scenario comparison, which is a reportable result in itself.

For reference, the AMBER paper reports Top-1 0.6415 / Top-3 0.8907 / DBA 0.9294.
That is **not** a like-for-like comparison: the paper trains on all four
scenarios including scenario 31's separate 7,012-frame release, uses a random
80/20 split (which leaks temporally adjacent frames), and keeps historical beam
indices enabled.

## run2 contents

```
config.json                             WRONG epochs value -- read run_info.json
run_info.json                           corrected provenance + known results
history.csv                             per-epoch loss / val metrics
metrics.json                            val + adaptation + test, per scenario
metrics_val.json, metrics_adaptation.json
predictions_{val,adaptation,test}.csv   ranked top-5 beams + confidences
submission_test.csv                     top-1 only, 1-indexed
preliminary_masking_ablation_val.csv    see the warning below
figures/fig1..fig4*.png
notebook_as_run.ipynb                   the executed Kaggle notebook
best.pt, last.pt                        git-ignored
```

## Two things to know before quoting these numbers

**`config.json` misstates the protocol.** It records `epochs: 15`; the run
actually used a 20-epoch cosine schedule and completed 17. The learning-rate
trace proves it. `run_info.json` has the full explanation. Immaterial to the
result — the best epoch was 12 and 13–17 did not improve on it — but do not
quote `config.json`.

**`preliminary_masking_ablation_val.csv` is NOT the modality-ablation
experiment.** It masks one modality at a time on a *single model trained with
all of them*, which measures how much that model leans on each input — not what
a model trained without it would achieve. The real experiment trains each
configuration independently and lives in
`notebooks/modality_ablation_part{1,2}.ipynb`.

It is kept because its signal is strong and worth carrying into that
experiment's design:

| removed | ΔTop-1 |
|---|---|
| **Camera** | **−0.0965** |
| LiDAR | −0.0159 |
| GPS | −0.0155 |
| Radar | −0.0132 |
| BeamIdx | −0.0105 |

Camera dominates by roughly 6×. If the full experiment confirms it, Camera is
the expensive modality worth selectively acquiring, which is the central
question for the adaptive-routing direction.

## Known result: scenario 31 collapses

On the adaptation split, scenario 31 scores Top-1 0.04 / DBA 0.156 against
0.85–0.91 everywhere else. **Expected, not a bug:** scenario 31 has zero
training samples — it appears only in the adaptation and test splits — and is
the only straight-road geometry. Scenarios 32 and 33 on the *same* split, with
the *same* absent beam history, score 0.56 and 0.44, which rules out the split,
the preprocessing and beam-history absence as causes.

This matters for the test predictions: **48 % of the test split (300/625) is
scenario 31**, so roughly half of it is predicted near-randomly. The model's own
confidence flags this — ~0.04 on scenario 31 against 0.12–0.20 on scenarios
33/34 — which is itself a useful signal for the uncertainty-gating work.

Fixing it needs data, not code: the standalone scenario-31 release (7,012
frames) that the paper used.
