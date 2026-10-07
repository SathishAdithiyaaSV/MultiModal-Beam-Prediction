# Adaptive gate (RQ4) — learned sample-adaptive routing

Does a small learned gate, fed only inference-available signals, route better
than a plain confidence threshold? **Yes.** On the full `val` split a gate
regressing the per-sample DBA gain reaches the always-expensive DBA for **34 %
less compute, against a confidence threshold's 13 %**, and transfers to
held-out scenarios.

Produced by [`notebooks/learned_adaptive_gate.ipynb`](../../notebooks/learned_adaptive_gate.ipynb).

| run | status |
|---|---|
| [`full_run/`](full_run/) | **the result.** 8,844 train / 2,198 eval. |
| [`quick_run/`](quick_run/) | smoke test, `QUICK_RUN = True`, 400/300. Superseded; do not cite. |

## Result

Escalation needed to reach always-expensive DBA (0.8835), cheap tier GPS at
0.39 GFLOPs, expensive tier GPS + Camera at 48.24:

| policy | escalated | GFLOPs | compute saved |
|---|---|---|---|
| oracle (DBA-optimal) | 26 % | 12.83 | 73 % |
| **learned gate (`regress_gain`)** | **66 %** | **31.97** | **34 %** |
| learned gate (`classify`) | 84 % | 40.58 | 16 % |
| confidence threshold | 87 % | 42.02 | 13 % |

`regress_gain` beats confidence at every escalation budget (+0.027 DBA at 10 %,
+0.017 at 25 %, +0.011 at 50 %, against a ±0.011 standard error on paired
comparisons over 2,198 samples), and recovers 63–76 % of the cheap→expensive
gap on held-out scenarios.

Two secondary conclusions:

- **The regression objective is the right one.** Predicting the continuous DBA
  gain matches the metric; predicting the binary "cheap wrong, expensive right"
  label discards how much each sample gains. `classify` loses to a plain
  confidence threshold below 50 % escalation and should be dropped.
- **About a third of the oracle headroom is now claimed**, measured on
  escalation-to-match. The rest remains open.

Full discussion, cross-scenario numbers and the file inventory are in
[`full_run/README.md`](full_run/README.md).

## Why `quick_run/` is kept but not cited

It capped the run at 400 training samples for a 2,113-parameter gate over 31
features, and concluded `regress_gain` was within noise of confidence (+0.001
DBA at 25 %). The full run shows that was an artefact of an unfitted gate, not
a property of the method. It is retained as the record of the smoke test that
verified the notebook executes, and because it is what exposed the oracle bug.

## Oracle correction

Both runs executed the notebook before the oracle-ordering fix. A single
`ORACLE (upper bound)` curve was ordered by the Top-1 criterion "cheap wrong
and expensive right", which is optimal for Top-1 but **not** for DBA — that is
maximised by ordering on the per-sample DBA gain. The tell appeared in the
quick run, where `regress_gain` scored 0.8271 against an "oracle" of 0.8236 at
10 % escalation, beating a ceiling.

Corrected, the DBA oracle is **higher** than plotted and matches
always-expensive at 26 % escalation rather than 33 %, so the headroom is larger
than either run reported. `full_run/operating_curves_oracle_corrected.csv`
holds the exact recomputation from the per-sample tables.

[`notebooks/_build_adaptive_nbs.py`](../../notebooks/_build_adaptive_nbs.py)
now emits `ORACLE (Top-1 optimal)` and `ORACLE (DBA optimal)` as separate
curves, each plotted only on the panel for the metric it bounds. Only the
oracle rows were ever affected; the confidence and learned-gate curves in both
runs stand as produced.
