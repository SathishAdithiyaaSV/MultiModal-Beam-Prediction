# Adaptive gate (RQ4) — learned sample-adaptive routing

Does a small learned gate, fed only inference-available signals, route better
than a plain confidence threshold? The oracle-routing study set the stakes:
perfect routing reaches the always-expensive DBA at a fraction of its cost,
and a confidence threshold claims almost none of that headroom.

Produced by [`notebooks/learned_adaptive_gate.ipynb`](../../notebooks/learned_adaptive_gate.ipynb).

| run | status |
|---|---|
| [`quick_run/`](quick_run/) | **smoke test only — `QUICK_RUN = True`.** Not a result. |
| full run | **not yet performed.** This is the blocking next step. |

## Why `quick_run/` cannot answer the question

`QUICK_RUN = True` caps the run at **400 training and 300 evaluation samples**.
The gate is a 2,113-parameter MLP over **31 features**. Fitting that on 400
samples is roughly 68 parameters per sample, so the learned gates are not being
given a fair test, and every evaluation number carries a binomial standard
error near ±0.03 DBA on n = 300. Differences below ~0.06 are not readable.

The run's purpose was to prove the notebook executes end to end. It does, with
no errors. That is the whole of what it establishes.

## What it nevertheless showed

Cheap tier GPS (0.39 GFLOPs), expensive tier GPS + Camera (48.24), so
`cost(f) = 0.3869 + f * (48.2398 - 0.3869)` for escalated fraction `f`.
Always-expensive reference: DBA 0.8936, Top-1 0.4700.

| curve | esc. to match | DBA@10% | DBA@25% | DBA@50% |
|---|---|---|---|---|
| oracle (DBA-optimal) | 21 % | 0.8458 | 0.9033 | 0.9253 |
| learned gate (`regress_gain`) | 99 % | **0.8271** | **0.8442** | **0.8682** |
| confidence threshold | 89 % | 0.8044 | 0.8429 | 0.8636 |
| learned gate (`classify`) | 96 % | 0.7893 | 0.8098 | 0.8358 |

Three readings, all provisional:

1. **The regression objective beats the classification one**, by 0.03–0.04 DBA
   throughout. This was the a-priori expectation and it held: predicting the
   continuous DBA gain matches the metric being optimised, whereas predicting
   the binary "cheap wrong, expensive right" label discards how *much* a
   sample stands to gain. `classify` is beaten by a plain confidence threshold
   and should be dropped.
2. **`regress_gain` edges out confidence only in the low-escalation regime**
   (+0.023 at 10 %, +0.001 at 25 %, +0.005 at 50 %) and loses above ~75 %.
   The low-escalation regime is the one that matters — it is where compute is
   actually saved — but at this sample size only the 10 % figure is near the
   noise floor, not clearly above it.
3. **Most of the oracle headroom is unclaimed.** The oracle matches
   always-expensive DBA at 21 % escalation; the best realisable gate needs
   99 %. This is the open problem, unchanged by this run.

Held-out-scenario transfer (`cross_scenario_generalisation.csv`) is weak: at
50 % escalation the gate recovers 45 % of the cheap→expensive gap on scenario
32 but only 4 % on 33 and 7 % on 34. Consistent with a gate fitted on 400
samples; must be re-read after the full run.

## Correction applied after this run

The notebook labelled a single curve `ORACLE (upper bound)` and ordered it by
the **Top-1** criterion "cheap wrong and expensive right". That is the optimal
order for Top-1, but **not for DBA**, which is maximised by ordering on the
per-sample DBA gain. The consequence was visible in these outputs: at 10 %
escalation `regress_gain` scored 0.8271 against an "oracle" of 0.8236 — a gate
appearing to beat its own ceiling, which is impossible and was the tell.

The true DBA-optimal oracle is **higher** than was plotted (0.9033 vs 0.8620 at
25 % escalation) and reaches the always-expensive DBA at **21 %, not 30 %**.
The headroom is larger than the quick run reported, so the finding that the gap
is mostly unclaimed is strengthened, not weakened.

`notebooks/_build_adaptive_nbs.py` now emits two separately-ordered curves,
`ORACLE (Top-1 optimal)` and `ORACLE (DBA optimal)`, each plotted only on the
panel for the metric it bounds. The CSVs in `quick_run/` predate the fix and
retain the mislabelled curve; `quick_run/README.md` records this.

## Files

| file | contents |
|---|---|
| `config.json` | run configuration, feature list, cost model, leakage note |
| `operating_curves.csv` | DBA / Top-1 / GFLOPs at 101 escalation fractions per gate |
| `final_comparison.csv` | gates at DBA-matched and 25 % operating points, against the fixed configurations |
| `cross_scenario_generalisation.csv` | leave-one-scenario-out transfer |
| `table_train_quick.csv`, `table_val_quick.csv` | cached per-sample features, both tiers' outcomes, and the gate targets |
| `plots/accuracy_vs_cost.png` | DBA and Top-1 against average GFLOPs |
| `notebook_as_run.ipynb` | the executed notebook, with outputs |

`table_*_quick.csv` hold the beam label and per-sample DBA. Those are the gate's
offline *supervision*; the 31 columns listed in `config.json` are its *inputs*.
No label-derived column is ever an input — see the leakage note in `config.json`.

## Next step

Re-run with `QUICK_RUN = False` (full `train` / `val` splits). Nothing else in
this experiment can be concluded until then.
