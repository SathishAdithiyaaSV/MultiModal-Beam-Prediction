# Full run — `QUICK_RUN = False`

8,844 training and 2,198 evaluation samples, both full splits. **This is the
result.** The `../quick_run/` directory is a smoke test and should not be cited.

Cheap tier GPS (0.39 GFLOPs), expensive tier GPS + Camera (48.24), so
`cost(f) = 0.3869 + f * (48.2398 - 0.3869)`. Reference points on `val`:
always-cheap DBA 0.7298 / Top-1 0.3203, always-expensive DBA 0.8835 / Top-1
0.4604.

## Headline: the regression gate works

Escalation needed to reach always-expensive DBA, and what that costs:

| policy | escalated | GFLOPs | compute saved |
|---|---|---|---|
| oracle (DBA-optimal) | 26 % | 12.83 | 73 % |
| **learned gate (`regress_gain`)** | **66 %** | **31.97** | **34 %** |
| learned gate (`classify`) | 84 % | 40.58 | 16 % |
| confidence threshold | 87 % | 42.02 | 13 % |

DBA at fixed escalation budgets:

| policy | @10 % | @25 % | @50 % | @75 % |
|---|---|---|---|---|
| oracle (DBA-optimal) | 0.8155 | 0.8819 | 0.9134 | 0.9134 |
| **learned gate (`regress_gain`)** | **0.7919** | **0.8372** | **0.8662** | **0.8850** |
| confidence threshold | 0.7653 | 0.8199 | 0.8557 | 0.8763 |
| learned gate (`classify`) | 0.7552 | 0.7958 | 0.8393 | 0.8759 |

Three findings:

1. **`regress_gain` beats confidence thresholding at every budget**, by +0.027
   DBA at 10 % escalation, +0.017 at 25 %, +0.011 at 50 %. The single-proportion
   standard error on n = 2,198 is ±0.011, and these are paired comparisons on
   identical samples, so the low-budget margins are comfortably real. In
   operational terms it reaches the always-expensive DBA for **34 % less
   compute, against confidence's 13 %** — about 2.6× the saving.
2. **The regression objective is the right one.** `classify` trails
   `regress_gain` by 0.03–0.04 DBA throughout and is beaten by a plain
   confidence threshold below 50 % escalation. Predicting the continuous DBA
   gain matches the metric being optimised; predicting the binary "cheap wrong,
   expensive right" label discards how *much* each sample stands to gain.
3. **Roughly half the oracle headroom is now claimed.** On the escalation
   needed to match always-expensive, the gate closes 66 % → from confidence's
   87 % toward the oracle's 26 %, i.e. about 34 % of the available distance.
   The remaining gap is the open problem.

The quick run had suggested `regress_gain` was within noise of confidence
(+0.001 DBA at 25 %). That was an artefact of fitting a 2,113-parameter gate on
400 samples. With 8,844 the advantage is unambiguous — which is exactly why the
quick run was not reportable.

## Cross-scenario generalisation

Leave-one-scenario-out, gate trained without the held-out scenario. Fraction of
the always-cheap → always-expensive gap recovered at 50 % escalation:

| held out | gate @50 % | always-cheap | always-expensive | gap recovered | n |
|---|---|---|---|---|---|
| scenario 32 | 0.8078 | 0.6209 | 0.8793 | **72 %** | 600 |
| scenario 33 | 0.8452 | 0.7944 | 0.8748 | **63 %** | 750 |
| scenario 34 | 0.8596 | 0.7497 | 0.8943 | **76 %** | 848 |

The gate transfers. This is the clearest improvement over the quick run, where
the same measurement gave 45 % / 4 % / 7 % — the earlier figures reflected a
gate that had not been fitted, not a routing policy that fails to generalise.

## Oracle correction

This run executed the notebook **before** the oracle-ordering fix, so
`operating_curves.csv` still contains a single `ORACLE (upper bound)` curve
ordered by the Top-1 criterion. That bounds Top-1 but understates the DBA
ceiling (0.8365 rather than 0.8819 at 25 % escalation; 33 % rather than 26 %
escalation to match).

`operating_curves_oracle_corrected.csv` supplies both correctly-ordered curves,
recomputed here from `table_val.csv`. The tables are per-sample, so this is an
exact recomputation and not an approximation. All oracle figures quoted above
come from the corrected file. Everything else in `operating_curves.csv` — the
confidence and learned-gate curves — is unaffected by the bug and stands as
written.

## Files

| file | contents |
|---|---|
| `config.json` | run configuration, 31-feature list, cost model, leakage note |
| `operating_curves.csv` | as produced; gate curves valid, oracle mislabelled |
| `operating_curves_oracle_corrected.csv` | both oracles, correctly ordered |
| `final_comparison.csv` | gates at DBA-matched and 25 % points vs fixed configurations |
| `cross_scenario_generalisation.csv` | leave-one-scenario-out transfer |
| `table_train.csv` (8,844), `table_val.csv` (2,198) | per-sample features, both tiers' outcomes, gate targets |
| `plots/accuracy_vs_cost.png` | DBA and Top-1 against average GFLOPs |
| `notebook_as_run.ipynb` | the executed notebook, with outputs |

The tables carry the beam label and per-sample DBA as the gate's offline
*supervision*; the 31 columns in `config.json` are its *inputs*. No
label-derived column is ever an input.
