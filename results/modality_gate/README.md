# Cost-aware modality gate

Full validation run from [`notebooks/modality_gate.ipynb`](../../notebooks/modality_gate.ipynb).
The notebook ran with `quick_run = False` on 2,198 validation samples and
trained a seven-output DBA-gain gate across five temporal folds and five seeds.
It compares GPS-only routing with seven multimodal experts and includes a
camera-blur health-routing experiment. The gate has 2,183 parameters and uses
27 non-constant inference-available features, 13 of which are sensor-health
probes.

## Findings

DBA at fixed estimated compute budgets on clean validation data:

| policy | ≤5 GFLOPs | ≤12 GFLOPs | ≤24 GFLOPs | ≤48 GFLOPs |
|---|---:|---:|---:|---:|
| learned gate, all modalities + probes | 0.7810 | 0.8229 | 0.8476 | 0.8638 |
| **learned gate, no probes** | **0.7911** | **0.8302** | **0.8583** | **0.8713** |
| confidence → GPS + Camera | 0.7608 | 0.8181 | 0.8530 | **0.8840** |
| best fixed expert | 0.7298 | 0.7298 | 0.7794 | 0.8025 |
| oracle (true per-sample gains) | 0.8587 | 0.9134 | 0.9542 | 0.9555 |

The no-probe gate leads the learned-gate variants at every clean budget shown.
At 5–24 GFLOPs it also beats the best fixed expert; at 48 GFLOPs the
confidence-to-GPS+Camera cascade scores higher.

With camera blur σ=2 injected into half the samples, at a 24-GFLOP budget the
probe gate chooses a camera expert for 42.7 % of clean samples and 10.6 % of
faulty samples, with 0.7900 DBA on the faulty subset. The no-probe gate uses a
camera expert at similar rates on clean and faulty samples (15.6 % and 15.4 %)
and scores 0.8018 DBA on the faulty subset. The probes respond to the injected
fault, but this run does not show a net accuracy gain from using them.

The online router matched its offline expert selection on the demo samples and
measured 36.3 ms mean latency, versus 47.0 ms for always running the four-modality
model. This is a notebook-level measurement, not an end-to-end system result.

## Evaluation limits

The gate's five-fold cross-fitting prevents a sample's own target gains from
being used to fit its corresponding prediction. However, the folds, budget
sweep and reported performance all come from the same validation split; this
is not an independent test set. Also, the budget is an estimated GFLOPs cost
and does not include sensor acquisition or read costs. Sensor-health probes
require sensor tensors, so the results address encoder-processing cost rather
than end-to-end sensing cost.

## Files

- `config.json` records the run settings, expert costs, gate features and
  cross-fitting configuration.
- `table_clean.csv` and `table_fault_image_blur_2.csv` contain per-sample
  results for the clean and camera-blur conditions.
- `gate_predictions_clean.csv` contains the clean-run gate predictions.
- `experts.csv`, `curves_*.csv`, `budget_table_*.csv`, `health_routing.csv`,
  `routing_mix.csv` and `router_log.csv` contain aggregate expert, budget,
  health-routing and router outputs.
- `modality_gate.pt` stores the five trained gate models and normalization
  parameters.
- `plots/` contains the accuracy-versus-compute and routing-mix figures.
