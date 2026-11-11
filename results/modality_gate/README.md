# Cost-aware modality gate

Full validation run from [`notebooks/modality_gate.ipynb`](../../notebooks/modality_gate.ipynb).
The notebook ran with `quick_run = False` on 2,198 validation samples and
trained five gate seeds. It compares GPS-only routing with seven multimodal
experts and includes a camera-blur health-routing experiment.

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
