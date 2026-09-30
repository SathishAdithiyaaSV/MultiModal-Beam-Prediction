# Part 1 provenance

`results_part1.csv` and `scenario_results_part1.csv` were **parsed from the
printed outputs of `notebook_as_run.ipynb`**, not copied from the run's own
output files. The authoritative artefacts are the four
`outputs/modality_ablation/per_config/*.json` files the notebook wrote in its
Kaggle session; they additionally record effective modality availability,
the exact protocol, and full per-split metrics.

**To make this directory authoritative:** download those four JSONs from the
Kaggle session (Save Version, then the Output tab) into `per_config/` here.
`modality_ablation_report.ipynb` consumes them directly and will then
regenerate these CSVs from source rather than from stdout.

The numbers themselves are not in doubt -- they are the notebook's own
printed metrics -- but a parsed table is one transformation further from the
run than a serialised record, so it is labelled as such.
