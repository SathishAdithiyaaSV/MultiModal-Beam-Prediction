# LAM-MSAC-style MoE baseline — DeepSense6G beam prediction

This directory records the completed 10-epoch run of the paper-inspired
LAM-MSAC-style sparse mixture-of-experts model implemented in
[`notebooks/lam_msac_beam_prediction.ipynb`](../../notebooks/lam_msac_beam_prediction.ipynb).
The executed Kaggle notebook is preserved as `notebook_as_run.ipynb`.

The model is an adaptation of **“Large AI Model for Multimodal Integrated
Sensing and Communication”**, IEEE Network, 2026,
[doi:10.1109/MNET.2026.3661589](https://doi.org/10.1109/MNET.2026.3661589).
It uses the DeepSense6G inputs available in this repository; it is not an exact
reproduction of the paper's raw point-cloud and complex-RF encoders.

## Result at a glance

On the 2,198-sample validation split (scenarios 32, 33, and 34), this run
achieved **35.12% Top-1, 66.33% Top-3, and 0.7868 DBA**. It trails both of the
project's independently trained AMBER modality baselines below.

Top-1/Top-3/Top-5 are the fractions where the exact target beam appears in the
top 1/3/5 predicted beams. DBA is the distance-based accuracy used by this
project: it also gives partial credit when a predicted beam is near the target
(within a beam-index distance scale of 5), averaging over the top three
predictions. Thus DBA can be higher than Top-1 even when the exact beam is not
ranked first.

| Model | Input configuration | Top-1 | Top-3 | Top-5 | DBA | Parameters |
|---|---|---:|---:|---:|---:|---:|
| LAM-MSAC-style MoE | Image + LiDAR BEV + radar maps + GPS | 0.3512 | 0.6633 | 0.8153 | 0.7868 | 50.36M |
| AMBER ablation | GPS + Radar + Camera + LiDAR | 0.4399 | 0.8103 | 0.9286 | 0.8759 | 49.39M |
| AMBER ablation | GPS + Camera | **0.4604** | **0.8139** | **0.9295** | **0.8835** | 26.65M |

All three validation results have `n=2,198` and use the same project validation
split. Against the full-modality AMBER ablation, this MoE run is lower by **8.87
percentage points Top-1** and **0.0891 DBA**. Against GPS + Camera, it is lower
by **10.92 points Top-1** and **0.0967 DBA**. The simpler GPS + Camera AMBER
model also has roughly half the parameter count.

These are observed results for these trained checkpoints, not evidence that
MoE is inherently worse than AMBER. Training and model details differ (see
“Comparison caveats”).

## What the result means

- The paper-inspired architecture successfully trains and routes each example
  through one of four Transformer-style cross-modal experts (`top-k=1`).
- Validation metrics improved overall during the 10 epochs, with the best
  recorded validation Top-1 at epoch 10. The run does not establish that the
  model had converged; additional training could change the ranking.
- Under this run's settings, the MoE model did **not** outperform the project's
  AMBER baselines. Its parameter count is close to the full AMBER model, and
  no FLOP, latency, or energy measurement was recorded, so this run does not
  show an efficiency advantage.
- The secondary adaptation result was Top-1 `0.22`, Top-3 `0.40`, DBA `0.4967`
  on 100 labelled examples. This split combines scenarios 31, 32, and 33; only
  50 rows are from unseen scenario 31. It is a small secondary check, not a
  reliable scenario-31 estimate. The LAM run did not save per-scenario
  adaptation metrics.

  For context, the separately trained full AMBER run scored Top-1 `0.27`,
  Top-3 `0.45`, DBA `0.5267` on the same 100-example adaptation split. The MoE
  run is lower on these aggregate metrics too, but this small split and the
  different training protocols do not establish a robust ranking. In particular,
  the aggregate must not be presented as a scenario-31-only score.

## Run configuration and protocol

- Seed `2022`; 10 epochs; batch size `8`.
- AdamW, learning rate `2e-4`, weight decay `1e-6`.
- Four experts; top-1 routing; load-balancing weight `0.01`.
- Inputs: five RGB, LiDAR BEV, and processed radar observations, plus two GPS
  positions. Historical beam indices are excluded.
- Encoders: ResNet-50 for images, ResNet-18 for LiDAR and radar, and an MLP for
  GPS. ResNet weights are randomly initialized (`pretrained=False`); convolution
  and linear layers use Xavier initialization, normalization layers use unit
  scales/zero biases, and temporal/query embeddings use normal initialization.
- Classification loss: cross-entropy plus the router balance penalty.
- The run used an NVIDIA Tesla T4. Each epoch took about 1,008 seconds (about
  16.8 minutes), or approximately 2 hours 48 minutes for 10 epochs.
- Train: 8,844 labelled samples from scenarios 32–34. Validation: 2,198
  labelled samples from scenarios 32–34. Adaptation: 100 labelled samples.

## Comparison caveats

The validation rows and split match the AMBER modality-ablation results, but
this is not a controlled architecture-only comparison:

1. The AMBER ablation used batch size `16` and learning rate `1e-4`; this MoE
   run used batch size `8` and learning rate `2e-4`.
2. AMBER's image backbone uses pretrained ResNet-34 weights; this MoE run uses
   a randomly initialized ResNet-50.
3. AMBER uses its soft-label focal objective with contrastive and regularizer
   terms. The MoE notebook uses ordinary cross-entropy plus its routing balance
   penalty.
4. The paper's RGB, scalar-power, point-cloud, and complex mmWave encoders
   cannot be reproduced from the project representation as-is. This model
   substitutes LiDAR BEV and processed real-valued radar maps; power vectors
   supply labels, not model inputs.
5. Only one seed was run for each of the compared modality-ablation models.
   There are no confidence intervals or multi-seed estimates here.
6. Parameter count is not a compute or energy measurement. No comparable
   FLOPs, end-to-end latency, sensor acquisition cost, or power consumption
   was measured for this model.
7. The project audit reports that **13.6% of validation windows still overlap
   training windows** despite the block split. This affects interpretation of
   absolute validation scores for all compared models; a guard-band evaluation
   is still needed. See the [split-leakage audit](../split_leakage/).

The project also has a separately trained full AMBER run in
[`results/amber/`](../amber/). Its validation Top-1 is `0.4345` and DBA is
`0.8611`; that run followed a different training schedule and should be treated
as contextual reference rather than a matched experiment.

## Files

| File | Purpose |
|---|---|
| `comparison.csv` | Compact validation comparison against AMBER ablations |
| `lam_msac_metrics.json` | Validation metrics, selected epoch, and core model settings |
| `adaptation_metrics.json` | Aggregate secondary adaptation metrics |
| `history.csv` | Per-epoch training loss and validation metrics |
| `notebook_as_run.ipynb` | Executed Kaggle notebook with output and provenance |
| `best_lam_msac.pt` | Local trained checkpoint; intentionally not tracked by Git |

`best_lam_msac.pt` is approximately 193 MB, above GitHub's 100 MB per-file
limit, and is ignored by `.gitignore`. Keep a downloadable copy from Kaggle if
the checkpoint is needed; the tracked CSV/JSON/history and executed notebook
preserve the metrics and experiment record.
