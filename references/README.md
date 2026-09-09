# Reference papers

The PDFs live in this directory locally but are **not committed** — they are
copyrighted and `.gitignore` excludes `references/*.pdf`. Obtain them from IEEE
Xplore / the publishers.

| # | Paper | Role in this project |
|---|---|---|
| 1 | AMBER: An Adaptive Multimodal Mask Transformer for Beam Prediction with Missing Modalities | Main / strong baseline (Baseline 4) |
| 2 | Multimodal Transformers for Wireless Communications: A Case Study in Beam Prediction | TII 2022 DeepSense6G challenge solution; source of the preprocessing this repo adapts (Baseline 3) |
| 3 | Multimodal Deep Learning Empowered Millimeter-Wave Beam Prediction | Earlier temporal/LSTM multimodal approach (Baseline 2) |
| 4 | Resource-Efficient Beam Prediction in mmWave Communications with Multimodal Realistic Simulation Framework | Radar-only student + knowledge distillation (Baseline 5) |
| 5 | Task-Oriented Resource Allocation for Cross-Modal Semantic Communications | Conceptual motivation only; not reproduced as a baseline |

Baseline 1 is the official DeepSense6G GPS/position-only LSTM, used as a
dataset/label/evaluation sanity check.
