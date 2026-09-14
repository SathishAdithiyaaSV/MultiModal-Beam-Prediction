"""Run a trained AMBER checkpoint over a split and export beam predictions.

The official DeepSense6G test release ships without labels, so it cannot be
scored locally -- it can only be predicted on. This is that path: it writes one
row per sample with the ranked top-K beam indices, which is what a challenge
submission needs and what downstream error analysis reads.

For labelled splits (`val`, `adaptation`) it additionally prints the usual
metrics, so the same command serves both purposes.

Output in <out-dir>/:
  predictions_<split>.csv   sample_id, scenario, top1..topK (1-indexed beams),
                            the model's confidence for each, and the ground
                            truth when the split has one
  submission_<split>.csv    just the 1-indexed top-1 beam per row, in the
                            official csv's original order

Beam indices are written **1-indexed** to match the dataset's `unit1_beam`
convention, while the model works 0-indexed throughout.

Run:
  python -m amber.predict --checkpoint runs/amber/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .dataset import AmberDataset
from .metrics import evaluate_by_group
from .model import AMBER
from .train import make_loader, pick_device, to_device


def load_checkpoint(path: Path, device: torch.device) -> tuple[AMBER, ModelConfig]:
    """Rebuild the model exactly as it was trained, then load the weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = ckpt.get("model_config")
    cfg = ModelConfig(**saved) if isinstance(saved, dict) else ModelConfig()
    # backbones are restored from the checkpoint, so skip the download
    cfg.pretrained = False
    model = AMBER(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_split(model: AMBER, index_dir: Path, split: str, cfg: ModelConfig,
                  train_cfg: TrainConfig, device: torch.device,
                  batch_size: int, workers: int, topk: int) -> pd.DataFrame:
    dataset = AmberDataset(index_dir, split, cfg)
    loader = make_loader(dataset, batch_size, shuffle=False, workers=workers)

    all_logits, all_targets = [], []
    for batch in loader:
        out = model(to_device(batch, device))
        all_logits.append(out.logits.float().cpu())
        all_targets.append(batch["target"].cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    k = min(topk, cfg.n_beams)
    probs = F.softmax(logits, dim=-1)
    conf, order = probs.topk(k, dim=-1)

    frame = pd.DataFrame({
        "sample_id": dataset.sample_ids,
        "scenario": dataset.scenarios,
    })
    for j in range(k):
        frame[f"top{j+1}_beam"] = (order[:, j] + 1).numpy()      # 1-indexed
        frame[f"top{j+1}_conf"] = conf[:, j].numpy().round(6)

    labelled = (targets >= 0).numpy()
    if labelled.any():
        frame["true_beam"] = [int(t) + 1 if t >= 0 else -1 for t in targets]
        frame["correct_top1"] = (frame["top1_beam"] == frame["true_beam"]).astype(int)

    metrics = None
    if labelled.all():
        metrics = evaluate_by_group(logits, targets, dataset.scenarios,
                                    ks=train_cfg.topk, delta=train_cfg.dba_delta)
    elif labelled.any():
        mask = torch.from_numpy(labelled)
        metrics = evaluate_by_group(logits[mask], targets[mask],
                                    [s for s, m in zip(dataset.scenarios, labelled) if m],
                                    ks=train_cfg.topk, delta=train_cfg.dba_delta)
    return frame, metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Export AMBER beam predictions for a split")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--index-dir", type=Path,
                   default=Path("data/processed_amber/index"))
    p.add_argument("--split", default="test",
                   help="split name in samples.csv (test | val | adaptation | ...)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="defaults to the checkpoint's directory")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args()

    device = pick_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint {args.checkpoint}  device {device}  split {args.split!r}")
    frame, metrics = predict_split(model, args.index_dir, args.split, cfg, TrainConfig(),
                                   device, args.batch_size, args.workers, args.topk)

    pred_path = out_dir / f"predictions_{args.split}.csv"
    frame.to_csv(pred_path, index=False)
    sub_path = out_dir / f"submission_{args.split}.csv"
    frame[["top1_beam"]].rename(columns={"top1_beam": "beam"}).to_csv(sub_path, index=False)

    print(f"\n{len(frame)} predictions -> {pred_path}")
    print(f"top-1 only              -> {sub_path}")
    print(f"\npredicted beam spread: min {frame.top1_beam.min()} "
          f"max {frame.top1_beam.max()} distinct {frame.top1_beam.nunique()}")
    print(frame.groupby("scenario").size().to_string())

    if metrics is not None:
        (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
        print(f"\nmetrics ({args.split}):")
        for group, vals in metrics.items():
            print(f"  {group:<12} " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items()))
    else:
        print("\nsplit is unlabelled -- no metrics computable (predictions only)")


if __name__ == "__main__":
    main()
