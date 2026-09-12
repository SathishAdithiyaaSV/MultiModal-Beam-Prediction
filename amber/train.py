"""Train and evaluate AMBER on the preprocessed DeepSense6G index.

Follows Sec. IV-A: AdamW, a cosine schedule with five warm-up steps, random
modality masking during training, and the eq. (36) composite objective.

Usage
    python -m amber.train --epochs 20 --batch-size 16
    python -m amber.train --epochs 1 --limit-train 64 --limit-eval 64   # smoke test
    python -m amber.train --eval-only --checkpoint runs/<name>/best.pt

Results land in runs/<name>/: config.json, metrics.json, history.csv, best.pt.
Model selection is on `val` Top-1; `adaptation` and `test` are only ever scored,
never selected on.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import MODALITIES, ModelConfig, TrainConfig, as_dict
from .dataset import AmberDataset, apply_modality_dropout
from .losses import AmberLoss
from .metrics import evaluate_by_group
from .model import AMBER


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warm-up then cosine decay to ~0, as described in Sec. IV-A."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int,
                limit: int | None = None) -> DataLoader:
    if limit is not None and limit < len(dataset):
        dataset = Subset(dataset, range(limit))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=False,
                      persistent_workers=workers > 0)


@torch.no_grad()
def run_eval(model: AMBER, loader: DataLoader, device: torch.device,
             scenarios: list[str], train_cfg: TrainConfig) -> dict:
    """Score a split. Rows with no label (target < 0) are skipped."""
    model.eval()
    logits, targets, groups = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch, use_cma=False)
        logits.append(out.logits.float().cpu())
        targets.append(batch["target"].cpu())
        groups.extend(scenarios[i] for i in batch["index"].tolist())

    logits = torch.cat(logits)
    targets = torch.cat(targets)
    groups = np.asarray(groups)

    labelled = targets >= 0
    if not labelled.any():
        return {"note": "split has no labels; predictions only", "n": float(len(targets))}
    return evaluate_by_group(logits[labelled], targets[labelled],
                             groups[labelled.numpy()].tolist(),
                             ks=train_cfg.topk, delta=train_cfg.dba_delta)


def train(args) -> None:
    model_cfg = ModelConfig(
        pool_hw=(args.pool, args.pool),
        temporal_pool=args.temporal_pool,
        pretrained=not args.no_pretrained,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        modality_dropout=args.modality_dropout, seed=args.seed,
        num_workers=args.workers, device=args.device,
    )

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    device = pick_device(train_cfg.device)

    index_dir = Path(args.index_dir)
    splits = {name: AmberDataset(index_dir, name, model_cfg)
              for name in ("train", "val", "adaptation")}
    # the 'test' split exists only once the official release is in place
    try:
        splits["test"] = AmberDataset(index_dir, "test", model_cfg)
    except ValueError:
        print("note: 'test' split is empty (official test release not present)")

    loaders = {
        "train": make_loader(splits["train"], train_cfg.batch_size, True,
                             train_cfg.num_workers, args.limit_train),
    }
    for name in ("val", "adaptation", "test"):
        if name in splits:
            loaders[name] = make_loader(splits[name], train_cfg.batch_size, False,
                                        train_cfg.num_workers, args.limit_eval)

    model = AMBER(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params/1e6:.1f}M  "
          f"tokens/modality={model.layout.counts}  N={2*model.layout.total}")
    for name, ds in splits.items():
        print(f"  {name:<11} {len(ds):>5} samples")

    optimizer = torch.optim.AdamW(model.parameter_groups(train_cfg.weight_decay),
                                  lr=train_cfg.lr)
    steps_per_epoch = max(len(loaders["train"]), 1)
    scheduler = cosine_schedule(optimizer, train_cfg.warmup_steps,
                                steps_per_epoch * train_cfg.epochs)
    criterion = AmberLoss(model_cfg.n_beams, train_cfg.lambda_focal,
                          train_cfg.lambda_contrastive, train_cfg.lambda_reg,
                          train_cfg.focal_balance, train_cfg.focal_scaling,
                          train_cfg.soft_label_sigma)

    run_dir = Path(args.run_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(as_dict(model_cfg, train_cfg), indent=2))

    generator = torch.Generator(device="cpu").manual_seed(train_cfg.seed)
    history, best = [], -1.0

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running = {"loss": 0.0, "focal": 0.0, "contrastive": 0.0, "l2": 0.0}
        seen = 0
        start = time.time()

        for batch in loaders["train"]:
            batch = apply_modality_dropout(batch, train_cfg.modality_dropout, generator)
            batch = to_device(batch, device)
            # unlabelled rows cannot contribute to the focal term
            if (batch["target"] < 0).any():
                keep = batch["target"] >= 0
                if not keep.any():
                    continue
                batch = {k: v[keep] for k, v in batch.items()}

            out = model(batch)
            loss, parts = criterion(out.logits, batch["target"], out.contrastive, out.reg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            n = batch["target"].shape[0]
            for k in running:
                running[k] += parts[k] * n
            seen += n

        means = {k: v / max(seen, 1) for k, v in running.items()}
        val = run_eval(model, loaders["val"], device, splits["val"].scenarios, train_cfg)
        val_top1 = val.get("overall", {}).get("top1", float("nan"))

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
               "seconds": round(time.time() - start, 1), **means,
               "val_top1": val_top1,
               "val_top3": val.get("overall", {}).get("top3", float("nan")),
               "val_dba": val.get("overall", {}).get("dba", float("nan"))}
        history.append(row)
        print(f"epoch {epoch:>3}/{train_cfg.epochs}  loss {means['loss']:.3f} "
              f"(focal {means['focal']:.3f} cont {means['contrastive']:.3f} "
              f"l2 {means['l2']:.4f})  val top1 {val_top1:.4f} "
              f"top3 {row['val_top3']:.4f} dba {row['val_dba']:.4f}  "
              f"[{row['seconds']}s]")

        if val_top1 == val_top1 and val_top1 > best:      # skips NaN
            best = val_top1
            torch.save({"model": model.state_dict(),
                        "config": as_dict(model_cfg, train_cfg),
                        "epoch": epoch, "val_top1": val_top1}, run_dir / "best.pt")

        import csv
        with open(run_dir / "history.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    # ---- final scoring with the selected checkpoint
    checkpoint = run_dir / "best.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        print(f"\nloaded best checkpoint (val top1 {best:.4f})")

    results = {}
    for name in ("val", "adaptation", "test"):
        if name in loaders:
            results[name] = run_eval(model, loaders[name], device,
                                     splits[name].scenarios, train_cfg)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    print("\nfinal metrics")
    for split, res in results.items():
        if "overall" not in res:
            print(f"  {split}: {res.get('note', '-')}")
            continue
        for group, m in res.items():
            print(f"  {split:<11} {group:<10} n={int(m['n']):>5} "
                  f"top1 {m['top1']:.4f}  top3 {m['top3']:.4f}  dba {m['dba']:.4f}")
    print(f"\nrun directory: {run_dir}")


def evaluate_only(args) -> None:
    device = pick_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    saved = payload["config"]
    model_cfg = ModelConfig(**saved["model"])
    train_cfg = TrainConfig(**saved["train"])
    model = AMBER(model_cfg).to(device)
    model.load_state_dict(payload["model"])

    results = {}
    for name in args.splits:
        try:
            ds = AmberDataset(Path(args.index_dir), name, model_cfg)
        except ValueError as exc:
            print(f"skipping {name}: {exc}")
            continue
        loader = make_loader(ds, train_cfg.batch_size, False, args.workers, args.limit_eval)
        results[name] = run_eval(model, loader, device, ds.scenarios, train_cfg)

    print(json.dumps(results, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/evaluate AMBER on DeepSense6G 31-34")
    p.add_argument("--index-dir", default="data/processed_amber/index")
    p.add_argument("--run-dir", default="runs")
    p.add_argument("--name", default="amber")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--modality-dropout", type=float, default=TrainConfig.modality_dropout)
    p.add_argument("--pool", type=int, default=ModelConfig.pool_hw[0],
                   help="VA = HA, the adaptive-pool grid side")
    p.add_argument("--temporal-pool", default=ModelConfig.temporal_pool,
                   choices=["concat", "mean", "tokens"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--limit-train", type=int, default=None, help="cap train samples")
    p.add_argument("--limit-eval", type=int, default=None, help="cap eval samples")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--splits", nargs="+", default=["val", "adaptation", "test"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--eval-only requires --checkpoint")
        evaluate_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
