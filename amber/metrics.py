"""Evaluation metrics — AMBER eqs. (37)-(39).

Top-K accuracy, eq. (37):
    (1/N) sum_i 1{ y_i in Q_K }

DBA score, eqs. (38)-(39) — how CLOSE the top-K predictions are, not just
whether one is exactly right:
    Y_k       = 1 - (1/N) sum_i min( min_{k'<=k} |y_i - yhat_{i,k'}| / Delta, 1 )
    DBA score = (1/K) sum_{k=1..K} Y_k

Note the nested minimum: Y_k uses the BEST of the first k predictions, so Y_k
is non-decreasing in k. Delta normalises beam distance; a prediction Delta or
more beams away scores zero for that sample.
"""
from collections.abc import Sequence

import numpy as np
import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                  ks: Sequence[int] = (1, 3, 5)) -> dict[int, float]:
    """eq. (37) for each k. logits (N, n_beams); targets (N,)."""
    if len(targets) == 0:
        return {k: float("nan") for k in ks}
    max_k = min(max(ks), logits.shape[1])
    ranked = logits.topk(max_k, dim=1).indices                  # (N, max_k)
    hit = ranked == targets.unsqueeze(1)
    return {k: hit[:, :min(k, max_k)].any(dim=1).float().mean().item() for k in ks}


def dba_score(logits: torch.Tensor, targets: torch.Tensor, k: int = 3,
              delta: float = 5.0) -> tuple[float, list[float]]:
    """eqs. (38)-(39). Returns (DBA score, [Y_1 .. Y_k])."""
    if len(targets) == 0:
        return float("nan"), [float("nan")] * k
    k = min(k, logits.shape[1])
    ranked = logits.topk(k, dim=1).indices.to(torch.float32)
    truth = targets.unsqueeze(1).to(torch.float32)
    dist = (ranked - truth).abs() / delta                       # (N, k)
    best = dist.cummin(dim=1).values.clamp(max=1.0)             # min over k' <= k
    ys = (1.0 - best.mean(dim=0)).tolist()                      # Y_1 .. Y_k
    return float(np.mean(ys)), ys


def evaluate(logits: torch.Tensor, targets: torch.Tensor,
             ks: Sequence[int] = (1, 3, 5), dba_k: int = 3,
             delta: float = 5.0) -> dict[str, float]:
    """All metrics for one set of predictions."""
    out = {f"top{k}": v for k, v in topk_accuracy(logits, targets, ks).items()}
    score, ys = dba_score(logits, targets, dba_k, delta)
    out["dba"] = score
    out.update({f"dba_y{i+1}": y for i, y in enumerate(ys)})
    out["n"] = float(len(targets))
    return out


def evaluate_by_group(logits: torch.Tensor, targets: torch.Tensor,
                      groups: Sequence[str], **kwargs) -> dict[str, dict[str, float]]:
    """Metrics overall and per group — used for the per-scenario breakdown."""
    results = {"overall": evaluate(logits, targets, **kwargs)}
    groups = np.asarray(groups)
    for g in sorted(set(groups.tolist())):
        sel = torch.from_numpy(groups == g)
        results[g] = evaluate(logits[sel], targets[sel], **kwargs)
    return results
