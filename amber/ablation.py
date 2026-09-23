"""Helpers for controlled modality-ablation experiments.

The point of an ablation here is to vary *only* which modalities the model may
use, holding everything else fixed. AMBER already has the mechanism for that:
the eq. (3) availability vector, which the eq. (23)/(30) masks consume. So a
configuration is expressed by forcing that vector rather than by building a
different architecture per configuration.

Verified property this relies on: a modality marked unavailable has **exactly
zero** influence on the logits. Perturbing a disabled modality's raw input by a
large amount moves the output by 0.0, while perturbing an enabled one moves it
normally. That is what makes both the restriction and the encoder skipping in
`ModelConfig.skip_unavailable_encoders` numerically faithful rather than
approximate.

Nothing here changes default repository behaviour; it is additive.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

import torch

from .config import MODALITIES

# The four sensing modalities the DeepSense6G challenge specifies as inputs.
# `beam` (historical beam indices) is an AMBER addition, not part of the
# challenge input spec -- see the notebooks' discussion of USE_HISTORICAL_BEAMS.
SENSING_MODALITIES = ("gps", "radar", "image", "lidar")

# Display names, so tables and plots read the way the paper would phrase them.
PRETTY = {"gps": "GPS", "radar": "Radar", "image": "Camera",
          "lidar": "LiDAR", "beam": "BeamIdx"}


def config_label(enabled: Sequence[str]) -> str:
    """'GPS + Radar + Camera' for a configuration, in a stable canonical order."""
    order = [m for m in ("gps", "radar", "image", "lidar", "beam") if m in enabled]
    return " + ".join(PRETTY[m] for m in order)


def parse_modalities(names: Sequence[str]) -> tuple[str, ...]:
    """Validate a modality list against config.MODALITIES.

    Accepts the friendly aliases 'camera' and 'rgb' for 'image'.
    """
    alias = {"camera": "image", "rgb": "image", "beamidx": "beam",
             "beam_history": "beam", "position": "gps"}
    out = []
    for raw in names:
        name = alias.get(str(raw).strip().lower(), str(raw).strip().lower())
        if name not in MODALITIES:
            raise ValueError(
                f"unknown modality {raw!r}; choose from {sorted(MODALITIES)} "
                f"(aliases: {sorted(alias)})")
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("a configuration must enable at least one modality")
    return tuple(out)


def enabled_mask(enabled: Sequence[str], device=None) -> torch.Tensor:
    """(n_modalities,) float mask in config.MODALITIES order."""
    enabled = set(parse_modalities(enabled))
    return torch.tensor([1.0 if m in enabled else 0.0 for m in MODALITIES],
                        dtype=torch.float32, device=device)


def restrict_availability(batch: dict[str, torch.Tensor],
                          mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """AND the batch's natural availability with a configuration's mask.

    Natural availability is preserved: a configuration cannot conjure a modality
    that is genuinely absent for a sample (scenario 34 has no radar for ~60% of
    its samples, and the official test split has no beam history at all). So the
    mask can only ever remove, never add -- which is what keeps the comparison
    honest.
    """
    if mask is None:
        return batch
    out = dict(batch)
    out["availability"] = batch["availability"] * mask.to(batch["availability"].device)
    return out


def effective_availability(frame, enabled: Sequence[str]) -> dict[str, float]:
    """Per-modality fraction of rows that will actually be available.

    `frame` is an index DataFrame (or slice of one). Reported alongside results
    so a configuration's numbers can be read against what it really received --
    'GPS + Radar' on scenario 34 is partly 'GPS only' in practice.
    """
    enabled = set(parse_modalities(enabled))
    return {m: (float(frame[f"m_{m}"].mean()) if m in enabled else 0.0)
            for m in MODALITIES}


# --------------------------------------------------------------------- cost
def active_parameters(model, enabled: Sequence[str]) -> dict[str, int]:
    """Parameter counts: total, and only those a configuration actually uses.

    A configuration never runs the encoders of its disabled modalities, so
    counting the whole model would overstate every configuration equally and
    hide the cost differences the experiment is about.
    """
    enabled = set(parse_modalities(enabled))
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for name, enc in model.encoders.items():
        if name not in enabled:
            inactive += sum(p.numel() for p in enc.parameters())
    return {"params_total": total, "params_active": total - inactive,
            "params_encoders_skipped": inactive}


@torch.no_grad()
def measure_latency(model, batch: dict[str, torch.Tensor], device,
                    warmup: int = 3, iters: int = 10) -> float:
    """Mean forward-pass latency in ms per sample. Indicative, not a benchmark.

    Reported as supporting evidence for the accuracy-vs-cost trade-off, so it is
    deliberately simple: same batch, same device, warmed up, CUDA-synchronised.
    """
    model.eval()
    n = batch["availability"].shape[0]
    for _ in range(warmup):
        model(batch, use_cma=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(batch, use_cma=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters / n * 1000.0


@torch.no_grad()
def measure_gflops(model, batch: dict[str, torch.Tensor]) -> float | None:
    """Forward GFLOPs per sample via torch's FlopCounterMode, or None.

    Returns None rather than an estimate if the counter is unavailable, so a
    missing value is never mistaken for a measured one.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    model.eval()
    n = batch["availability"].shape[0]
    try:
        counter = FlopCounterMode(display=False)
        with counter:
            model(batch, use_cma=False)
        return counter.get_total_flops() / n / 1e9
    except Exception:
        return None
