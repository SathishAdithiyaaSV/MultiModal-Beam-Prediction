"""Inference-available sensor-quality indicators, for the routing gate.

Every feature here is computed from the *input* tensors. None touches the beam
label, the 64-dim power vector, or whether any model happened to be correct.
Those are admissible only as offline supervision when training the gate, never
as gate inputs.

## The cost model this assumes, stated because it decides what is admissible

The project measures cost in forward GFLOPs, i.e. **processing** cost, and the
camera encoder is ~48 GFLOPs against GPS's 0.39. These indicators cost well
under 0.01 GFLOPs -- a few reductions over a 256x256 tensor. So under a
processing-cost model, computing an image-sharpness statistic and *then*
deciding whether to run ResNet34 is a genuine saving.

That is **not** true under a sensing-cost model, where the expensive thing is
powering and reading the sensor at all. If the camera must be acquired before
any statistic can be computed, image-derived features are unavailable to the
gate by construction.

The two cases are separated in the experiments: feature groups A and B use only
signals available before any expensive sensor is touched, and group C adds
image and LiDAR statistics, which are legitimate only under the processing-cost
model. Results are reported for both so the claim can be read either way.
"""
from __future__ import annotations

import numpy as np
import torch


def image_quality(frames: torch.Tensor) -> dict[str, float]:
    """Sharpness, contrast and saturation statistics for a (W, 3, H, W') stack.

    Frames arrive already standardised by eq. (10), so absolute brightness is
    gone by construction -- these describe structure, which is what survives.
    `img_sharpness` is the mean squared Laplacian response, the standard
    blur/defocus indicator. Measured on a smooth scene it falls by ~15x under
    blur and ~2x under occlusion, while additive noise *raises* it, so
    `img_hf_ratio` -- fine-scale over coarse-scale energy -- is what separates
    noise from blur: it collapses 0.30 -> 0.005 under blur and climbs
    0.30 -> 12.6 under noise.

    A plain contrast statistic is deliberately absent: eq. (10) forces unit
    standard deviation per image, so it is pinned at 1.0 and carries no
    information. Same reason brightness cannot be studied at all.
    """
    x = frames.float()
    lap = (x[..., 1:-1, 1:-1] * 4
           - x[..., :-2, 1:-1] - x[..., 2:, 1:-1]
           - x[..., 1:-1, :-2] - x[..., 1:-1, 2:])
    # high-frequency energy at two scales: noise lifts the fine scale much more
    # than the coarse one, which is what distinguishes it from blur
    coarse = torch.nn.functional.avg_pool2d(x.flatten(0, 1), 4).view(
        x.shape[0], x.shape[1], x.shape[2] // 4, x.shape[3] // 4)
    lap_c = (coarse[..., 1:-1, 1:-1] * 4
             - coarse[..., :-2, 1:-1] - coarse[..., 2:, 1:-1]
             - coarse[..., 1:-1, :-2] - coarse[..., 1:-1, 2:])
    fine, crse = float(lap.pow(2).mean()), float(lap_c.pow(2).mean())
    return {
        "img_sharpness": fine,
        "img_sharpness_coarse": crse,
        "img_hf_ratio": fine / max(crse, 1e-9),
        "img_dark_frac": float((x < x.mean() - 2 * x.std()).float().mean()),
        "img_temporal_change": float((x[1:] - x[:-1]).abs().mean()) if len(x) > 1 else 0.0,
    }


def lidar_quality(maps: torch.Tensor) -> dict[str, float]:
    """Density and occupancy statistics for a (W, 1, H, W') BEV stack.

    The BEV cell value is count / 5, so `lidar_points` recovers the return
    count directly. Point density is the natural LiDAR reliability indicator
    and is exactly what point dropout degrades.
    """
    x = maps.float()
    counts = x * 5.0
    occupied = (x > 0).float()
    return {
        "lidar_points": float(counts.sum() / max(len(x), 1)),
        "lidar_occupancy": float(occupied.mean()),
        "lidar_mean_count": float(counts.sum() / occupied.sum().clamp(min=1.0)),
        "lidar_temporal_change": float((occupied[1:] - occupied[:-1]).abs().mean())
        if len(x) > 1 else 0.0,
    }


def radar_quality(maps: torch.Tensor) -> dict[str, float]:
    """Energy concentration for a (W, 2, H, W') RA/RV stack.

    No synthetic radar degradation is studied (see amber.degrade.REJECTED), but
    these still matter: radar is genuinely absent for ~60% of scenario-34
    samples, and a concentration statistic distinguishes a map with a clear
    target return from a diffuse one.
    """
    x = maps.float()
    flat = x.flatten(1)
    peak = flat.max(dim=1).values
    return {
        "radar_peak": float(peak.mean()),
        "radar_mean": float(flat.mean()),
        "radar_peak_to_mean": float((peak / flat.mean(dim=1).clamp(min=1e-9)).mean()),
        "radar_energy_frac_top1pct": float(
            flat.topk(max(1, flat.shape[1] // 100), dim=1).values.sum()
            / flat.sum().clamp(min=1e-9)),
    }


def gps_quality(gps: torch.Tensor) -> dict[str, float]:
    """Displacement and plausibility statistics for the (2, 2) GPS tensor.

    Only two observations exist, so the available signals are the step between
    them and whether either sits at a clipped boundary -- the latter is how an
    implausible or heavily-noised position reveals itself, since _load_gps
    clips to [0, 1].
    """
    x = gps.float()
    step = (x[1] - x[0])
    at_bound = ((x <= 1e-6) | (x >= 1 - 1e-6)).float().mean()
    return {
        "gps_step": float(step.norm()),
        "gps_step_x": float(step[0].abs()),
        "gps_step_y": float(step[1].abs()),
        "gps_at_boundary": float(at_bound),
        "gps_radius": float(x.mean(dim=0).norm()),
    }


def availability(mask: torch.Tensor, modalities) -> dict[str, float]:
    """The eq. (3) mask as named features, plus how many modalities are present.

    Free, and the most obviously legitimate gate input: a system always knows
    which of its sensors reported.
    """
    out = {f"avail_{m}": float(mask[i]) for i, m in enumerate(modalities)}
    out["avail_count"] = float(mask.sum())
    return out


def prediction_signals(logits: torch.Tensor) -> dict[str, float]:
    """Uncertainty statistics from the cheap model's own output distribution.

    Already paid for -- the cheap model has run. Entropy and the top-1/top-2
    margin are the standard uncertainty signals; `top1_conf` alone is what the
    oracle-routing analysis used, where it reached AUC 0.686.
    """
    p = torch.softmax(logits.float(), dim=-1)
    top = p.topk(min(5, p.shape[-1]))
    v = top.values
    return {
        "top1_conf": float(v[0]),
        "top2_conf": float(v[1]),
        "margin_12": float(v[0] - v[1]),
        "margin_15": float(v[0] - v[4]),
        "entropy": float(-(p * p.clamp(min=1e-12).log()).sum()),
        "top5_mass": float(v.sum()),
        "pred_beam_norm": float(top.indices[0]) / max(p.shape[-1] - 1, 1),
    }


# Which features belong to which group. Group membership encodes the cost model
# argument above: A and B need no expensive sensor read, C does.
FEATURE_GROUPS = {
    "A_confidence_only": ["top1_conf"],
    "B_confidence_plus_uncertainty_availability": [
        "top1_conf", "top2_conf", "margin_12", "margin_15", "entropy", "top5_mass",
        "avail_image", "avail_lidar", "avail_radar", "avail_gps", "avail_count",
        "gps_step", "gps_step_x", "gps_step_y", "gps_at_boundary", "gps_radius",
    ],
    "C_plus_sensor_quality": None,   # every feature; filled in at run time
}
