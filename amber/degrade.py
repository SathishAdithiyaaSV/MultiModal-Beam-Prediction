"""Modality degradations, chosen to be faithful to the stored representation.

Only the preprocessed tensors are available at experiment time, so a degradation
is admissible here only if applying it to that representation corresponds to a
real sensor failure. Three consequences, each documented in `REJECTED` below:

  * **Camera brightness and contrast are provably no-ops.** AMBER eq. (10)
    standardises each image by its own mean and standard deviation, and that is
    affine-invariant: x -> a*x + b becomes
    (a*x + b - a*mean - b)/(a*std) = (x - mean)/std, exactly the original.
    Verified numerically to 4e-7. Any lighting experiment on this architecture
    measures nothing.
  * **Radar degradation is not implementable faithfully.** The stored tensor is
    a post-2D-FFT magnitude pair under a single joint min-max. Noise added in
    the map domain has no clean pre-image in the IQ cube, where real noise
    would spread across the whole map after the transform. Radar unreliability
    is instead studied through the dataset's own missingness, which is
    substantial: scenario 34 supplies radar for only ~40% of its samples.
  * **LiDAR range truncation is near-vacuous.** The BEV histogram is already
    cropped to a +/-50 m region of interest containing 99% of points, so
    truncating further mostly removes empty cells.

Everything here operates on the representation the model actually consumes, and
is applied *before* standardisation for the camera, so the degradation is not
cancelled by it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

BEV_MAX_PER_CELL = 5          # the cap build_index applied; BEV value = count / 5

REJECTED = {
    "camera_brightness": "provably cancelled by eq. (10) per-image standardisation",
    "camera_contrast": "provably cancelled by eq. (10) per-image standardisation",
    "radar_noise": "RA/RV maps are post-FFT magnitudes; map-domain noise has no "
                   "physical pre-image. Natural missingness used instead",
    "radar_map_corruption": "same reason as radar_noise",
    "lidar_range_truncation": "BEV is already cropped to the +/-50 m ROI holding 99% "
                              "of points",
    "gps_fewer_observations": "only two GPS observations exist; dropping one is the "
                              "missing-modality experiment, not a degradation curve",
}


@dataclass(frozen=True)
class Degradation:
    """One degradation to apply. `severity` meaning depends on `kind`."""
    modality: str          # image | lidar | gps
    kind: str
    severity: float

    def label(self) -> str:
        return f"{self.modality}:{self.kind}={self.severity:g}"


# ---------------------------------------------------------------- camera
def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def camera_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. Defocus, motion, or a dirty lens.

    `img` is (C, H, W) in [0, 1], before standardisation.
    """
    if sigma <= 0:
        return img
    k = _gaussian_kernel(sigma, img.device)
    c = img.shape[0]
    pad = (len(k) - 1) // 2
    x = img.unsqueeze(0)
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"),
                 k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"),
                 k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x.squeeze(0)


def camera_noise(img: torch.Tensor, sigma: float,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Additive Gaussian noise. Low-light sensor noise -- scenarios 33 and 34
    are night scenes, so this is the most realistic camera failure here."""
    if sigma <= 0:
        return img
    n = torch.randn(img.shape, generator=generator, device=img.device) * sigma
    return (img + n).clamp(0.0, 1.0)


def camera_occlusion(img: torch.Tensor, area_frac: float,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Zero a contiguous block covering `area_frac` of the frame.

    A physical obstruction: dirt, a raindrop, or something blocking part of the
    field of view. Not affine, so it survives standardisation.
    """
    if area_frac <= 0:
        return img
    _c, h, w = img.shape
    side = int(round((area_frac * h * w) ** 0.5))
    side = min(side, h, w)
    if side <= 0:
        return img
    top = int(torch.randint(0, h - side + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - side + 1, (1,), generator=generator).item())
    out = img.clone()
    out[:, top:top + side, left:left + side] = 0.0
    return out


def camera_resolution(img: torch.Tensor, scale: float) -> torch.Tensor:
    """Downsample by `scale` then restore the size. A cheaper sensor, or heavy
    compression: detail is destroyed, geometry preserved."""
    if scale >= 1.0:
        return img
    _c, h, w = img.shape
    small = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode="bilinear",
                          align_corners=False, recompute_scale_factor=False)
    return F.interpolate(small, size=(h, w), mode="bilinear",
                         align_corners=False).squeeze(0)


# ---------------------------------------------------------------- lidar
def lidar_point_dropout(bev: torch.Tensor, keep: float,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """Independently drop each LiDAR return with probability 1 - `keep`.

    Exact rather than approximate: the BEV cell value is count / 5 with the
    count an integer in 0..5, so thinning the counts binomially is precisely
    "each return was independently lost". Represents a sparser sensor, heavy
    rain, or low reflectivity.
    """
    if keep >= 1.0:
        return bev
    counts = (bev * BEV_MAX_PER_CELL).round().clamp(0, BEV_MAX_PER_CELL)
    # Binomial(count, keep), built as a sum of independent Bernoulli draws over
    # the return slots. The i-th slot contributes only where count > i, which is
    # what makes this exactly "each return independently survives".
    kept = torch.zeros_like(counts)
    for i in range(BEV_MAX_PER_CELL):
        exists = (counts > i).to(counts.dtype)
        draw = (torch.rand(counts.shape, generator=generator,
                           device=bev.device) < keep).to(counts.dtype)
        kept += exists * draw
    return kept / BEV_MAX_PER_CELL


# ---------------------------------------------------------------- gps
def gps_position_noise(gps: torch.Tensor, metres: float, span: np.ndarray,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Add isotropic positional noise of `metres` standard deviation.

    `gps` is the (2, 2) min-max scaled tensor the model sees; `span` is the
    per-column (max - min) in metres from gps_norm.json, which is what converts
    a physical error into this representation. Directly interpretable: consumer
    GNSS is 1-5 m, while this dataset's positions are far more precise, so the
    curve says whether the cheap tier survives realistic localisation.
    """
    if metres <= 0:
        return gps
    scale = torch.as_tensor(metres / np.maximum(span, 1e-9), dtype=gps.dtype,
                            device=gps.device).view(2, 2)
    n = torch.randn(gps.shape, generator=generator, device=gps.device) * scale
    return (gps + n).clamp(0.0, 1.0)


# ---------------------------------------------------------------- dispatch
def apply_camera(img: torch.Tensor, deg: Degradation,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    fn = {"blur": lambda: camera_blur(img, deg.severity),
          "noise": lambda: camera_noise(img, deg.severity, generator),
          "occlusion": lambda: camera_occlusion(img, deg.severity, generator),
          "resolution": lambda: camera_resolution(img, deg.severity)}
    if deg.kind not in fn:
        raise ValueError(f"unknown camera degradation {deg.kind!r}; "
                         f"rejected kinds: {sorted(REJECTED)}")
    return fn[deg.kind]()


def grid_for(deg_kind: str) -> list[float]:
    """The severity sweep per degradation, ordered intact -> severe."""
    return {
        # sigma in pixels on a 256x256 frame
        "blur": [0.0, 1.0, 2.0, 4.0, 8.0],
        # sigma in [0,1] intensity units
        "noise": [0.0, 0.05, 0.10, 0.20, 0.40],
        # fraction of frame area obscured
        "occlusion": [0.0, 0.05, 0.15, 0.35, 0.60],
        # downsample factor before restoring size
        "resolution": [1.0, 0.5, 0.25, 0.125, 0.0625],
        # fraction of LiDAR returns retained
        "dropout": [1.0, 0.5, 0.25, 0.10, 0.02],
        # GPS error standard deviation in metres
        "position_noise": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
    }[deg_kind]
