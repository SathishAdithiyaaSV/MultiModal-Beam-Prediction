"""Torch dataset over the preprocessed AMBER index.

Reads `data/processed_amber/index/samples.csv` — produced by
`preprocessing_amber/build_index.py` — and returns exactly the tensors
`AMBER.forward` expects.

Per-sample contents:
    image         (W, 3, H, W')  standardised by each frame's own mean/std, eq. (10)
    lidar         (W, 1, H, W')  BEV histogram, already in [0, 1]
    radar         (W, 2, H, W')  RA/RV maps, already in [0, 1]
    gps           (2, 2)         (x, y) metres from the BS, min-max normalised
    beam          (W-1, 2)       (normalised past beam index, present flag)
    availability  (5,)           AMBER's m, in config.MODALITIES order
    target        ()             beam index in [0, 63], or -1 if unlabelled

Two details that matter for correctness:

  * GPS normalisation uses the constants in `index/gps_norm.json`, which were
    fit on the TRAIN split only. Normalising per split would leak split
    statistics into evaluation.
  * A missing beam-history step is encoded as (0, 0) rather than as index -1.
    Feeding -1 into the encoder would make "absent" look like a beam adjacent
    to beam 0; the explicit present flag keeps the two distinguishable.
"""
import json
from pathlib import Path

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import MODALITIES, ModelConfig

GPS_COLS = ("ue_x_1", "ue_y_1", "ue_x_2", "ue_y_2")
MASK_COLS = tuple(f"m_{m}" for m in MODALITIES)
EPS = 1e-6


class AmberDataset(Dataset):
    """One split of the preprocessed DeepSense6G index."""

    def __init__(self, index_dir: str | Path, split: str | list[str],
                 cfg: ModelConfig, root: str | Path | None = None,
                 modalities: Sequence[str] | None = None,
                 standardise_images: bool = True):
        self.index_dir = Path(index_dir)
        self.root = Path(root) if root is not None else self.index_dir.parents[2]
        self.cfg = cfg
        self.standardise_images = standardise_images

        frame = pd.read_csv(self.index_dir / "samples.csv")
        splits = [split] if isinstance(split, str) else list(split)
        frame = frame[frame["split"].isin(splits)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"no rows for split(s) {splits} in {self.index_dir/'samples.csv'}; "
                f"available: {sorted(pd.read_csv(self.index_dir/'samples.csv').split.unique())}")
        self.frame = frame

        norm = json.loads((self.index_dir / "gps_norm.json").read_text())["columns"]
        self.gps_lo = np.array([norm[c]["min"] for c in GPS_COLS], dtype=np.float32)
        self.gps_hi = np.array([norm[c]["max"] for c in GPS_COLS], dtype=np.float32)

        self.image_cols = [f"image_{i}" for i in range(1, cfg.window + 1)]
        self.lidar_cols = [f"lidar_bev_{i}" for i in range(1, cfg.window + 1)]
        self.radar_cols = [f"radar_{i}" for i in range(1, cfg.window + 1)]
        self.beam_cols = [f"beam_hist_{i}" for i in range(1, cfg.window)]
        # Modality-ablation restriction. A disabled modality is neither read from
        # disk nor marked available, which for the cheap configurations removes
        # most of the per-sample I/O -- a GPS-only configuration would otherwise
        # still load ~4 MB of image/LiDAR/radar tensors per sample only for the
        # model to ignore them.
        self.enabled = set(MODALITIES) if modalities is None else set(modalities)
        unknown = self.enabled - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities {sorted(unknown)}")
        self._shapes = self._probe_shapes()

    def __len__(self) -> int:
        return len(self.frame)

    # ------------------------------------------------------------------ loaders
    def _probe_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-frame tensor shape of each grid modality, from the first row that
        has it available.

        Needed because an unavailable modality must still yield a correctly
        shaped tensor -- the eq. (23)/(30) masks stop the model attending to it,
        but the collate still has to stack something.
        """
        shapes = {}
        for key, cols, mask in (("image", self.image_cols, "m_image"),
                                ("lidar", self.lidar_cols, "m_lidar"),
                                ("radar", self.radar_cols, "m_radar")):
            have = self.frame.index[self.frame[mask] == 1]
            if len(have) == 0:
                shapes[key] = None                # modality absent everywhere
                continue
            row = self.frame.loc[have[0]]
            if key == "image":
                with Image.open(self.root / row[cols[0]]) as img:
                    w, h = img.size
                shapes[key] = (3, h, w)
            else:
                shapes[key] = np.load(self.root / row[cols[0]]).shape
        return shapes

    def _load_images(self, row) -> torch.Tensor:
        # Unavailable modality: return zeros rather than touching the disk. The
        # files genuinely do not exist for these samples.
        if not row["m_image"] or "image" not in self.enabled:
            shape = self._shapes["image"] or (3, 256, 256)
            return torch.zeros(len(self.image_cols), *shape)
        frames = []
        for col in self.image_cols:
            with Image.open(self.root / row[col]) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            if self.standardise_images:                      # eq. (10)
                t = (t - t.mean()) / t.std().clamp(min=EPS)
            frames.append(t)
        return torch.stack(frames)

    def _load_maps(self, row, columns, key) -> torch.Tensor:
        if not row[f"m_{key}"] or key not in self.enabled:
            shape = self._shapes[key] or ((1, 256, 256) if key == "lidar" else (2, 256, 256))
            return torch.zeros(len(columns), *shape)
        return torch.stack([torch.from_numpy(
            np.load(self.root / row[col]).astype(np.float32)) for col in columns])

    def _load_gps(self, row) -> torch.Tensor:
        raw = np.array([row[c] for c in GPS_COLS], dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0.0)
        scaled = (raw - self.gps_lo) / np.maximum(self.gps_hi - self.gps_lo, EPS)
        return torch.from_numpy(scaled.clip(0.0, 1.0)).view(2, 2)

    def _load_beam_history(self, row) -> torch.Tensor:
        out = torch.zeros(len(self.beam_cols), 2)
        for i, col in enumerate(self.beam_cols):
            idx = int(row[col])
            if idx >= 0:
                out[i, 0] = idx / (self.cfg.n_beams - 1)     # normalised, eq. (13)
                out[i, 1] = 1.0                              # present flag
        return out

    # ------------------------------------------------------------------ item
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[i]
        return {
            "image": self._load_images(row),
            "lidar": self._load_maps(row, self.lidar_cols, "lidar"),
            "radar": self._load_maps(row, self.radar_cols, "radar"),
            "gps": self._load_gps(row),
            "beam": self._load_beam_history(row),
            "availability": torch.tensor(
                [float(row[c]) * (m in self.enabled)
                 for c, m in zip(MASK_COLS, MODALITIES)]),
            "target": torch.tensor(int(row["beam"]), dtype=torch.long),
            "index": torch.tensor(i, dtype=torch.long),
        }

    # ------------------------------------------------------------------ metadata
    @property
    def scenarios(self) -> list[str]:
        """Scenario label per row, for the per-scenario metric breakdown."""
        return self.frame["scenario"].tolist()

    @property
    def sample_ids(self) -> list[str]:
        return self.frame["sample_id"].tolist()


def apply_modality_dropout(batch: dict[str, torch.Tensor],
                           p: float | Sequence[float],
                           generator: torch.Generator | None = None
                           ) -> dict[str, torch.Tensor]:
    """Random modality masking during training (Sec. IV-A).

    Each present modality is independently dropped, which is what teaches the
    mask of eqs. (23)/(30) to generalise to arbitrary missing-modality patterns.
    At least one modality is always kept, since a sample with nothing available
    carries no signal to learn from.

    `p` is either one probability for all modalities, or one per modality in
    config.MODALITIES order. The per-modality form matters here because the
    official DeepSense6G test release ships no mmWave_data at all, so beam
    history is absent for 100% of test samples while present for ~96% of
    training ones. Dropping it at the paper's uniform low rate would let the
    model lean on a signal that vanishes at inference.
    """
    availability = batch["availability"]
    if isinstance(p, (int, float)):
        if p <= 0:
            return batch
        probs = torch.full((availability.shape[1],), float(p), device=availability.device)
    else:
        probs = torch.as_tensor(list(p), dtype=torch.float32, device=availability.device)
        if probs.numel() != availability.shape[1]:
            raise ValueError(f"expected {availability.shape[1]} dropout probabilities, "
                             f"got {probs.numel()}")
        if float(probs.max()) <= 0:
            return batch

    keep = (torch.rand(availability.shape, generator=generator,
                       device=availability.device) >= probs).to(availability.dtype)
    dropped = availability * keep

    # restore one modality for any row left entirely empty
    empty = dropped.sum(dim=1) == 0
    if empty.any():
        fallback = availability[empty]
        first = fallback.argmax(dim=1, keepdim=True)
        restored = torch.zeros_like(fallback).scatter_(1, first, 1.0) * fallback
        dropped[empty] = restored

    batch = dict(batch)
    batch["availability"] = dropped
    return batch
