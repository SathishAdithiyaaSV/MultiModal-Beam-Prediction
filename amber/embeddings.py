"""Positional embeddings and the modality-weight indicator — AMBER eqs. (16)-(19).

Sinusoidal embeddings, eq. (16):
    PE(k, 2i)   = sin(k / Phi^(2i/d))
    PE(k, 2i+1) = cos(k / Phi^(2i/d)),   Phi = 10000

Applied as the paper describes:
  * 2D spatial embeddings for the modalities with spatial structure (image,
    lidar, radar), along the vertical axis k in [0, VA-1] and the horizontal
    axis k in [0, HA-1], each over half the channels;
  * a temporal embedding for the sequence modalities, with k in [0, W-2] for
    beam history and k in [0, 1] for GPS.

The fusion token carries "identical spatial and temporal embeddings ... as
those added to their corresponding modality tokens" (Sec. III-B-3), so the same
embedding vector is reused for it rather than a second learned one.
"""
import torch
import torch.nn as nn

from .config import ModelConfig

_PHI = 10000.0


def sinusoidal_1d(n_positions: int, dim: int) -> torch.Tensor:
    """(n_positions, dim) table from eq. (16). `dim` must be even."""
    if dim % 2:
        raise ValueError(f"sinusoidal dim must be even, got {dim}")
    pos = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(0, dim, 2, dtype=torch.float32)
    div = torch.pow(_PHI, idx / dim)
    out = torch.zeros(n_positions, dim)
    out[:, 0::2] = torch.sin(pos / div)
    out[:, 1::2] = torch.cos(pos / div)
    return out


def sinusoidal_2d(height: int, width: int, dim: int) -> torch.Tensor:
    """(height*width, dim): half the channels encode the row, half the column."""
    if dim % 4:
        raise ValueError(f"2D sinusoidal dim must be divisible by 4, got {dim}")
    half = dim // 2
    rows = sinusoidal_1d(height, half)                  # k in [0, VA-1]
    cols = sinusoidal_1d(width, half)                   # k in [0, HA-1]
    grid = torch.cat([rows.unsqueeze(1).expand(height, width, half),
                      cols.unsqueeze(0).expand(height, width, half)], dim=-1)
    return grid.reshape(height * width, dim)


class PositionalEmbeddings(nn.Module):
    """Fixed (non-learned) embedding tables, one per modality, as buffers."""

    def __init__(self, cfg: ModelConfig, token_counts: dict[str, int]):
        super().__init__()
        d = cfg.embed_dim
        va, ha = cfg.pool_hw

        for name in ("image", "lidar", "radar"):
            n = token_counts[name]
            spatial = sinusoidal_2d(va, ha, d)
            if n != spatial.shape[0]:
                # temporal_pool="tokens": W repeats of the grid, each offset by
                # the temporal embedding for its frame
                reps = n // spatial.shape[0]
                temporal = sinusoidal_1d(reps, d)
                spatial = (spatial.unsqueeze(0) + temporal.unsqueeze(1)).reshape(n, d)
            self.register_buffer(f"pe_{name}", spatial, persistent=False)

        self.register_buffer("pe_beam", sinusoidal_1d(token_counts["beam"], d),
                             persistent=False)
        self.register_buffer("pe_gps", sinusoidal_1d(token_counts["gps"], d),
                             persistent=False)

    def get(self, modality: str) -> torch.Tensor:
        return getattr(self, f"pe_{modality}")

    def forward(self, tokens: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v + self.get(k).unsqueeze(0).to(v.dtype) for k, v in tokens.items()}


class ModalityWeightIndicator(nn.Module):
    """Learnable modality-weight indicator P_ind — AMBER eqs. (18)-(20).

        alpha = softmax(w_mod / tau_mod)          (18)
        xi_i  = alpha_i * xi_i                    (19)
        L2    = sum_i m_i * w_i^2                 (20)

    Note that alpha sums to 1 across the five modalities, so eq. (19) shrinks
    every modality's features by roughly 1/5. That is what the paper specifies,
    and the subsequent LayerNorms absorb the scale; it is called out here
    because it looks like a bug and is not one.
    """

    def __init__(self, modalities: tuple[str, ...], temperature: float):
        super().__init__()
        self.modalities = modalities
        self.temperature = temperature
        self.weights = nn.Parameter(torch.zeros(len(modalities)))

    def alpha(self) -> torch.Tensor:
        return torch.softmax(self.weights / self.temperature, dim=0)

    def forward(self, tokens: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        a = self.alpha()
        return {name: tokens[name] * a[i] for i, name in enumerate(self.modalities)}

    def l2_penalty(self, availability: torch.Tensor) -> torch.Tensor:
        """eq. (20). availability: (B, n_modalities) of 0/1 -> scalar."""
        return (availability.to(self.weights.dtype) * self.weights.pow(2)).sum(dim=1).mean()
