"""Adaptive multimodal mask transformer — AMBER eqs. (21)-(31), Sec. III-B.

Two blocks, distinguished only by their attention mask:

  modality-specific block (eqs. 21-27)
      Self-attention over the concatenated token sequence under the mask of
      eq. (23), M[j,i] = 1 iff i == j, so tokens attend only within their own
      modality. Concatenating and masking is mathematically identical to
      running a separate transformer per modality, but does it in one batched
      call, which is exactly the paper's stated reason for the formulation.

  modality-fusion block (eqs. 28-31)
      Cross-attention where a learnable fusion token sequence is the query and
      the keys/values are the modality tokens plus the fusion token itself,
      under the mask of eq. (30): the fusion branch attends to itself and to
      every AVAILABLE modality. This is where the availability vector m from
      eq. (3) actually changes the computation.

Both blocks are pre-norm-free (post-norm), matching eqs. (26)/(27):
    Z'   = LN(MSA(xi) + xi)
    Zbar = LN(FFN(Z') + Z')
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

NEG_INF = -1e9   # finite, so a fully masked row cannot produce NaN


@dataclass(frozen=True)
class TokenLayout:
    """Where each modality's tokens sit in the concatenated sequence."""
    order: tuple[str, ...]
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts[m] for m in self.order)

    def span(self, modality: str) -> tuple[int, int]:
        start = 0
        for m in self.order:
            if m == modality:
                return start, start + self.counts[m]
            start += self.counts[m]
        raise KeyError(modality)

    def modality_of_token(self) -> torch.Tensor:
        """(total,) int tensor mapping each token to its modality index."""
        ids = [torch.full((self.counts[m],), i, dtype=torch.long)
               for i, m in enumerate(self.order)]
        return torch.cat(ids)


def block_diagonal_mask(layout: TokenLayout) -> torch.Tensor:
    """eq. (23): (total, total) bool, True where a token may attend."""
    ids = layout.modality_of_token()
    return ids.unsqueeze(0) == ids.unsqueeze(1)


def availability_mask(layout: TokenLayout, availability: torch.Tensor) -> torch.Tensor:
    """eq. (30) key mask: (B, total) bool, True where the key is usable.

    availability: (B, n_modalities) of 0/1 in `layout.order`.
    """
    ids = layout.modality_of_token().to(availability.device)
    return availability.bool()[:, ids]


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with an explicit additive mask, eqs. (21)-(25)."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"embed_dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)              # W^O in eq. (25)
        self.dropout = dropout

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """query (B,Lq,C), key_value (B,Lk,C), mask (B,Lq,Lk) or (Lq,Lk) bool."""
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(key_value))
        v = self._heads(self.v_proj(key_value))

        attn_bias = None
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # a row with no visible key would softmax to NaN; let it attend to
            # everything and rely on the caller discarding that row
            dead = ~mask.any(dim=-1, keepdim=True)
            mask = mask | dead
            attn_bias = torch.zeros(mask.shape, dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(~mask, NEG_INF).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).flatten(2)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """The FFN of eq. (27)."""

    def __init__(self, dim: int, ratio: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """One post-norm layer. Self-attention when key_value is None."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = FeedForward(cfg.embed_dim, cfg.ffn_ratio, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: torch.Tensor, key_value: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        kv = x if key_value is None else key_value
        attended = self.norm1(self.attn(x, kv, mask) + x)          # eq. (26)/(29)
        out = self.norm2(self.ffn(attended) + attended)            # eq. (27)/(31)
        return out, attended        # `attended` is Z', which the CMA consumes


class ModalitySpecificBlock(nn.Module):
    """Stack of layers under the block-diagonal mask of eq. (23)."""

    def __init__(self, cfg: ModelConfig, layout: TokenLayout):
        super().__init__()
        self.layout = layout
        self.layers = nn.ModuleList(TransformerLayer(cfg)
                                    for _ in range(cfg.n_layers_specific))
        self.register_buffer("mask", block_diagonal_mask(layout), persistent=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens (B, total, C) -> (Zbar, Z') both (B, total, C)."""
        x, attended = tokens, tokens
        for layer in self.layers:
            x, attended = layer(x, mask=self.mask)
        return x, attended


class ModalityFusionBlock(nn.Module):
    """Learnable fusion token + masked cross-attention, eqs. (28)-(31).

    The fusion token has "the same sequence length and feature dimension as the
    summation of the modality-specific tokens" (eq. 28), i.e. `layout.total`
    tokens of width C, which is what makes the paper's N = 2 * layout.total.
    """

    def __init__(self, cfg: ModelConfig, layout: TokenLayout):
        super().__init__()
        self.layout = layout
        self.fusion_token = nn.Parameter(torch.randn(layout.total, cfg.embed_dim) * 0.02)
        self.layers = nn.ModuleList(TransformerLayer(cfg)
                                    for _ in range(cfg.n_layers_fusion))

    def forward(self, modality_tokens: torch.Tensor, availability: torch.Tensor,
                pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """modality_tokens (B, total, C); availability (B, n_modalities);
        pos (total, C) the same positional embeddings the modalities received.

        Returns (Zbar_F, Z'_F), both (B, total, C).
        """
        b = modality_tokens.shape[0]
        fusion = self.fusion_token.unsqueeze(0).expand(b, -1, -1) + pos.unsqueeze(0)

        # keys/values = [modality tokens, fusion tokens]; eq. (30) mask
        kv = torch.cat([modality_tokens, fusion], dim=1)
        key_ok = torch.cat([availability_mask(self.layout, availability),
                            torch.ones(b, self.layout.total, dtype=torch.bool,
                                       device=modality_tokens.device)], dim=1)
        mask = key_ok.unsqueeze(1).expand(b, self.layout.total, kv.shape[1])

        x, attended = fusion, fusion
        for layer in self.layers:
            x, attended = layer(x, key_value=kv, mask=mask)
            kv = torch.cat([modality_tokens, x], dim=1)
        return x, attended
