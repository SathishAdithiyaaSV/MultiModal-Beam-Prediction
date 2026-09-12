"""Class-former-aided modality alignment — AMBER eqs. (32)-(34), Sec. III-B-4.

A learnable class query per modality, plus one global query for the fusion
branch, each abstracting token-level features into a single class vector via
cross-attention:

    c_j = CA(c_j, Z'_j),  j in I        (32)
    c_f = CA(c_f, Z'_F)                 (33)

The queries are then aligned by an InfoNCE contrastive loss over the
mini-batch, eq. (34): for a modality query c_j, its positive is the fusion
query c_f of the SAME sample and its negatives are the fusion queries of the
other samples in the batch.

Per the paper's Remark, this module is used during TRAINING ONLY — it acts as a
regulariser that shapes the representations. The fusion features used for beam
prediction come from the fusion block itself, which runs in both phases. So
`AMBER.forward` skips the CMA at eval time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .transformer import MultiHeadAttention, TokenLayout


class ClassFormer(nn.Module):
    """One learnable class query attending over a token sequence, eqs. (32)-(33)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, cfg.embed_dim) * 0.02)
        self.attn = MultiHeadAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, tokens: torch.Tensor,
                key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens (B, L, C) -> (B, C) class vector."""
        b = tokens.shape[0]
        q = self.query.unsqueeze(0).expand(b, -1, -1)          # (B, 1, C)
        mask = None if key_mask is None else key_mask.unsqueeze(1)
        return self.norm(self.attn(q, tokens, mask) + q).squeeze(1)


class CMAModule(nn.Module):
    """Per-modality class-formers, the fusion class-former, and eq. (34)."""

    def __init__(self, cfg: ModelConfig, layout: TokenLayout):
        super().__init__()
        self.layout = layout
        self.temperature = cfg.contrastive_temperature
        self.modality_formers = nn.ModuleDict(
            {m: ClassFormer(cfg) for m in layout.order})
        self.fusion_former = ClassFormer(cfg)

    def forward(self, z_specific: torch.Tensor,
                z_fusion: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """z_specific (B, total, C) = Z'; z_fusion (B, total, C) = Z'_F.

        Returns the per-modality class queries and the fusion class query.
        """
        class_queries = {}
        for modality, former in self.modality_formers.items():
            lo, hi = self.layout.span(modality)
            class_queries[modality] = former(z_specific[:, lo:hi])
        return class_queries, self.fusion_former(z_fusion)

    def contrastive_loss(self, class_queries: dict[str, torch.Tensor],
                         fusion_query: torch.Tensor,
                         availability: torch.Tensor) -> torch.Tensor:
        """eq. (34), averaged over available modalities and the batch.

        availability: (B, n_modalities) of 0/1, columns in `layout.order`.
        Missing modalities carry no meaningful class query, so they are excluded
        from the average rather than pulled towards the fusion representation.
        """
        f = F.normalize(fusion_query, dim=-1)
        batch = f.shape[0]
        if batch < 2:
            # InfoNCE needs at least one negative
            return fusion_query.new_zeros(())

        target = torch.arange(batch, device=f.device)
        total = fusion_query.new_zeros(())
        weight = fusion_query.new_zeros(())

        for i, modality in enumerate(self.layout.order):
            c = F.normalize(class_queries[modality], dim=-1)
            logits = c @ f.t() / self.temperature              # (B, B) cosine sims
            per_sample = F.cross_entropy(logits, target, reduction="none")
            m = availability[:, i].to(per_sample.dtype)
            total = total + (per_sample * m).sum()
            weight = weight + m.sum()

        return total / weight.clamp(min=1.0)
