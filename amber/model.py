"""AMBER, end to end — Fig. 2 and Sec. III.

Forward pass, following the paper's three stages:

  1. modality-specific encoders            eqs. (9)-(13)   -> per-modality tokens
  2. embeddings + modality-weight indicator eqs. (14)-(19)  -> weighted tokens
  3. modality-specific block (masked MSA)   eqs. (21)-(27)  -> Zbar, Z'
     modality-fusion block (masked MCA)     eqs. (28)-(31)  -> Zbar_F, Z'_F
     class-former alignment, training only  eqs. (32)-(34)  -> contrastive loss
  4. beam prediction head                   Sec. III-C      -> 64 logits

A missing modality is handled in three places, which together are what makes
the model robust to arbitrary missing-modality patterns:
  * its input tensor is zeroed, so the encoder cannot read stale values;
  * the fusion mask of eq. (30) stops the fusion token attending to it;
  * it is excluded from the contrastive average and the eq. (20) penalty.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import MODALITIES, ModelConfig
from .cma import CMAModule
from .embeddings import ModalityWeightIndicator, PositionalEmbeddings
from .encoders import build_encoders
from .transformer import ModalityFusionBlock, ModalitySpecificBlock, TokenLayout


@dataclass
class AmberOutput:
    """Everything the training step needs from one forward pass."""
    logits: torch.Tensor                       # (B, n_beams)
    contrastive: torch.Tensor                  # scalar; 0 when CMA is inactive
    reg: torch.Tensor                          # scalar, eq. (20)
    alpha: torch.Tensor                        # (n_modalities,), eq. (18)


class BeamPredictionHead(nn.Module):
    """Sec. III-C: "a dimensionality expansion layer followed by a fully
    connected network that outputs the predicted beam index".

    The fusion output Zbar_F is a token sequence, so it is mean-pooled over
    tokens before the expansion. Pooling rather than flattening keeps the head
    independent of VA/HA, so changing the pooling grid does not resize it.
    """

    def __init__(self, cfg: ModelConfig, expansion: int = 4):
        super().__init__()
        d = cfg.embed_dim
        self.net = nn.Sequential(
            nn.Linear(d, d * expansion),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * expansion, cfg.n_beams),
        )

    def forward(self, fusion: torch.Tensor) -> torch.Tensor:
        return self.net(fusion.mean(dim=1))


class AMBER(nn.Module):
    """The full Adaptive Multimodal Mask Transformer for beam prediction."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoders = build_encoders(cfg)

        token_counts = {
            "image": self.encoders["image"].n_tokens,
            "lidar": self.encoders["lidar"].n_tokens,
            "radar": self.encoders["radar"].n_tokens,
            "beam": cfg.n_beam_tokens,
            "gps": cfg.n_gps_tokens,
        }
        self.layout = TokenLayout(MODALITIES, token_counts)

        self.pos = PositionalEmbeddings(cfg, token_counts)
        self.indicator = ModalityWeightIndicator(MODALITIES, cfg.modality_temperature)
        self.specific_block = ModalitySpecificBlock(cfg, self.layout)
        self.fusion_block = ModalityFusionBlock(cfg, self.layout)
        self.cma = CMAModule(cfg, self.layout)
        self.head = BeamPredictionHead(cfg)

    # ------------------------------------------------------------------ helpers
    def _concat_positional(self) -> torch.Tensor:
        """(total, C): the modality positional embeddings laid out in token order.

        Reused verbatim for the fusion token, per Sec. III-B-3.
        """
        return torch.cat([self.pos.get(m) for m in self.layout.order], dim=0)

    @staticmethod
    def _zero_missing(tokens: dict[str, torch.Tensor],
                      availability: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for i, (name, value) in enumerate(tokens.items()):
            keep = availability[:, i].view(-1, *([1] * (value.dim() - 1)))
            out[name] = value * keep.to(value.dtype)
        return out

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict[str, torch.Tensor],
                use_cma: bool | None = None) -> AmberOutput:
        """`batch` keys:
            image (B,W,3,H,W'), lidar (B,W,1,H,W'), radar (B,W,2,H,W'),
            beam  (B,W-1,2),    gps   (B,2,2),
            availability (B,5) in MODALITIES order.
        """
        availability = batch["availability"]
        use_cma = self.training if use_cma is None else use_cma

        # 1. modality-specific encoders, eqs. (9)-(13)
        raw = {m: batch[m] for m in MODALITIES}
        raw = self._zero_missing(raw, availability)
        if self.cfg.skip_unavailable_encoders:
            # A modality unavailable for the whole batch contributes nothing, so
            # its encoder need not run. Its tokens are still emitted as zeros to
            # keep the sequence layout fixed.
            batch_size = availability.shape[0]
            tokens = {}
            for i, m in enumerate(MODALITIES):
                if bool(availability[:, i].any()):
                    tokens[m] = self.encoders[m](raw[m])
                else:
                    tokens[m] = raw[m].new_zeros(
                        batch_size, self.layout.counts[m], self.cfg.embed_dim)
        else:
            tokens = {m: self.encoders[m](raw[m]) for m in MODALITIES}

        # 2. positional embeddings then the weight indicator, eqs. (16)-(19)
        tokens = self.pos(tokens)
        tokens = self.indicator(tokens)
        sequence = torch.cat([tokens[m] for m in self.layout.order], dim=1)

        # 3. masked transformer blocks, eqs. (21)-(31)
        z_specific, z_specific_attn = self.specific_block(sequence)
        pos = self._concat_positional().to(sequence.dtype)
        z_fusion, z_fusion_attn = self.fusion_block(z_specific, availability, pos)

        # CMA is a training-time regulariser only (paper's Remark)
        if use_cma:
            class_queries, fusion_query = self.cma(z_specific_attn, z_fusion_attn)
            contrastive = self.cma.contrastive_loss(class_queries, fusion_query,
                                                    availability)
        else:
            contrastive = sequence.new_zeros(())

        # 4. prediction head
        logits = self.head(z_fusion)
        return AmberOutput(logits=logits, contrastive=contrastive,
                           reg=self.indicator.l2_penalty(availability),
                           alpha=self.indicator.alpha())

    # ------------------------------------------------------------------ utility
    def parameter_groups(self, weight_decay: float) -> list[dict]:
        """AdamW groups: no weight decay on norms, biases or learnable tokens."""
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() <= 1 or name.endswith(("fusion_token", "query", "weights")):
                no_decay.append(param)
            else:
                decay.append(param)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]
