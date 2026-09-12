"""Modality-specific encoders — AMBER eqs. (9)-(13), Sec. III-A.

  radar  ResNet18, first conv widened to 2 channels, classifier removed  (9)
  image  ResNet34, per-image mean/std standardisation                   (10)
  lidar  ResNet18, first conv narrowed to 1 channel                     (11)
  gps    3-layer MLP, Cartesian metres -> C                             (12)
  beam   3-layer MLP, same shape as the GPS encoder                     (13)

Reducing the temporal window
----------------------------
The paper states two things that cannot both hold literally:

  * the first conv accepts "two-channel radar input" and eq. (9) encodes a
    single instant XR[t], implying one forward pass per frame;
  * eq. (15) gives xi_I, xi_L, xi_R in R^{VA*HA x C} with no W factor, and the
    token count is N = 2(3*VA*HA + W + 1), where the W+1 = 6 accounts only for
    the beam (W-1=4) and GPS (2) tokens.

So the W per-frame feature maps must collapse into VA*HA tokens before the
transformer, yet the paper never says how, and its temporal positional
embedding is described only for the beam and GPS modalities.

`temporal_pool` selects the reconciliation:

  "concat" (default)  encode each frame at the stated channel count, tile the W
                      feature maps along the width axis, then a single adaptive
                      pool to (VA, HA). Keeps both stated facts true, preserves
                      temporal structure, and gives exactly VA*HA tokens.
  "mean"              per-frame encode, then average over time. Simplest, but
                      discards the ordering the paper emphasises.
  "tokens"            keep W*VA*HA tokens and add the temporal embedding used
                      for beam/GPS. Richer, but N no longer matches the paper.
"""
import torch
import torch.nn as nn
import torchvision

from .config import ModelConfig

_BACKBONES = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT),
    "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.DEFAULT),
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT),
}


def _resnet_trunk(name: str, in_channels: int, pretrained: bool) -> tuple[nn.Module, int]:
    """A ResNet with its avgpool/fc removed and conv1 adapted to `in_channels`.

    When the channel count changes, pretrained conv1 weights are averaged over
    the RGB axis and repeated, which preserves the learned filter shapes far
    better than reinitialising the layer.
    """
    if name not in _BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(_BACKBONES)}")
    ctor, weights = _BACKBONES[name]
    net = ctor(weights=weights if pretrained else None)

    if in_channels != 3:
        old = net.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=old.bias is not None)
        if pretrained:
            with torch.no_grad():
                mean = old.weight.mean(dim=1, keepdim=True)          # (out,1,k,k)
                new.weight.copy_(mean.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
        net.conv1 = new

    feat_dim = net.fc.in_features
    trunk = nn.Sequential(*list(net.children())[:-2])   # drop avgpool + fc
    return trunk, feat_dim


class GridEncoder(nn.Module):
    """ResNet encoder for a W-frame stack of 2D maps -> (B, VA*HA, C) tokens.

    Implements eqs. (9)-(11): backbone, adaptive pooling to (VA, HA), spatial
    flattening, then the eq. (14) MLP embedding into the common dimension C.
    """

    def __init__(self, backbone: str, in_channels: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.temporal_pool = cfg.temporal_pool
        self.trunk, feat_dim = _resnet_trunk(backbone, in_channels, cfg.pretrained)
        self.pool = nn.AdaptiveAvgPool2d(cfg.pool_hw)
        self.embed = nn.Sequential(                      # eq. (14): MLP(sf(x))
            nn.Linear(feat_dim, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
        )

    @property
    def n_tokens(self) -> int:
        n = self.cfg.n_grid_tokens
        return n * self.cfg.window if self.temporal_pool == "tokens" else n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, Cin, H, W_) -> (B, n_tokens, C)"""
        b, w = x.shape[:2]
        feats = self.trunk(x.flatten(0, 1))              # (B*W, F, h, w)
        f, h, wd = feats.shape[1:]

        if self.temporal_pool == "concat":
            # (B, F, h, W*w) -> one adaptive pool over the whole window
            feats = feats.view(b, w, f, h, wd).permute(0, 2, 3, 1, 4).reshape(b, f, h, w * wd)
            feats = self.pool(feats)
        elif self.temporal_pool == "mean":
            feats = self.pool(feats).view(b, w, f, *self.cfg.pool_hw).mean(dim=1)
        elif self.temporal_pool == "tokens":
            feats = self.pool(feats)                     # (B*W, F, VA, HA)
        else:
            raise ValueError(f"unknown temporal_pool {self.temporal_pool!r}")

        tokens = feats.flatten(2).transpose(1, 2)        # spatial flatten, sf(.)
        if self.temporal_pool == "tokens":
            tokens = tokens.reshape(b, self.n_tokens, f)
        return self.embed(tokens)


class VectorEncoder(nn.Module):
    """Three-layer MLP for the GPS and beam-history modalities, eqs. (12)-(13).

    Layer sizes follow Sec. IV-A exactly: in -> 64 -> 256 -> C, each linear
    followed by LayerNorm, with ReLU after the first two.
    """

    def __init__(self, in_dim: int, cfg: ModelConfig, hidden: int = 64):
        super().__init__()
        mid = 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, mid), nn.LayerNorm(mid), nn.ReLU(inplace=True),
            nn.Linear(mid, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim) -> (B, T, C); one token per timestep."""
        return self.net(x)


def build_encoders(cfg: ModelConfig) -> nn.ModuleDict:
    """The five modality encoders, keyed by the names in config.MODALITIES."""
    return nn.ModuleDict({
        "image": GridEncoder(cfg.image_backbone, cfg.image_channels * 1, cfg),
        "lidar": GridEncoder(cfg.lidar_backbone, cfg.lidar_channels, cfg),
        "radar": GridEncoder(cfg.radar_backbone, cfg.radar_channels, cfg),
        # GPS token = (x, y) in metres relative to the BS
        "gps": VectorEncoder(2, cfg),
        # beam token = normalised past beam index + an "is present" indicator,
        # so a missing history step is distinguishable from beam 0
        "beam": VectorEncoder(2, cfg),
    })
