"""AMBER model and training hyperparameters.

Values marked [Table II] are taken from the AMBER paper's system-parameter
table. Values marked [UNSPECIFIED] are not given numerically anywhere in the
paper; each is a documented default of ours, overridable from the CLI and
recorded into the run directory so every result stays traceable.

Reference: "AMBER: An Adaptive Multimodal Mask Transformer for Beam Prediction
with Missing Modalities", Wen, Shi, Li, Zhao, Zhao & Wang.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple

# The five modalities, in the order AMBER lists them in m = [mI, mL, mR, mB, mG].
MODALITIES: Tuple[str, ...] = ("image", "lidar", "radar", "beam", "gps")


@dataclass
class ModelConfig:
    # ---- task
    n_beams: int = 64                  # codebook size K            [Table II]
    window: int = 5                    # sliding window W           [Table II]

    # ---- token/embedding geometry
    embed_dim: int = 256               # common embedding dim C, stated for the
                                       # GPS encoder in Sec. IV-A
    n_heads: int = 8                   # H                          [Table II]
    dropout: float = 0.1               # in attention and FFN       [Table II]
    pool_hw: Tuple[int, int] = (4, 4)  # (VA, HA) adaptive-pool grid [UNSPECIFIED]
    n_layers_specific: int = 2         # modality-specific block depth [UNSPECIFIED]
    n_layers_fusion: int = 2           # modality-fusion block depth   [UNSPECIFIED]
    ffn_ratio: int = 4                 # FFN hidden = ratio * C        [UNSPECIFIED]

    # ---- encoder backbones (Sec. III-A)
    image_backbone: str = "resnet34"   # deeper, per the paper's reasoning
    lidar_backbone: str = "resnet18"
    radar_backbone: str = "resnet18"
    pretrained: bool = True            # "we adopt a pretrained ResNet-18/34"

    # ---- per-frame input channels
    image_channels: int = 3
    lidar_channels: int = 1            # BEV histogram
    radar_channels: int = 2            # range-angle + range-velocity

    # How the W per-frame feature maps of image/lidar/radar are reduced to the
    # VA*HA tokens that eq. (15) and the token count N = 2(3*VA*HA + W + 1)
    # both require. See amber/encoders.py for why this needs a choice at all.
    temporal_pool: str = "concat"      # concat | mean | tokens      [UNSPECIFIED]

    # ---- modality-weight indicator, eq. (18)
    modality_temperature: float = 1.0  # tau_mod                     [UNSPECIFIED]

    # ---- CMA, eq. (34)
    contrastive_temperature: float = 0.07   # tau                    [UNSPECIFIED]

    @property
    def n_gps_tokens(self) -> int:
        return 2                       # tau = t-1 .. t

    @property
    def n_beam_tokens(self) -> int:
        return self.window - 1         # tau = t-W+1 .. t-1

    @property
    def n_grid_tokens(self) -> int:
        va, ha = self.pool_hw
        return va * ha


@dataclass
class TrainConfig:
    batch_size: int = 16               #                             [Table II]
    lr: float = 1e-4                   #                             [Table II]
    epochs: int = 20                   #                             [Table II]
    optimizer: str = "adamw"           #                             [Table II]
    weight_decay: float = 0.05         # "AdamW with weight decay"   [UNSPECIFIED]
    warmup_steps: int = 5              # "five warm-up steps"        Sec. IV-A
    grad_clip: float = 1.0             #                             [UNSPECIFIED]

    # ---- loss weights, eq. (36)
    lambda_focal: float = 10.0         # lambda_f                    [Table II]
    lambda_contrastive: float = 0.2    # lambda_c                    [Table II]
    lambda_reg: float = 0.2            # lambda_r                    [Table II]

    # ---- focal loss, eq. (35)
    focal_balance: float = 0.25        # beta_1                      [Table II]
    focal_scaling: float = 2.0         # beta_2                      [Table II]
    soft_label_sigma: float = 1.0      # Gaussian soft-label width   [UNSPECIFIED]

    # ---- "random modality masking ... using predefined probabilities"
    # Per-modality probability of being dropped during training.
    modality_dropout: float = 0.15     #                             [UNSPECIFIED]

    # ---- evaluation
    topk: Tuple[int, ...] = (1, 3, 5)
    dba_delta: float = 5.0             # Delta in eq. (39)           [UNSPECIFIED]

    seed: int = 2022
    num_workers: int = 4
    device: str = "auto"               # auto | cpu | cuda | mps


def as_dict(model: ModelConfig, train: TrainConfig) -> dict:
    """Flat, JSON-serialisable record of a run's configuration."""
    return {"model": asdict(model), "train": asdict(train),
            "modalities": list(MODALITIES)}
