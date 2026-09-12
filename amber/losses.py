"""Loss functions — AMBER eqs. (20), (34), (35), (36).

Total objective, eq. (36):
    L = lambda_f * L_focal + lambda_c * L_contrastive + lambda_r * L_2

The focal term, eq. (35), is written per output element:
    L_f = - k*  * beta_1       * (1 - k)^beta_2 * log(k)
          - (1-k*) * (1-beta_1) * k^beta_2       * log(1 - k)

so each of the 64 beams is a sigmoid, not a softmax over the codebook. `k*` is
a SOFT label: "instead of using conventional one-hot encoding, we propose to
utilize soft labels ... where the values near the ground truth decay according
to a Gaussian function across adjacent beam indices".

The Gaussian is the natural choice for this task: neighbouring beams in the
codebook point in neighbouring directions, so predicting beam 31 when the truth
is 30 should cost far less than predicting beam 5. This is also what the DBA
metric measures, so the loss and the metric agree.
"""
import torch
import torch.nn.functional as F

EPS = 1e-7


def gaussian_soft_labels(targets: torch.Tensor, n_beams: int, sigma: float,
                         circular: bool = False) -> torch.Tensor:
    """(B,) beam indices -> (B, n_beams) Gaussian-smoothed targets, peak 1.0.

    `circular` wraps the distance around the codebook. It is off by default:
    the DeepSense6G codebook sweeps a finite angular sector rather than a full
    circle, so beams 0 and 63 are not neighbours.
    """
    idx = torch.arange(n_beams, device=targets.device, dtype=torch.float32)
    dist = (idx.unsqueeze(0) - targets.unsqueeze(1).to(torch.float32)).abs()
    if circular:
        dist = torch.minimum(dist, n_beams - dist)
    return torch.exp(-dist.pow(2) / (2.0 * sigma ** 2))


def focal_loss(logits: torch.Tensor, soft_targets: torch.Tensor,
               balance: float = 0.25, scaling: float = 2.0) -> torch.Tensor:
    """eq. (35), summed over beams and averaged over the batch.

    logits (B, n_beams) raw scores; soft_targets (B, n_beams) in [0, 1].
    """
    p = torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)
    pos = soft_targets * balance * (1.0 - p).pow(scaling) * torch.log(p)
    neg = (1.0 - soft_targets) * (1.0 - balance) * p.pow(scaling) * torch.log(1.0 - p)
    return -(pos + neg).sum(dim=1).mean()


class AmberLoss:
    """Assembles eq. (36) and reports each term for logging."""

    def __init__(self, n_beams: int, lambda_focal: float, lambda_contrastive: float,
                 lambda_reg: float, focal_balance: float, focal_scaling: float,
                 soft_label_sigma: float):
        self.n_beams = n_beams
        self.lambda_focal = lambda_focal
        self.lambda_contrastive = lambda_contrastive
        self.lambda_reg = lambda_reg
        self.focal_balance = focal_balance
        self.focal_scaling = focal_scaling
        self.soft_label_sigma = soft_label_sigma

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor,
                 contrastive: torch.Tensor,
                 reg: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        soft = gaussian_soft_labels(targets, self.n_beams, self.soft_label_sigma)
        l_focal = focal_loss(logits, soft, self.focal_balance, self.focal_scaling)
        total = (self.lambda_focal * l_focal
                 + self.lambda_contrastive * contrastive
                 + self.lambda_reg * reg)
        parts = {"loss": total.detach().item(), "focal": l_focal.detach().item(),
                 "contrastive": contrastive.detach().item(),
                 "l2": reg.detach().item()}
        return total, parts
