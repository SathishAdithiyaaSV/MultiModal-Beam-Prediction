"""Invariants of the AMBER implementation, checked against the paper.

Run:  pytest -q                      (skips the slow optimisation test)
      pytest -q -m slow              (only the slow test)
"""
import numpy as np
import pytest
import torch

from amber.config import MODALITIES, ModelConfig, TrainConfig
from amber.cma import CMAModule
from amber.embeddings import ModalityWeightIndicator, sinusoidal_1d, sinusoidal_2d
from amber.losses import AmberLoss, focal_loss, gaussian_soft_labels
from amber.metrics import dba_score, topk_accuracy
from amber.model import AMBER
from amber.transformer import (TokenLayout, availability_mask, block_diagonal_mask)


@pytest.fixture(scope="module")
def cfg():
    # small backbones-free config so the suite stays fast
    return ModelConfig(pretrained=False, pool_hw=(2, 2))


@pytest.fixture(scope="module")
def layout(cfg):
    return TokenLayout(MODALITIES, {"image": 4, "lidar": 4, "radar": 4, "beam": 4, "gps": 2})


def dummy_batch(cfg, batch=2, size=64, availability=None):
    n = len(MODALITIES)
    return {
        "image": torch.randn(batch, cfg.window, 3, size, size),
        "lidar": torch.rand(batch, cfg.window, 1, size, size),
        "radar": torch.rand(batch, cfg.window, 2, size, size),
        "beam": torch.rand(batch, cfg.window - 1, 2),
        "gps": torch.rand(batch, 2, 2),
        "availability": torch.ones(batch, n) if availability is None else availability,
    }


# --------------------------------------------------------------- token geometry
def test_token_count_matches_paper_N():
    """N = 2(3*VA*HA + W + 1), AMBER eq. (28) footnote."""
    cfg = ModelConfig(pretrained=False)
    total = 3 * cfg.n_grid_tokens + cfg.n_beam_tokens + cfg.n_gps_tokens
    assert 2 * total == 2 * (3 * cfg.n_grid_tokens + cfg.window + 1)


def test_layout_spans_are_contiguous_and_complete(layout):
    covered = []
    for m in layout.order:
        lo, hi = layout.span(m)
        covered.extend(range(lo, hi))
    assert covered == list(range(layout.total))


# --------------------------------------------------------------- masks
def test_block_diagonal_mask_is_within_modality_only(layout):
    """eq. (23): M[j,i] = 1 iff i == j."""
    mask = block_diagonal_mask(layout)
    assert mask.shape == (layout.total, layout.total)
    for m in layout.order:
        lo, hi = layout.span(m)
        assert mask[lo:hi, lo:hi].all()                      # sees its own modality
        assert mask[lo:hi].sum().item() == (hi - lo) ** 2    # and nothing else


def test_availability_mask_drops_exactly_the_missing_modalities(layout):
    """eq. (30) key mask."""
    av = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1]], dtype=torch.float32)
    mask = availability_mask(layout, av)
    assert mask[0].sum().item() == layout.total
    dropped = layout.counts["lidar"] + layout.counts["beam"]
    assert mask[1].sum().item() == layout.total - dropped


# --------------------------------------------------------------- embeddings
def test_sinusoidal_positions_are_distinct():
    assert len(torch.unique(sinusoidal_1d(16, 64), dim=0)) == 16
    assert len(torch.unique(sinusoidal_2d(4, 4, 64), dim=0)) == 16


def test_modality_weights_form_a_distribution():
    """eq. (18): alpha >= 0 and sums to 1."""
    ind = ModalityWeightIndicator(MODALITIES, 1.0)
    a = ind.alpha()
    assert torch.allclose(a.sum(), torch.tensor(1.0))
    assert (a >= 0).all()


def test_l2_penalty_ignores_missing_modalities():
    """eq. (20): L2 = sum_i m_i w_i^2."""
    ind = ModalityWeightIndicator(MODALITIES, 1.0)
    with torch.no_grad():
        ind.weights.copy_(torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]))
    full = ind.l2_penalty(torch.ones(1, 5))
    without_lidar = ind.l2_penalty(torch.tensor([[1.0, 0, 1, 1, 1]]))
    assert pytest.approx(full.item(), abs=1e-6) == 5.0       # 1 + 4
    assert pytest.approx(without_lidar.item(), abs=1e-6) == 1.0


# --------------------------------------------------------------- losses
def test_soft_labels_peak_at_the_true_beam_and_decay():
    s = gaussian_soft_labels(torch.tensor([30]), 64, 1.0)
    assert s.argmax(1).item() == 30
    assert s[0, 30] == pytest.approx(1.0)
    # strictly decreasing as we move away
    assert s[0, 31] > s[0, 32] > s[0, 33]


def test_focal_loss_orders_correct_then_near_then_far():
    """A near miss must cost less than a distant miss -- the point of eq. (35)."""
    soft = gaussian_soft_labels(torch.tensor([30]), 64, 1.0)
    def logits_for(beam):
        z = torch.full((1, 64), -6.0)
        z[0, beam] = 6.0
        return z
    correct = focal_loss(logits_for(30), soft)
    near = focal_loss(logits_for(31), soft)
    far = focal_loss(logits_for(5), soft)
    assert correct < near < far


def test_total_loss_weights_each_term(cfg):
    tc = TrainConfig()
    crit = AmberLoss(64, tc.lambda_focal, tc.lambda_contrastive, tc.lambda_reg,
                     tc.focal_balance, tc.focal_scaling, tc.soft_label_sigma)
    logits = torch.zeros(2, 64)
    total, parts = crit(logits, torch.tensor([1, 2]),
                        torch.tensor(1.0), torch.tensor(2.0))
    expected = (tc.lambda_focal * parts["focal"]
                + tc.lambda_contrastive * 1.0 + tc.lambda_reg * 2.0)
    assert parts["loss"] == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------- metrics
def test_topk_accuracy_is_monotone_in_k():
    torch.manual_seed(0)
    logits, targets = torch.randn(200, 64), torch.randint(0, 64, (200,))
    acc = topk_accuracy(logits, targets, (1, 3, 5))
    assert acc[1] <= acc[3] <= acc[5]


def test_dba_matches_the_closed_form():
    """eq. (39): an off-by-one prediction scores 1 - 1/Delta."""
    targets = torch.tensor([10, 20, 30])
    logits = torch.full((3, 64), -9.0)
    for i, t in enumerate(targets):
        logits[i, t + 1] = 9.0
    score, _ = dba_score(logits, targets, k=1, delta=5.0)
    assert score == pytest.approx(0.8)


def test_dba_is_zero_beyond_delta():
    targets = torch.tensor([10])
    logits = torch.full((1, 64), -9.0)
    logits[0, 40] = 9.0
    assert dba_score(logits, targets, k=1, delta=5.0)[0] == pytest.approx(0.0)


def test_dba_y_curve_is_non_decreasing():
    torch.manual_seed(0)
    _, ys = dba_score(torch.randn(300, 64), torch.randint(0, 64, (300,)), k=5)
    assert all(a <= b + 1e-9 for a, b in zip(ys, ys[1:]))


# --------------------------------------------------------------- model
def test_forward_shapes(cfg):
    model = AMBER(cfg)
    out = model(dummy_batch(cfg))
    assert out.logits.shape == (2, cfg.n_beams)
    assert out.alpha.shape == (len(MODALITIES),)


def test_every_missing_modality_pattern_stays_finite(cfg):
    """AMBER must handle arbitrary missing-modality cases, including none at all."""
    model = AMBER(cfg).eval()
    n = len(MODALITIES)
    for bits in range(1 << n):
        av = torch.tensor([[(bits >> i) & 1 for i in range(n)]], dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_batch(cfg, batch=1, availability=av))
        assert torch.isfinite(out.logits).all(), f"non-finite for availability {av.tolist()}"


def test_missing_modality_changes_the_prediction(cfg):
    """The eq. (30) mask must actually take effect, not be silently ignored."""
    torch.manual_seed(0)
    model = AMBER(cfg).eval()
    batch = dummy_batch(cfg, batch=1)
    with torch.no_grad():
        full = model(batch).logits
        partial = model({**batch, "availability": torch.tensor([[1.0, 0, 0, 0, 1]])}).logits
    assert not torch.allclose(full, partial)


def test_cma_runs_in_training_and_is_skipped_in_eval(cfg):
    model = AMBER(cfg)
    batch = dummy_batch(cfg, batch=4)
    model.train()
    assert model(batch).contrastive.item() > 0
    model.eval()
    with torch.no_grad():
        assert model(batch).contrastive.item() == 0.0


def test_contrastive_loss_rewards_alignment(cfg, layout):
    cma = CMAModule(cfg, layout)
    fusion = torch.randn(4, cfg.embed_dim)
    av = torch.ones(4, len(MODALITIES))
    aligned = cma.contrastive_loss({m: fusion.clone() for m in MODALITIES}, fusion, av)
    random = cma.contrastive_loss(
        {m: torch.randn(4, cfg.embed_dim) for m in MODALITIES}, fusion, av)
    assert aligned < random


@pytest.mark.parametrize("temporal_pool", ["concat", "mean", "tokens"])
def test_all_temporal_pool_modes_work(temporal_pool):
    cfg = ModelConfig(pretrained=False, pool_hw=(2, 2), temporal_pool=temporal_pool)
    model = AMBER(cfg).eval()
    with torch.no_grad():
        out = model(dummy_batch(cfg, batch=2))
    assert out.logits.shape == (2, cfg.n_beams)


def test_gradients_reach_every_trainable_parameter(cfg):
    model = AMBER(cfg)
    model.train()
    out = model(dummy_batch(cfg, batch=2))
    tc = TrainConfig()
    crit = AmberLoss(cfg.n_beams, tc.lambda_focal, tc.lambda_contrastive, tc.lambda_reg,
                     tc.focal_balance, tc.focal_scaling, tc.soft_label_sigma)
    loss, _ = crit(out.logits, torch.tensor([1, 2]), out.contrastive, out.reg)
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing[:10]}"


@pytest.mark.slow
def test_model_can_overfit_a_tiny_batch():
    """The end-to-end wiring check: if it cannot memorise 8 samples, something
    in the encoder -> mask -> fusion -> head chain is broken."""
    torch.manual_seed(0)
    cfg = ModelConfig(pretrained=False, pool_hw=(2, 2))
    model = AMBER(cfg)
    tc = TrainConfig()
    crit = AmberLoss(cfg.n_beams, tc.lambda_focal, tc.lambda_contrastive, tc.lambda_reg,
                     tc.focal_balance, tc.focal_scaling, tc.soft_label_sigma)
    batch = dummy_batch(cfg, batch=8)
    targets = torch.randint(0, cfg.n_beams, (8,))
    opt = torch.optim.AdamW(model.parameter_groups(0.05), lr=3e-4)
    model.train()
    for _ in range(120):
        out = model(batch)
        loss, _ = crit(out.logits, targets, out.contrastive, out.reg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = topk_accuracy(model(batch).logits, targets, (1,))[1]
    assert acc == 1.0, f"could only reach top1={acc}"
