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


def test_unavailable_modality_yields_zeros_without_touching_disk(tmp_path):
    """A modality marked unavailable must not be read from disk.

    Regression test. Scenario 34 ships no radar for ~60% of its samples, and the
    loader used to call np.load unconditionally, so adding that scenario to
    training crashed with FileNotFoundError on the first such batch. The eq. (3)
    mask already tells the model to ignore the modality; the loader just has to
    return a correctly shaped tensor so the batch still collates.
    """
    import json

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from amber.config import ModelConfig
    from amber.dataset import AmberDataset

    cfg = ModelConfig(pretrained=False)
    root = tmp_path
    index = root / "data" / "processed_amber" / "index"
    index.mkdir(parents=True)

    # one sample with every modality present, one missing radar entirely
    made = {}
    for scn, has_radar in (("scenarioA", True), ("scenarioB", False)):
        base = root / "data" / "processed_amber" / "dev" / scn
        for mod in ("camera", "lidar_bev", "radar_ra_rv"):
            (base / mod).mkdir(parents=True)
        for i in range(1, cfg.window + 1):
            Image.new("RGB", (32, 32)).save(base / "camera" / f"img_{i}.jpg")
            np.save(base / "lidar_bev" / f"l_{i}.npy", np.zeros((1, 32, 32), np.float32))
            if has_radar:
                np.save(base / "radar_ra_rv" / f"r_{i}.npy",
                        np.ones((2, 32, 32), np.float32))
        made[scn] = base

    rows = []
    for scn, has_radar in (("scenarioA", 1), ("scenarioB", 0)):
        rel = f"data/processed_amber/dev/{scn}"
        row = {"sample_id": scn, "source": "dev", "scenario": scn, "frame": 1,
               "frame_target": 5, "split": "train", "beam": 3, "beam_official": 3,
               "pwr_row": len(rows), "pwr_n_nan": 0, "bs_lat": 0.0, "bs_lon": 0.0,
               "m_image": 1, "m_lidar": 1, "m_radar": has_radar,
               "m_beam": 1, "m_gps": 1, "n_beam_hist_available": 4}
        for i in range(1, cfg.window + 1):
            row[f"image_{i}"] = f"{rel}/camera/img_{i}.jpg"
            row[f"lidar_bev_{i}"] = f"{rel}/lidar_bev/l_{i}.npy"
            row[f"radar_{i}"] = f"{rel}/radar_ra_rv/r_{i}.npy"
        for k in (1, 2):
            row[f"ue_x_{k}"] = row[f"ue_y_{k}"] = 0.0
        for i in range(1, cfg.window):
            row[f"beam_hist_{i}"] = i
        rows.append(row)
    pd.DataFrame(rows).to_csv(index / "samples.csv", index=False)
    (index / "gps_norm.json").write_text(json.dumps({"columns": {
        c: {"min": 0.0, "max": 1.0} for c in
        ("ue_x_1", "ue_y_1", "ue_x_2", "ue_y_2")}}))

    ds = AmberDataset(index, "train", cfg)
    present, absent = ds[0], ds[1]

    # the available sample reads real data; the unavailable one is zeros
    assert present["radar"].abs().sum() > 0
    assert absent["radar"].abs().sum() == 0
    # and both share a shape, so a mixed batch collates
    assert present["radar"].shape == absent["radar"].shape
    assert absent["availability"].tolist()[2] == 0.0
    batch = torch.utils.data.default_collate([present, absent])
    assert batch["radar"].shape[0] == 2


def test_disabled_modality_has_zero_influence_and_skipping_is_equivalent():
    """The scientific foundation of the modality-ablation experiment.

    Two properties must hold for a configuration expressed by forcing the eq. (3)
    availability vector to be a faithful ablation rather than an approximation:

    1. a modality marked unavailable must have EXACTLY zero influence on the
       logits -- otherwise 'GPS + Radar' is silently still seeing the camera;
    2. skipping that modality's encoder must change nothing -- otherwise the
       latency and FLOPs reported per configuration describe a different model
       from the one that produced the accuracy.
    """
    from amber.ablation import enabled_mask, restrict_availability
    from amber.model import AMBER

    B = 3
    base = {"image": torch.randn(B, 5, 3, 64, 64),
            "lidar": torch.randn(B, 5, 1, 64, 64),
            "radar": torch.randn(B, 5, 2, 64, 64),
            "beam": torch.randn(B, 4, 2),
            "gps": torch.randn(B, 2, 2),
            "availability": torch.ones(B, len(MODALITIES))}
    batch = restrict_availability(base, enabled_mask(["gps", "radar"]))

    # the mask removed exactly the other three modalities
    assert batch["availability"][0].tolist() == [
        1.0 if m in ("gps", "radar") else 0.0 for m in MODALITIES]

    torch.manual_seed(0)
    cfg = ModelConfig(pretrained=False, pool_hw=(2, 2))
    model = AMBER(cfg).eval()
    with torch.no_grad():
        ref = model(batch, use_cma=False).logits
        # perturbing the disabled modalities must not move the output at all
        loud = {**batch, "image": torch.randn_like(batch["image"]) * 50,
                "lidar": torch.randn_like(batch["lidar"]) * 50,
                "beam": torch.randn_like(batch["beam"]) * 50}
        assert torch.equal(ref, model(loud, use_cma=False).logits)
        # while perturbing an enabled one must
        louder = {**batch, "radar": torch.randn_like(batch["radar"]) * 50}
        assert not torch.allclose(ref, model(louder, use_cma=False).logits)

    # encoder skipping is numerically equivalent
    torch.manual_seed(0)
    skipped = AMBER(ModelConfig(pretrained=False, pool_hw=(2, 2),
                                skip_unavailable_encoders=True)).eval()
    with torch.no_grad():
        assert torch.allclose(ref, skipped(batch, use_cma=False).logits, atol=1e-6)


def test_restriction_cannot_conjure_an_unavailable_modality():
    """A configuration may only remove modalities, never add them.

    Scenario 34 genuinely has no radar for ~60% of its samples and the official
    test split has no beam history at all, so 'GPS + Radar' must not pretend
    radar exists where it does not -- that would silently feed the model zeros
    while telling it the modality is present.
    """
    from amber.ablation import enabled_mask, restrict_availability

    B = 2
    natural = torch.zeros(B, len(MODALITIES))
    natural[:, MODALITIES.index("gps")] = 1.0          # only GPS really present
    out = restrict_availability({"availability": natural},
                                enabled_mask(["gps", "radar", "image"]))
    assert out["availability"][:, MODALITIES.index("radar")].sum() == 0
    assert out["availability"][:, MODALITIES.index("image")].sum() == 0
    assert out["availability"][:, MODALITIES.index("gps")].sum() == B
