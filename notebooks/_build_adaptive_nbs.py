"""Generate the three adaptive-routing notebooks.

  1. modality_robustness.ipynb      RQ1 degradation curves + RQ2 missing modality
  2. modality_quality_signal.ipynb  RQ3 which inference-time signals predict need
  3. learned_adaptive_gate.ipynb    RQ4 the gate, plus the final comparison

Merged from the five originally proposed: degradation and missing-modality are
mechanically the same experiment (perturb the input, evaluate a fixed
checkpoint), and the final comparison needs exactly what the gate notebook
already holds, so separating them would only add a Kaggle session.

Note: code cells must not contain triple double-quotes -- use ''' inside them.
"""
import json
import pathlib

REPO = "https://github.com/SathishAdithiyaaSV/MultiModal-Beam-Prediction.git"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").split("\n")}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.strip("\n").split("\n")}


def nb(cells, gpu=True):
    return {"cells": cells,
            "metadata": {"accelerator": "GPU" if gpu else "None",
                         "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none",
                                    "dataSources": [], "isInternetEnabled": True,
                                    "language": "python", "sourceType": "notebook"},
                         "kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


# ------------------------------------------------------------------ shared cells
ENV = code(r"""
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print(f"torch {torch.__version__}   cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")


def has_internet(host="github.com", port=443, timeout=6):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


assert has_internet(), "Internet is off; the code cannot be cloned."
""")

STAGE = code(r"""
import shutil

INPUT_ROOT = Path("/kaggle/input")
SOURCES = ("development", "adaptation", "test")


def walk_dirs(base, follow=True):
    for dirpath, dirnames, filenames in os.walk(base, followlinks=follow):
        yield Path(dirpath), dirnames, filenames


def find_source(base, name):
    for c in (base / name / name, base / name):
        if c.is_dir() and any(c.glob("scenario*")):
            return c
    return None


def find_index(base):
    hits = []
    for c in (base / "index" / "index", base / "index"):
        if (c / "samples.csv").exists():
            hits.append(c)
    for d, _dn, fn in walk_dirs(base):
        if "samples.csv" in fn and d not in hits:
            hits.append(d)
    if not hits:
        return None
    # prefer the index that sits beside the actual tensors, not a copy inside a
    # results dump
    score = lambda h: sum(1 for s in SOURCES
                          for r in (h.parent, h.parent.parent, h)
                          if find_source(r, s) is not None)
    return sorted(hits, key=score, reverse=True)[0]


INDEX_SRC = find_index(INPUT_ROOT)
assert INDEX_SRC is not None, "samples.csv not found; attach the preprocessed dataset"
ROOTS = [INDEX_SRC.parent, INDEX_SRC.parent.parent, INDEX_SRC]
SOURCE_DIRS = {s: next((d for d in (find_source(r, s) for r in ROOTS) if d), None)
               for s in SOURCES}
for s in [k for k, v in SOURCE_DIRS.items() if v is None]:
    for d, dn, _f in walk_dirs(INPUT_ROOT):
        if d.name == s and any(x.startswith("scenario") for x in dn):
            SOURCE_DIRS[s] = d
            break

# stage idempotently: /kaggle/working survives between cell runs
STAGE_DIR = Path("/kaggle/working/data/processed_amber")
STAGE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = STAGE_DIR / "index"
staged, src_csv = INDEX_DIR / "samples.csv", INDEX_SRC / "samples.csv"
if not staged.exists():
    shutil.copytree(INDEX_SRC, INDEX_DIR, dirs_exist_ok=True)
elif staged.stat().st_size != src_csv.stat().st_size:
    shutil.rmtree(INDEX_DIR)
    shutil.copytree(INDEX_SRC, INDEX_DIR, dirs_exist_ok=True)
for name, src in SOURCE_DIRS.items():
    if src is None:
        continue
    link = STAGE_DIR / name
    if link.is_symlink():
        try:
            same = link.resolve(strict=True) == src.resolve(strict=True)
        except OSError:
            same = False
        if not same:
            link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    if not link.exists():
        link.symlink_to(src)
print(f"index {INDEX_DIR}")
""")

CODE_CELL = code(rf"""
PROJECT = Path("/kaggle/working/project")
if not (PROJECT / "amber" / "model.py").exists():
    r = subprocess.run(["git", "clone", "--depth", "1", "{REPO}", str(PROJECT)],
                       capture_output=True, text=True,
                       env={{**os.environ, "GIT_TERMINAL_PROMPT": "0"}})
    assert r.returncode == 0, (r.stderr or "")[:400]
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from amber import degrade, quality
from amber.ablation import config_label, enabled_mask, parse_modalities, restrict_availability
from amber.config import MODALITIES, TrainConfig
from amber.dataset import AmberDataset
from amber.degrade import Degradation
from amber.metrics import evaluate
from amber.predict import load_checkpoint
from amber.train import make_loader, pick_device, to_device

GFLOPS = {{"gps": 0.3869, "gps_radar": 23.5668, "gps_lidar": 23.0530,
           "gps_radar_lidar": 46.2328, "gps_image": 48.2398,
           "gps_image_lidar": 70.9058, "gps_radar_image": 71.4196,
           "gps_radar_image_lidar": 94.0856}}


def find_checkpoints():
    found = {{}}
    for d, _dn, fn in walk_dirs(INPUT_ROOT):
        if "best.pt" in fn and d.name in GFLOPS:
            found.setdefault(d.name, d / "best.pt")
        for f in fn:
            if f.endswith(".pt") and f[:-3] in GFLOPS:
                found.setdefault(f[:-3], d / f)
    return found


CKPTS = find_checkpoints()
print(f"{{len(CKPTS)}}/8 checkpoints: {{', '.join(sorted(CKPTS))}}")
device = pick_device("auto")
train_cfg = TrainConfig()
""")


def eval_helper(extra=""):
    return code(r"""
import gc
import json
import time

_MODEL_CACHE = {}


def get_model(slug):
    '''Load once and keep -- several sweeps reuse the same checkpoint.'''
    if slug not in _MODEL_CACHE:
        if len(_MODEL_CACHE) >= 2:          # keep GPU memory bounded
            _MODEL_CACHE.clear()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _MODEL_CACHE[slug] = load_checkpoint(CKPTS[slug], device)
    return _MODEL_CACHE[slug]


@torch.no_grad()
def run_eval(slug, split, degradations=(), mask_out=(), limit=None,
             return_per_sample=False, collect_features=False):
    '''Evaluate one checkpoint, optionally with degraded or masked inputs.

    `mask_out` zeroes availability bits at inference -- the eq. (3) mechanism,
    not a retrained model. `degradations` perturb the input tensors.
    '''
    model, cfg = get_model(slug)
    mods = parse_modalities(slug.split("_"))
    # Sensor-quality features must be computed from the *real* tensors. The
    # dataset returns zeros for any modality outside `modalities`, so a cheap
    # GPS-only configuration would otherwise yield identically-zero image,
    # LiDAR and radar statistics -- which silently collapses feature group C
    # onto group B. Load every modality when collecting them; the model is
    # still restricted through `keep` below, and availability masking is
    # numerically equivalent to not loading (verified: delta logits = 0.0), so
    # the logits are unchanged.
    ds_mods = (tuple(MODALITIES) if (collect_features and COLLECT_SENSOR_QUALITY)
               else mods)
    ds = AmberDataset(INDEX_DIR, split, cfg, modalities=ds_mods,
                      degradations=list(degradations))
    if limit:
        from torch.utils.data import Subset
        idx = np.linspace(0, len(ds) - 1, min(limit, len(ds))).astype(int)
        scen = [ds.scenarios[i] for i in idx]
        sids = [ds.sample_ids[i] for i in idx]
        ds = Subset(ds, idx.tolist())
    else:
        scen, sids = ds.scenarios, ds.sample_ids
    loader = make_loader(ds, BATCH_SIZE, shuffle=False, workers=WORKERS)

    keep = enabled_mask([m for m in mods if m not in mask_out], device) if mask_out \
        else enabled_mask(mods, device)
    logits, targets, feats = [], [], []
    for batch in loader:
        raw = to_device(batch, device)
        batch = restrict_availability(raw, keep)
        out = model(batch, use_cma=False)
        logits.append(out.logits.float().cpu())
        targets.append(batch["target"].cpu())
        if collect_features:
            # from `raw`, not `batch`: avail_* should report which sensors
            # actually reported, not which this configuration was permitted to
            # consume. The latter is constant per configuration and so carries
            # no information for the gate.
            feats += quality.batch_features(
                {k: v.cpu() for k, v in raw.items() if torch.is_tensor(v)},
                out.logits.float().cpu(), MODALITIES, COLLECT_SENSOR_QUALITY)
    logits, targets = torch.cat(logits), torch.cat(targets)
    lab = targets >= 0
    m = evaluate(logits[lab], targets[lab], ks=train_cfg.topk,
                 delta=train_cfg.dba_delta)
    if not (return_per_sample or collect_features):
        return m
    ranked = logits.topk(3, dim=1).indices.numpy()
    true = targets.numpy()[:, None]
    best = np.minimum.accumulate(np.abs(ranked - true) / train_cfg.dba_delta,
                                 axis=1).clip(max=1.0)
    per = pd.DataFrame({"sample_id": sids, "scenario": scen,
                        "true_beam": targets.numpy(),
                        "pred_beam": ranked[:, 0],
                        "correct1": (ranked[:, 0] == true[:, 0]).astype(int),
                        "correct3": (ranked[:, :3] == true).any(axis=1).astype(int),
                        "dba": (1.0 - best).mean(axis=1)})
    if collect_features:
        per = pd.concat([per, pd.DataFrame(feats)], axis=1)
    return m, per
""" + extra)


# ------------------------------------------------------------------ notebook 1
def build_robustness():
    C = []
    C.append(md(r"""
# Modality robustness — degradation and missing modalities

**RQ1** How does beam prediction degrade as each modality becomes unreliable?
**RQ2** Can AMBER cope when a modality is missing at inference?

Both evaluate **existing checkpoints under modified inputs** — no retraining.
That is the right experiment for these questions: they ask how the *deployed*
predictor behaves, not whether a model could learn to compensate. Retraining at
every severity would cost tens of GPU-hours and answer a different question.

---

## Design: which degradations are admissible, and which are not

Only the preprocessed tensors exist at experiment time, so a degradation counts
only if applying it *there* corresponds to a real sensor failure. Inspecting the
representations rejected several of the obvious candidates.

| Modality | Applied | Represents | Why it is admissible |
|---|---|---|---|
| **Camera** | Gaussian blur | defocus, motion, dirty lens | not affine, survives eq. (10) |
| | additive noise | low-light sensor noise — scenarios 33/34 are night | not affine |
| | occlusion (block) | obstruction, raindrop, blocked FOV | not affine |
| | resolution loss | cheaper sensor, heavy compression | not affine |
| **LiDAR** | point dropout | sparse returns, rain, low reflectivity | BEV value is count/5, so binomial thinning of the counts is *exactly* "each return independently lost" |
| **GPS** | positional noise (metres) | consumer GNSS is 1–5 m; this dataset is far more precise | converted through `gps_norm.json`, so severity is physical |

### Rejected, with reasons

| Rejected | Why |
|---|---|
| **Camera brightness / contrast** | **Provably a no-op.** Eq. (10) standardises each image by its own mean and std, which is affine-invariant: `a·x + b` → `(a·x + b − a·mean − b)/(a·std)` = the original. Verified to 4e-7. Any lighting experiment here measures nothing. |
| **Radar noise / map corruption** | The stored tensor is a post-2D-FFT magnitude pair under one joint min-max. Noise in the map domain has no pre-image in the IQ cube, where real noise spreads across the whole map. A curve would measure an artefact. **Radar unreliability is studied through the dataset's own missingness instead** — scenario 34 supplies radar for only ~40 % of its samples. |
| **LiDAR range truncation** | The BEV is already cropped to the ±50 m ROI holding 99 % of points. |
| **Fewer GPS observations** | Only two exist; dropping one is the missing-modality experiment. |

### What each curve is for

Camera is the priority: the ablation showed it is the **only expensive modality
with substantial incremental value** (+0.104 DBA against LiDAR's +0.023), so its
reliability decides whether escalating to it is ever safe.

GPS matters just as much for a different reason: it is the **cheap tier**. The
whole revised framing rests on GPS alone reaching 83 % of the full model's DBA.
If that collapses at realistic localisation error, the framing is fragile. This
is arguably the single most load-bearing experiment here.

LiDAR gets one short curve, not for completeness but because **point density is
a candidate gate feature** — we need to know whether it tracks usefulness.
"""))
    C.append(md("## 1. Configuration"))
    C.append(code(r"""
QUICK_RUN = True        # True: ~200 samples for debugging. False: the real run.
SPLIT = "val"
BATCH_SIZE = 32
WORKERS = 2
COLLECT_SENSOR_QUALITY = False

LIMIT = 200 if QUICK_RUN else None
from pathlib import Path
OUT = Path("/kaggle/working/outputs/modality_degradation")
(OUT / "plots").mkdir(parents=True, exist_ok=True)
MISS = Path("/kaggle/working/outputs/missing_modality")
MISS.mkdir(parents=True, exist_ok=True)
print(f"QUICK_RUN={QUICK_RUN}  limit={LIMIT}  out={OUT}")
"""))
    C.append(md("## 2. Environment, data, code"))
    C.append(ENV)
    C.append(STAGE)
    C.append(CODE_CELL)
    C.append(eval_helper())
    C.append(md(r"""
## 3. RQ1 — degradation curves

Each sweep evaluates one checkpoint whose performance depends on the degraded
modality. GPS is swept on **two** configurations, because whether camera can
rescue a degraded cheap tier is itself a routing question.

Results are appended to disk after every point, so an interrupted session keeps
its progress.
"""))
    C.append(code(r"""
SWEEPS = [
    # (slug, modality, kind)  -- severity grids come from amber.degrade
    ("gps_image", "image", "blur"),
    ("gps_image", "image", "noise"),
    ("gps_image", "image", "occlusion"),
    ("gps_image", "image", "resolution"),
    ("gps",       "gps",   "position_noise"),
    ("gps_image", "gps",   "position_noise"),
    ("gps_lidar", "lidar", "dropout"),
]
SWEEPS = [s for s in SWEEPS if s[0] in CKPTS]

rows_path = OUT / f"degradation_{SPLIT}{'_quick' if QUICK_RUN else ''}.csv"
done = set()
if rows_path.exists():
    prev = pd.read_csv(rows_path)
    done = set(zip(prev.slug, prev.modality, prev.kind, prev.severity))
    rows = prev.to_dict("records")
    print(f"resuming: {len(rows)} points already measured")
else:
    rows = []

for slug, modality, kind in SWEEPS:
    for sev in degrade.grid_for(kind):
        key = (slug, modality, kind, sev)
        if key in done:
            continue
        degs = [] if sev == degrade.grid_for(kind)[0] else [Degradation(modality, kind, sev)]
        t0 = time.time()
        m = run_eval(slug, SPLIT, degradations=degs, limit=LIMIT)
        rows.append({"slug": slug, "config": config_label(parse_modalities(slug.split("_"))),
                     "modality": modality, "kind": kind, "severity": sev,
                     "Top-1": m["top1"], "Top-3": m["top3"], "Top-5": m["top5"],
                     "DBA": m["dba"], "n": m["n"], "seconds": round(time.time() - t0, 1)})
        pd.DataFrame(rows).to_csv(rows_path, index=False)
        print(f"  {slug:<12} {kind:<15} sev={sev:<7g} "
              f"Top-1 {m['top1']:.4f}  DBA {m['dba']:.4f}  [{rows[-1]['seconds']}s]")

deg = pd.DataFrame(rows)
print(f"\n{len(deg)} points -> {rows_path}")
"""))
    C.append(md("### Degradation table and where each modality collapses"))
    C.append(code(r"""
for (slug, kind), g in deg.groupby(["slug", "kind"], sort=False):
    g = g.sort_values("severity", key=lambda s: s.abs() if kind != "dropout" else -s)
    base = g.iloc[0]
    print(f"\n{g.iloc[0]['config']}  --  {kind}")
    print(f"{'severity':>10}{'Top-1':>9}{'DBA':>9}{'ΔDBA':>9}{'% of intact DBA':>18}")
    for _, r in g.iterrows():
        print(f"{r.severity:>10g}{r['Top-1']:>9.4f}{r.DBA:>9.4f}"
              f"{r.DBA - base.DBA:>+9.4f}{r.DBA / base.DBA * 100:>17.1f}%")

# where does each sweep lose a quarter of its intact DBA?
print("\n\ntipping points -- first severity costing >25% of intact DBA:")
for (slug, kind), g in deg.groupby(["slug", "kind"], sort=False):
    g = g.sort_values("severity", key=lambda s: -s if kind == "dropout" else s)
    base = g.iloc[0].DBA
    hit = g[g.DBA < 0.75 * base]
    where = f"{hit.iloc[0].severity:g}" if len(hit) else "never in range"
    print(f"  {g.iloc[0]['config']:<24} {kind:<15} -> {where}")
"""))
    C.append(md(r"""
## 4. RQ2 — missing modalities

Two different questions, separated.

**(a) Does AMBER's eq. (3) mask actually work?** Take the model trained on all
four modalities, mask one out at inference, and compare against the model
*independently trained* without it. If masking is far worse, the
missing-modality mechanism the paper is built on is not delivering — a direct
test of its central claim.

**(b) Natural missingness.** Radar is genuinely absent for ~60 % of scenario-34
samples. Those rows are reported separately from synthetically masked ones, so
the two are never conflated.

Subsets are chosen, not enumerated: each single modality, plus the three pairs
that leave a usable configuration. Dropping GPS is included because it is the
cheap tier.
"""))
    C.append(code(r"""
FULL = "gps_radar_image_lidar"
SUBSETS = [(), ("image",), ("lidar",), ("radar",), ("gps",),
           ("image", "radar"), ("image", "lidar"), ("radar", "lidar")]
miss_rows = []
if FULL in CKPTS:
    for drop in SUBSETS:
        remaining = [m for m in ("gps", "radar", "image", "lidar") if m not in drop]
        m = run_eval(FULL, SPLIT, mask_out=drop, limit=LIMIT)
        miss_rows.append({
            "dropped": "none" if not drop else "+".join(drop),
            "remaining": "+".join(remaining),
            "source": "masked full model",
            "Top-1": m["top1"], "Top-3": m["top3"], "Top-5": m["top5"], "DBA": m["dba"],
            "GFLOPs": GFLOPS[FULL],   # masking does not save compute: see note below
            "n": m["n"]})
        print(f"  drop {str(drop):<22} Top-1 {m['top1']:.4f}  DBA {m['dba']:.4f}")
miss = pd.DataFrame(miss_rows)
miss.to_csv(MISS / f"masked_full_model_{SPLIT}.csv", index=False)
print(f"\n-> {MISS}/masked_full_model_{SPLIT}.csv")
print("\nNote: masking the full model does NOT reduce its cost -- the encoders still")
print("run unless skip_unavailable_encoders is set. Cost savings come from using a")
print("smaller configuration, which is what the ablation measured.")
"""))
    C.append(md("### Masking vs independent training — does the mask mechanism hold up?"))
    C.append(code(r"""
# the ablation trained a model for each subset; compare like with like
ABL = {"gps": "GPS", "gps_radar": "GPS + Radar", "gps_lidar": "GPS + LiDAR",
       "gps_image": "GPS + Camera", "gps_radar_lidar": "GPS + Radar + LiDAR",
       "gps_radar_image": "GPS + Radar + Camera", "gps_image_lidar": "GPS + Camera + LiDAR",
       "gps_radar_image_lidar": "GPS + Radar + Camera + LiDAR"}
cmp_rows = []
for _, r in miss.iterrows():
    want = set(r.remaining.split("+"))
    twin = next((s for s in ABL if set(s.split("_")) == want and s in CKPTS), None)
    if twin is None:
        continue
    t = run_eval(twin, SPLIT, limit=LIMIT)
    cmp_rows.append({"modalities": r.remaining,
                     "masked full model DBA": r.DBA,
                     "independently trained DBA": t["dba"],
                     "gap": r.DBA - t["dba"],
                     "masked Top-1": r["Top-1"], "trained Top-1": t["top1"]})
cmp = pd.DataFrame(cmp_rows).sort_values("gap")
cmp.to_csv(MISS / f"masking_vs_training_{SPLIT}.csv", index=False)
print(cmp.round(4).to_string(index=False))
if len(cmp):
    print(f"\nmean gap {cmp.gap.mean():+.4f} DBA")
    print("Negative means masking the full model is WORSE than training without the")
    print("modality -- i.e. the mask leaves the model depending on something absent.")
    print("Near zero means the eq. (3) mechanism genuinely handles the absence.")
"""))
    C.append(md("### Natural vs synthetic missingness"))
    C.append(code(r"""
samples = pd.read_csv(INDEX_DIR / "samples.csv")
v = samples[samples.split == SPLIT]
nat = v.groupby("scenario")[["m_image", "m_lidar", "m_radar", "m_gps"]].mean().round(3)
print("naturally available fraction per scenario on this split:")
print(nat.to_string())
print("\nRadar is naturally absent for a large share of scenario 34. Splitting the")
print("best configuration's per-sample results on that real mask separates")
print("'sensor genuinely absent' from 'we removed it', which the table above mixes.")

best = "gps_radar_image_lidar" if FULL in CKPTS else sorted(CKPTS)[0]
_m, per = run_eval(best, SPLIT, limit=LIMIT, return_per_sample=True)
per = per.merge(v[["sample_id", "m_radar"]], on="sample_id", how="left")
nat_rows = [{"radar_naturally_present": bool(k), "n": len(g),
             "Top-1": g.correct1.mean(), "DBA": g.dba.mean()}
            for k, g in per.groupby("m_radar")]
nat_df = pd.DataFrame(nat_rows)
nat_df.to_csv(MISS / f"natural_missingness_{SPLIT}.csv", index=False)
print()
print(nat_df.round(4).to_string(index=False))
"""))
    C.append(md("## 5. Plots"))
    C.append(code(r"""
import matplotlib.pyplot as plt

groups = list(deg.groupby(["slug", "kind"], sort=False))
ncol = min(4, len(groups))
nrow = (len(groups) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
for ax, ((slug, kind), g) in zip(axes.flat, groups):
    g = g.sort_values("severity")
    x = g.severity if kind != "dropout" else 1 - g.severity
    ax.plot(x, g.DBA, marker="o", label="DBA")
    ax.plot(x, g["Top-1"], marker="s", ls="--", label="Top-1")
    ax.set(title=f"{g.iloc[0]['config']}\n{kind}",
           xlabel="severity" if kind != "dropout" else "fraction of points lost",
           ylim=(0, 1))
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=8)
for ax in axes.flat[len(groups):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "plots" / "degradation_curves.png", dpi=150)
plt.show()
print(f"-> {OUT}/plots/degradation_curves.png")
"""))
    C.append(md("## 6. Save"))
    C.append(code(r"""
meta = {"quick_run": QUICK_RUN, "split": SPLIT, "limit": LIMIT,
        "sweeps": [{"slug": s, "modality": m, "kind": k,
                    "grid": degrade.grid_for(k)} for s, m, k in SWEEPS],
        "rejected_degradations": degrade.REJECTED,
        "missing_subsets": [list(s) for s in SUBSETS],
        "checkpoints_used": sorted(CKPTS)}
(OUT / "config.json").write_text(json.dumps(meta, indent=2))
for p in sorted(list(OUT.rglob("*")) + list(MISS.rglob("*"))):
    if p.is_file():
        print(f"  {p}  ({p.stat().st_size/1e3:.1f} KB)")
"""))
    return nb(C)


# ------------------------------------------------------------------ notebook 2
def build_quality_signal():
    C = []
    C.append(md(r"""
# Modality quality as a routing signal

**RQ3** Can information available *at inference time* tell us whether an
expensive modality is worth using?

The oracle-routing analysis established the stakes: perfect routing reaches
Top-1 0.5496 at 11.36 GFLOPs against always-GPS+Camera's 0.4604 at 48.24, while
thresholding the cheap model's confidence reached only **AUC 0.686** and needed
~88 % escalation to match the fixed model. The bottleneck is the *signal*, not
the idea.

This notebook asks what else is available, and what each addition buys.

## What may and may not be a gate input

**Never:** the beam label, the 64-dim power vector, any target-derived
difficulty measure, or whether a model happened to be correct. Those appear
only as offline supervision.

**The cost model decides the rest.** Cost here is forward GFLOPs — *processing*,
not sensing. Camera encoding is ~48 GFLOPs; an image sharpness statistic is
under 0.01. So computing a cheap statistic and then deciding whether to run
ResNet34 is a real saving. That argument **fails** under a sensing-cost model,
where the camera must be powered before any statistic exists.

So the feature groups are split along exactly that line:

| Group | Features | Needs an expensive sensor read? |
|---|---|---|
| **A** | cheap model top-1 confidence | no |
| **B** | A + full uncertainty (entropy, margins) + availability mask + GPS statistics | no |
| **C** | B + image, LiDAR and radar quality statistics | **yes** |

A and B are valid under either cost model. C is valid only under the
processing-cost model, and is reported separately so the claim can be read
either way.
"""))
    C.append(md("## 1. Configuration"))
    C.append(code(r"""
QUICK_RUN = True
CHEAP, EXPENSIVE = "gps", "gps_image"
EVAL_SPLIT = "val"
BATCH_SIZE = 32
WORKERS = 2
COLLECT_SENSOR_QUALITY = True       # group C needs these

LIMIT = 300 if QUICK_RUN else None
from pathlib import Path
OUT = Path("/kaggle/working/outputs/modality_quality")
(OUT / "plots").mkdir(parents=True, exist_ok=True)
print(f"QUICK_RUN={QUICK_RUN}  limit={LIMIT}")
"""))
    C.append(ENV)
    C.append(STAGE)
    C.append(CODE_CELL)
    C.append(eval_helper())
    C.append(md(r"""
## 2. Features and oracle labels

One pass of the cheap model collects the features; one pass of the expensive
model gives the comparison. The oracle labels are built **after** the features
and never enter them.
"""))
    C.append(code(r"""
cache = OUT / f"features_{EVAL_SPLIT}{'_quick' if QUICK_RUN else ''}.csv"
if cache.exists():
    F = pd.read_csv(cache)
    print(f"loaded cached features: {F.shape}")
else:
    _mc, cheap = run_eval(CHEAP, EVAL_SPLIT, limit=LIMIT, collect_features=True)
    _me, exp = run_eval(EXPENSIVE, EVAL_SPLIT, limit=LIMIT, return_per_sample=True)
    exp = exp.rename(columns={"correct1": "exp_correct1", "correct3": "exp_correct3",
                              "dba": "exp_dba", "pred_beam": "exp_pred_beam"})
    F = cheap.merge(exp[["sample_id", "exp_correct1", "exp_correct3", "exp_dba",
                         "exp_pred_beam"]], on="sample_id")
    # offline supervision, NOT features
    F["label_escalate"] = ((F.correct1 == 0) & (F.exp_correct1 == 1)).astype(int)
    F["gain_dba"] = F.exp_dba - F.dba
    F["label_cheap_ok"] = F.correct1
    F.to_csv(cache, index=False)
    print(f"built features: {F.shape} -> {cache.name}")

FEATURE_COLS = [c for c in F.columns if c not in (
    "sample_id", "scenario", "true_beam", "pred_beam", "correct1", "correct3", "dba",
    "exp_correct1", "exp_correct3", "exp_dba", "exp_pred_beam",
    "label_escalate", "gain_dba", "label_cheap_ok")]
# Guard: a constant feature contributes exactly nothing, and a *whole group*
# of constant features silently collapses group C onto group B. This happened
# once -- the dataset returns zeros for modalities outside the configuration's
# slug, so every sensor-quality statistic was identically zero. Fail loudly.
_const = [c for c in FEATURE_COLS if F[c].nunique() <= 1]
if _const:
    print(f"WARNING: {len(_const)} constant features carry no information:")
    print("   " + ", ".join(_const))
    _sensor_const = [c for c in _const if c.startswith(("img_", "lidar_", "radar_"))]
    if _sensor_const:
        raise RuntimeError(
            f"{len(_sensor_const)} sensor-quality features are constant, so feature "
            f"group C is identical to group B and the processing-vs-sensing "
            f"comparison is void. The dataset must be loaded with every modality "
            f"when collecting these features -- check COLLECT_SENSOR_QUALITY and "
            f"the ds_mods logic in run_eval.")

print(f"\n{len(FEATURE_COLS)} candidate features")
print(f"cheap correct    {F.correct1.mean():.4f}")
print(f"expensive correct {F.exp_correct1.mean():.4f}")
print(f"escalation pays on {F.label_escalate.mean()*100:.1f}% of samples")
print(f"mean DBA gain from escalating: {F.gain_dba.mean():+.4f}")
"""))
    C.append(md("## 3. Which signals predict that the cheap tier suffices?"))
    C.append(code(r"""
def auc(score, label):
    '''Rank-based AUC, no sklearn dependency.'''
    label = np.asarray(label).astype(bool)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    s, label = s[ok], label[ok]
    if label.all() or not label.any():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


target = F.label_cheap_ok.astype(bool)
single = pd.DataFrame([
    {"feature": c, "AUC": max(auc(F[c], target), 1 - auc(F[c], target)),
     "needs_sensor_read": c.startswith(("img_", "lidar_", "radar_"))}
    for c in FEATURE_COLS if F[c].nunique() > 1]).sort_values("AUC", ascending=False)
single.to_csv(OUT / "single_feature_auc.csv", index=False)
print("individual features, best first (AUC folded so 0.5 = no signal):\n")
print(single.head(15).round(4).to_string(index=False))
"""))
    C.append(md(r"""
## 4. Feature groups

A small logistic model per group, trained with a scenario-disjoint split so the
AUC is not inflated by memorising an environment. This is the comparison that
matters: does richer inference-time information close the gap to the oracle?
"""))
    C.append(code(r"""
def fit_logistic(X, y, epochs=400, lr=0.05, weight=None):
    '''Tiny logistic regression in torch -- no sklearn dependency.'''
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    Xn = (X - mu) / sd
    w = torch.zeros(Xn.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    pw = torch.tensor(weight) if weight else None
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Xn @ w + b, y, pos_weight=pw)
        loss.backward()
        opt.step()
    return lambda Z: ((torch.tensor(Z, dtype=torch.float32) - mu) / sd
                      ) @ w.detach() + b.detach()


GROUPS = {
    "A_confidence_only": [c for c in ["top1_conf"] if c in FEATURE_COLS],
    "B_uncertainty_plus_availability": [
        c for c in FEATURE_COLS if not c.startswith(("img_", "lidar_", "radar_"))],
    "C_plus_sensor_quality": FEATURE_COLS,
}
scen = sorted(F.scenario.unique())
print(f"scenario-disjoint folds: {scen}\n")
rows = []
for name, cols in GROUPS.items():
    if not cols:
        continue
    aucs = []
    for held in scen:
        tr, te = F[F.scenario != held], F[F.scenario == held]
        if te.label_cheap_ok.nunique() < 2 or len(tr) < 50:
            continue
        f = fit_logistic(tr[cols].to_numpy(), tr.label_cheap_ok.to_numpy())
        aucs.append(auc(f(te[cols].to_numpy()).numpy(), te.label_cheap_ok.astype(bool)))
    rows.append({"group": name, "n_features": len(cols),
                 "needs_sensor_read": name.startswith("C"),
                 "mean_AUC_heldout_scenario": float(np.nanmean(aucs)) if aucs else float("nan"),
                 "per_fold": " ".join(f"{a:.3f}" for a in aucs)})
groups_df = pd.DataFrame(rows)
groups_df.to_csv(OUT / "feature_group_auc.csv", index=False)
print(groups_df.round(4).to_string(index=False))
print("\nHeld-out-scenario AUC, so a group that only works by recognising the")
print("environment will not score well here. The oracle-routing baseline for")
print("confidence alone was 0.686 measured in-domain.")
"""))
    C.append(md("## 5. Plots and save"))
    C.append(code(r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
top = single.head(12).iloc[::-1]
ax[0].barh(top.feature, top.AUC,
           color=["tab:grey" if s else "tab:purple" for s in top.needs_sensor_read])
ax[0].axvline(0.5, color="k", lw=.8, ls="--")
ax[0].set(xlim=(0.45, max(0.75, top.AUC.max() + 0.03)),
          xlabel="AUC for 'cheap tier already correct'",
          title="Individual signals (grey = needs a sensor read)")
ax[0].grid(alpha=.3, axis="x")

g = groups_df.dropna(subset=["mean_AUC_heldout_scenario"])
ax[1].bar(g.group, g.mean_AUC_heldout_scenario,
          color=["tab:grey" if s else "tab:purple" for s in g.needs_sensor_read])
ax[1].axhline(0.5, color="k", lw=.8, ls="--")
ax[1].axhline(0.686, color="tab:red", lw=1, ls=":", label="oracle-run confidence (in-domain)")
ax[1].set(ylabel="held-out-scenario AUC", ylim=(0.4, 0.9),
          title="Feature groups")
ax[1].tick_params(axis="x", rotation=20)
ax[1].legend(frameon=False, fontsize=8)
ax[1].grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(OUT / "plots" / "gate_signals.png", dpi=150)
plt.show()

(OUT / "config.json").write_text(json.dumps(
    {"quick_run": QUICK_RUN, "cheap": CHEAP, "expensive": EXPENSIVE,
     "eval_split": EVAL_SPLIT, "limit": LIMIT,
     "n_features": len(FEATURE_COLS), "groups": {k: v for k, v in GROUPS.items()},
     "note": "group C requires an expensive sensor read; valid only under a "
             "processing-cost model"}, indent=2))
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}  ({p.stat().st_size/1e3:.1f} KB)")
"""))
    return nb(C)


# ------------------------------------------------------------------ notebook 3
def build_gate():
    C = []
    C.append(md(r"""
# Learned adaptive gate, and the final comparison

**RQ4** Can a learned gate exploit inference-time information better than
confidence thresholding?

Architecture, kept binary deliberately:

```
   GPS  (0.39 GFLOPs)
     │
     ├── features ──> learned gate ──> STOP
     │                      │
     │                      └────────> + Camera (48.24 GFLOPs) ──> final beam
```

**Why binary and not multi-class.** The ablation gives camera +0.154 DBA over
GPS against LiDAR's +0.047 and radar's +0.050, and GPS + Camera is the single
best configuration overall, so camera is the only escalation worth a route. One
caveat is recorded rather than acted on: on the *unseen* scenario 31, GPS+LiDAR
(0.2240 DBA) beat GPS+Camera (0.0200), which would argue for a LiDAR route — but
that rests on 50 samples, so `ROUTES` is left configurable and the third route
is off by default.

**The gate runs after the cheap model**, because its most informative feature is
that model's own uncertainty, and the cheap model costs 0.8 % of the expensive
one. Gating before any encoder would discard the single best signal to save
almost nothing.

## Training

Features come from the **train** split; the gate is evaluated on **val**. The
gate never sees a label, a power vector, or any correctness indicator — those
build the target only.

Two formulations are compared, because the right one depends on the class
balance the oracle run reported (escalation pays 22.9 %, hurts 8.9 %, neither
45 %):

1. **Classification** of "escalation pays", with positive weighting.
2. **Regression on the per-sample DBA gain.** Preferred a priori: it handles
   the 45 % where escalation is useless and the 8.9 % where it is harmful as
   small and negative targets rather than forcing them into one negative class,
   and thresholding a predicted gain is exactly the cost-aware decision
   `gain > λ·cost`.
"""))
    C.append(md("## 1. Configuration"))
    C.append(code(r"""
QUICK_RUN = True
CHEAP, EXPENSIVE = "gps", "gps_image"
ROUTES = [CHEAP, EXPENSIVE]          # add "gps_lidar" only if justified
TRAIN_SPLIT, EVAL_SPLIT = "train", "val"
BATCH_SIZE = 32
WORKERS = 2
COLLECT_SENSOR_QUALITY = True
GATE_EPOCHS = 300

LIMIT_TRAIN = 400 if QUICK_RUN else None
LIMIT_EVAL = 300 if QUICK_RUN else None
from pathlib import Path
OUT = Path("/kaggle/working/outputs/adaptive_gate")
FINAL = Path("/kaggle/working/outputs/adaptive_final")
for d in (OUT / "plots", FINAL / "plots"):
    d.mkdir(parents=True, exist_ok=True)
print(f"QUICK_RUN={QUICK_RUN}  train limit={LIMIT_TRAIN}  eval limit={LIMIT_EVAL}")
"""))
    C.append(ENV)
    C.append(STAGE)
    C.append(CODE_CELL)
    C.append(eval_helper())
    C.append(md("## 2. Build the gate's training and evaluation tables"))
    C.append(code(r"""
def build_table(split, limit):
    tag = f"{split}{'_quick' if QUICK_RUN else ''}"
    cache = OUT / f"table_{tag}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    _mc, cheap = run_eval(CHEAP, split, limit=limit, collect_features=True)
    _me, exp = run_eval(EXPENSIVE, split, limit=limit, return_per_sample=True)
    exp = exp.rename(columns={"correct1": "exp_correct1", "correct3": "exp_correct3",
                              "dba": "exp_dba"})
    T = cheap.merge(exp[["sample_id", "exp_correct1", "exp_correct3", "exp_dba"]],
                    on="sample_id")
    T["label_escalate"] = ((T.correct1 == 0) & (T.exp_correct1 == 1)).astype(int)
    T["gain_dba"] = T.exp_dba - T.dba
    T.to_csv(cache, index=False)
    return T


missing = [c for c in (CHEAP, EXPENSIVE) if c not in CKPTS]
if missing:
    raise SystemExit(
        f"This notebook needs the {missing} checkpoint(s) and they are not attached.\n"
        f"It runs standalone, but the two tiers are mandatory:\n"
        f"  {CHEAP!r}       the cheap tier (ablation part 1)\n"
        f"  {EXPENSIVE!r}   the escalation target (ablation part 2)\n"
        f"Found: {sorted(CKPTS) or 'none'}.\n"
        f"Any further checkpoints are optional and only enrich the final "
        f"comparison table.")
print(f"required tiers present: {CHEAP}, {EXPENSIVE}")
print(f"optional extras for the comparison table: "
      f"{sorted(set(CKPTS) - {CHEAP, EXPENSIVE}) or 'none'}\n")

TR = build_table(TRAIN_SPLIT, LIMIT_TRAIN)
EV = build_table(EVAL_SPLIT, LIMIT_EVAL)
FEATS = [c for c in TR.columns if c not in (
    "sample_id", "scenario", "true_beam", "pred_beam", "correct1", "correct3", "dba",
    "exp_correct1", "exp_correct3", "exp_dba", "label_escalate", "gain_dba")]
print(f"train {TR.shape}   eval {EV.shape}   {len(FEATS)} features\n")
print("class balance on train:")
print(f"  escalation pays : {TR.label_escalate.mean()*100:5.1f}%")
print(f"  cheap already ok: {TR.correct1.mean()*100:5.1f}%")
print(f"  neither correct : {((TR.correct1==0)&(TR.exp_correct1==0)).mean()*100:5.1f}%")
print(f"  escalation hurts: {((TR.correct1==1)&(TR.exp_correct1==0)).mean()*100:5.1f}%")
print(f"\nmean DBA gain {TR.gain_dba.mean():+.4f}   "
      f"negative for {(TR.gain_dba<0).mean()*100:.1f}% of samples")
"""))
    C.append(md("## 3. Train the gate"))
    C.append(code(r"""
class Gate(torch.nn.Module):
    '''Small MLP. Deliberately small: ~2k parameters against the 48 GFLOPs it
    decides about, so its own cost is negligible in the cost model.'''

    def __init__(self, d_in, hidden=32, out=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_gate(X, y, objective, epochs=GATE_EPOCHS, lr=1e-3):
    Xt = torch.tensor(X, dtype=torch.float32)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp(min=1e-6)
    Xn = (Xt - mu) / sd
    yt = torch.tensor(y, dtype=torch.float32)
    g = Gate(Xn.shape[1])
    opt = torch.optim.Adam(g.parameters(), lr=lr, weight_decay=1e-4)
    if objective == "classify":
        pw = torch.tensor([(len(yt) - yt.sum()) / yt.sum().clamp(min=1)])
        lossfn = lambda p, t: torch.nn.functional.binary_cross_entropy_with_logits(
            p, t, pos_weight=pw)
    else:
        lossfn = torch.nn.functional.mse_loss
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossfn(g(Xn), yt)
        loss.backward()
        opt.step()
    g.eval()
    n_par = sum(p.numel() for p in g.parameters())
    return (lambda Z: g((torch.tensor(Z, dtype=torch.float32) - mu) / sd).detach().numpy()), n_par


torch.manual_seed(0)
gates = {}
gates["classify"], npar = train_gate(TR[FEATS].to_numpy(),
                                     TR.label_escalate.to_numpy(), "classify")
gates["regress_gain"], _ = train_gate(TR[FEATS].to_numpy(),
                                      TR.gain_dba.to_numpy(), "regress")
print(f"two gates trained, {npar} parameters each "
      f"({npar*2/1e9:.2e} GFLOPs per decision -- negligible)")
"""))
    C.append(md(r"""
## 4. Operating curves

The full curve matters more than any single point. Each gate ranks the
evaluation samples by its escalation score; sweeping the fraction escalated
traces accuracy against average compute.

Cost model: GPS always runs, and escalation adds the camera encoder, so
`cost(f) = 0.39 + f · (48.24 − 0.39)`.
"""))
    C.append(code(r"""
c1, e1 = EV.correct1.to_numpy(), EV.exp_correct1.to_numpy()
cd, ed = EV.dba.to_numpy(), EV.exp_dba.to_numpy()
n = len(EV)
C_LO, C_HI = GFLOPS[CHEAP], GFLOPS[EXPENSIVE]


def sweep(score, name):
    order = np.argsort(-np.asarray(score))      # most-in-need first
    rows = []
    for f in np.linspace(0, 1, 101):
        k = int(round(f * n))
        esc = np.zeros(n, bool)
        esc[order[:k]] = True
        rows.append({"gate": name, "escalated_frac": f,
                     "Top-1": np.where(esc, e1, c1).mean(),
                     "DBA": np.where(esc, ed, cd).mean(),
                     "GFLOPs": C_LO + f * (C_HI - C_LO)})
    return pd.DataFrame(rows)


curves = [sweep(-EV.top1_conf.to_numpy(), "confidence threshold")]
for name, fn in gates.items():
    curves.append(sweep(fn(EV[FEATS].to_numpy()), f"learned gate ({name})"))
# Oracle ceilings, for reference only. There are *two*, because the optimal
# escalation order depends on the metric: ranking by "cheap wrong, expensive
# right" maximises Top-1, while ranking by the per-sample DBA gain maximises
# DBA. Using the Top-1 order as the DBA ceiling understates it by up to 0.04
# DBA here, and a gain-regressing gate can legitimately appear to beat it --
# so each oracle bounds only its own metric, and only that column is read.
curves.append(sweep(((EV.correct1 == 0) & (EV.exp_correct1 == 1)).astype(float).to_numpy()
                    + 1e-6 * EV.gain_dba.to_numpy(), "ORACLE (Top-1 optimal)"))
curves.append(sweep(EV.gain_dba.to_numpy(), "ORACLE (DBA optimal)"))
curves = pd.concat(curves, ignore_index=True)
curves.to_csv(OUT / "operating_curves.csv", index=False)

base_dba, base_t1 = ed.mean(), e1.mean()
print(f"always expensive: Top-1 {base_t1:.4f}  DBA {base_dba:.4f}  {C_HI:.2f} GFLOPs\n")
print(f"{'gate':<34}{'esc@DBA-match':>15}{'GFLOPs':>9}{'saved':>8}")
for name, g in curves.groupby("gate", sort=False):
    hit = g[g.DBA >= base_dba]
    if len(hit):
        r = hit.iloc[0]
        print(f"{name:<34}{r.escalated_frac*100:>14.1f}%{r.GFLOPs:>9.2f}"
              f"{(1-r.GFLOPs/C_HI)*100:>7.0f}%")
    else:
        print(f"{name:<34}{'never':>15}{g.DBA.max():>9.4f}{'--':>8}")
print(f"\n{'gate':<34}{'DBA@25%':>9}{'DBA@50%':>9}")
for name, g in curves.groupby("gate", sort=False):
    gr = g.reset_index(drop=True)
    q = lambda f: gr.loc[(gr.escalated_frac - f).abs().idxmin(), "DBA"]
    print(f"{name:<34}{q(.25):>9.4f}{q(.50):>9.4f}")
"""))
    C.append(md("## 5. Final comparison"))
    C.append(code(r"""
rows = []
for slug in sorted(CKPTS):
    m = run_eval(slug, EVAL_SPLIT, limit=LIMIT_EVAL)
    rows.append({"model": config_label(parse_modalities(slug.split("_"))),
                 "Top-1": m["top1"], "Top-3": m["top3"], "DBA": m["dba"],
                 "GFLOPs": GFLOPS[slug], "escalated_%": np.nan,
                 "avg_modalities": len(slug.split("_"))})
for name, g in curves.groupby("gate", sort=False):
    if "ORACLE" in name:
        continue
    hit = g[g.DBA >= base_dba]
    r = hit.iloc[0] if len(hit) else g.iloc[-1]
    rows.append({"model": f"{name} (DBA-matched)", "Top-1": r["Top-1"], "Top-3": np.nan,
                 "DBA": r.DBA, "GFLOPs": r.GFLOPs,
                 "escalated_%": r.escalated_frac * 100,
                 "avg_modalities": 1 + r.escalated_frac})
    gr = g.reset_index(drop=True)
    r25 = gr.loc[(gr.escalated_frac - .25).abs().idxmin()]
    rows.append({"model": f"{name} @25% escalated", "Top-1": r25["Top-1"], "Top-3": np.nan,
                 "DBA": r25.DBA, "GFLOPs": r25.GFLOPs, "escalated_%": 25.0,
                 "avg_modalities": 1.25})
final = pd.DataFrame(rows).sort_values("DBA", ascending=False)
final.to_csv(FINAL / "final_comparison.csv", index=False)
print(final.round(4).to_string(index=False))
"""))
    C.append(code(r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for a, col in zip(ax, ("DBA", "Top-1")):
    for name, g in curves.groupby("gate", sort=False):
        # each oracle bounds only the metric it was ordered by, so draw it on
        # that panel alone -- the Top-1 oracle is not a ceiling on DBA
        if "ORACLE" in name and col not in name:
            continue
        style = {"ls": "--", "color": "k", "lw": 1} if "ORACLE" in name else {"lw": 2}
        a.plot(g.GFLOPs, g[col], label=name, **style)
    fx = final[final["escalated_%"].isna()]      # the fixed configurations
    a.scatter(fx.GFLOPs, fx[col], s=70, c="tab:red", zorder=5, label="fixed configurations")
    for _, r in fx.iterrows():
        a.annotate(r.model.replace(" + ", "+"), (r.GFLOPs, r[col]), fontsize=7,
                   xytext=(4, 3), textcoords="offset points")
    a.set(xlabel="average forward GFLOPs per sample", ylabel=col,
          title=f"{col} vs inference cost")
    a.grid(alpha=.3)
    a.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(FINAL / "plots" / "accuracy_vs_cost.png", dpi=150)
plt.show()
"""))
    C.append(md(r"""
## 6. Does the gate generalise across environments?

The ablation showed camera is excellent on seen scenarios and poor on the
unseen one, so a gate that has merely learned "be confident → use camera" would
escalate exactly wrongly off-distribution. Training on some scenarios and
evaluating on a held-out one tests for that. Scenario ID is never a feature.
"""))
    C.append(code(r"""
rows = []
for held in sorted(EV.scenario.unique()):
    tr = TR[TR.scenario != held]
    te = EV[EV.scenario == held]
    if len(tr) < 100 or len(te) < 30:
        print(f"  {held}: too few samples ({len(tr)} train / {len(te)} eval), skipped")
        continue
    fn, _ = train_gate(tr[FEATS].to_numpy(), tr.gain_dba.to_numpy(), "regress")
    s = fn(te[FEATS].to_numpy())
    order = np.argsort(-s)
    for f in (0.25, 0.50):
        k = int(round(f * len(te)))
        esc = np.zeros(len(te), bool)
        esc[order[:k]] = True
        rows.append({"held_out": held, "escalated": f,
                     "DBA": float(np.where(esc, te.exp_dba, te.dba).mean()),
                     "DBA_always_cheap": float(te.dba.mean()),
                     "DBA_always_expensive": float(te.exp_dba.mean()),
                     "n": len(te)})
gen = pd.DataFrame(rows)
if len(gen):
    gen.to_csv(OUT / "cross_scenario_generalisation.csv", index=False)
    print(gen.round(4).to_string(index=False))
    print("\nIf the gate beats always-cheap on a scenario it never trained on, it has")
    print("learned something transferable rather than an environment shortcut.")
"""))
    C.append(md("## 7. Save"))
    C.append(code(r"""
meta = {"quick_run": QUICK_RUN, "cheap": CHEAP, "expensive": EXPENSIVE,
        "routes": ROUTES, "train_split": TRAIN_SPLIT, "eval_split": EVAL_SPLIT,
        "limits": {"train": LIMIT_TRAIN, "eval": LIMIT_EVAL},
        "n_features": len(FEATS), "features": FEATS,
        "gate_objectives": list(gates), "gate_parameters": npar,
        "cost_model": f"cost(f) = {C_LO} + f * ({C_HI} - {C_LO})",
        "leakage_note": "gate inputs contain no beam, power vector, or correctness "
                        "signal; those build the training target only"}
(OUT / "config.json").write_text(json.dumps(meta, indent=2))
(FINAL / "config.json").write_text(json.dumps(meta, indent=2))
for d in (OUT, FINAL):
    for p in sorted(d.rglob("*")):
        if p.is_file():
            print(f"  {p}  ({p.stat().st_size/1e3:.1f} KB)")
"""))
    return nb(C)


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    for name, builder in (("modality_robustness", build_robustness),
                          ("modality_quality_signal", build_quality_signal),
                          ("learned_adaptive_gate", build_gate)):
        out = here / f"{name}.ipynb"
        d = builder()
        out.write_text(json.dumps(d, indent=1))
        print(f"wrote {out.name} ({len(d['cells'])} cells)")
