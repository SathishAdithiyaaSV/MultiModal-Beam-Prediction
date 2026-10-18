"""Generate the two scenario-31 Kaggle notebooks.

Scenario 31 is the only out-of-domain data this project has, and every
generalisation claim currently rests on the 50 labelled samples the
`adaptation` split provides. The standalone release holds 7,012 frames, public
on Kaggle as `flippedliu/deepsense6g`, which `adapt_scenario31.py` converts into
6,804 challenge-format windows.

Two notebooks, because the work splits cleanly by resource:

  scenario31_preprocess.ipynb   CPU, hours. Raw release -> AMBER tensors.
                                Saves to /kaggle/working for release as a
                                dataset the second notebook attaches.
  scenario31_evaluate.ipynb     GPU, under an hour. Scores all available
                                checkpoints on scenario 31.

Both verify their inputs before doing any real work, because the expensive
failure mode here is discovering after four hours of radar FFTs that the
release did not ship the modality you needed.

NOTE for whoever edits this: use ''' inside cell bodies, never \"\"\". Nested
triple double-quotes break the generator, which has bitten this project three
times.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/SathishAdithiyaaSV/MultiModal-Beam-Prediction.git"
RAW_DS = "flippedliu/deepsense6g"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.strip("\n").splitlines(True)}


def nb(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}


# ---------------------------------------------------------------- notebook 1
def preprocess_cells():
    C = [md(r"""
# Scenario 31 — preprocessing the standalone release

Every generalisation claim in this project rests on **50** labelled scenario-31
samples. This notebook converts the standalone release into **6,804** AMBER
samples, which is the single highest-value data task available to us.

Scenario 31 is held-out evaluation data. It is **never trained on** — doing so
would destroy the unseen-scenario claim it exists to test.

**Runtime: hours, CPU-bound.** Radar 2D-FFTs and LiDAR BEV histograms over ~7k
frames dominate. Internet must be on (the code is cloned).
""")]

    C.append(md("## 1. Verify the release before doing hours of work"))
    C.append(code(r'''
import os, sys, subprocess
from pathlib import Path

RAW = Path("/kaggle/input/deepsense6g/scenario31")
assert RAW.is_dir(), f"{RAW} not found -- attach the deepsense6g dataset"

# The expensive failure is discovering after four hours of radar FFTs that a
# modality was never shipped. Check first, cheaply.
expect = {
    "camera": RAW / "unit1/camera_data",
    "lidar":  RAW / "unit1/lidar_data",
    "radar":  RAW / "unit1/radar_data",
    "power":  RAW / "unit1/mmWave_data",
    "gps_bs": RAW / "unit1/GPS_data",
    "gps_ue": RAW / "unit2/GPS_data",
}
print(f"{'modality':<10}{'present':>9}{'files':>9}")
missing = []
for k, p in expect.items():
    n = len(list(p.iterdir())) if p.is_dir() else 0
    print(f"{k:<10}{str(p.is_dir()):>9}{n:>9}")
    if not p.is_dir():
        missing.append(k)
csv = RAW / "scenario31_dev.csv"
print(f"\nlabels csv: {csv.exists()}  ({csv.stat().st_size/1e6:.1f} MB)" if csv.exists()
      else "\nlabels csv MISSING")
if missing:
    print(f"\nMISSING: {missing}")
    print("The pipeline tolerates a missing modality -- those samples get an"
          " availability-mask zero -- but configurations using it cannot be"
          " scored on scenario 31. Continuing.")
'''))

    C.append(md("## 2. Code"))
    C.append(code(rf'''
PROJECT = Path("/kaggle/working/project")
if not (PROJECT / "amber" / "model.py").exists():
    r = subprocess.run(["git", "clone", "--depth", "1", "{REPO}", str(PROJECT)],
                       capture_output=True, text=True,
                       env={{**os.environ, "GIT_TERMINAL_PROMPT": "0"}})
    assert r.returncode == 0, (r.stderr or "")[:400]
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print("code at", PROJECT)
'''))

    C.append(md(r"""
## 3. Stage into the layout the pipeline expects

`config.SOURCES["scenario31_full"]` points at `data/raw/scenario31_full`, and
`build_index` recovers the scenario by matching `scenario\d+` against the image
path — so the tree must contain a `scenario31/` component. We symlink rather
than copy: the release is ~9 GB and Kaggle input is read-only but fast.
"""))
    C.append(code(r'''
DATA = PROJECT / "data" / "raw" / "scenario31_full"
DATA.mkdir(parents=True, exist_ok=True)
link = DATA / "scenario31"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(RAW)
print("staged", link, "->", RAW)

# convert the per-frame csv into challenge-format windows
out_csv = DATA / "scenario31_challenge_format.csv"
r = subprocess.run([sys.executable, "-u", "preprocessing_amber/adapt_scenario31.py",
                    "--csv", str(RAW / "scenario31_dev.csv"), "--out", str(out_csv)],
                   capture_output=True, text=True)
print(r.stdout[-1500:])
assert r.returncode == 0, r.stderr[-2000:]
'''))

    C.append(md(r"""
## 4. Preprocess

Radar and LiDAR dominate the runtime. `--source scenario31_full` keeps this off
the existing scenarios, whose tensors we are not rebuilding.
"""))
    C.append(code(r'''
import time
def run(label, *args):
    t0 = time.time()
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([sys.executable, "-u", *args], text=True)
    print(f"--- {label}: {(time.time()-t0)/60:.1f} min, rc={r.returncode}", flush=True)
    return r.returncode

run("radar RA/RV", "preprocessing_amber/radar_ra_rv.py", "--source", "scenario31_full")
run("lidar BEV",   "preprocessing_amber/lidar_bev.py",   "--source", "scenario31_full")
run("image cache", "preprocessing_amber/image_cache.py", "--source", "scenario31_full")
'''))

    C.append(md(r"""
## 5. Index

**The GPS normalisation constants must be the ones the models were trained
under.** There is no train split here to fit on, so refitting — or leaving them
empty, which is what the unpatched code does — would put GPS on a different
scale and silently invalidate every number measured against those weights. We
reuse the originals and report how much of scenario 31 falls outside the
training range, since a geographically distant scenario will clip.
"""))
    C.append(code(r'''
INDEX_SRC = None
for c in Path("/kaggle/input").rglob("gps_norm.json"):
    INDEX_SRC = c
    break
assert INDEX_SRC, ("no gps_norm.json in the inputs -- attach the preprocessed "
                   "dataset so the training-time GPS constants can be reused")
print("reusing GPS constants from", INDEX_SRC)

run("index", "preprocessing_amber/build_index.py", "--gps-norm-from", str(INDEX_SRC))
'''))

    C.append(md("## 6. Verify and save"))
    C.append(code(r'''
import pandas as pd, shutil, json
IDX = PROJECT / "data" / "processed_amber" / "index"
s = pd.read_csv(IDX / "samples.csv")
print("rows:", len(s))
print(s.groupby(["split", "scenario"]).size().to_string())
print("\navailability:")
for m in ("image", "lidar", "radar", "gps"):
    c = f"m_{m}"
    if c in s.columns:
        print(f"  {m:<8}{s[c].mean():.3f}")
assert (s["split"] == "scenario31_full").all(), \
    "scenario 31 must be held out whole -- nothing here may be train/val"
assert len(s) > 6000, f"expected ~6804 samples, got {len(s)}"

OUT = Path("/kaggle/working/scenario31_preprocessed")
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(PROJECT / "data" / "processed_amber", OUT,
                symlinks=False, ignore=shutil.ignore_patterns("*.tmp"))
tot = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"\nsaved {OUT}  ({tot/1e9:.2f} GB)")
print("Release this as a dataset, then attach it to scenario31_evaluate.ipynb")
'''))
    return C


# ---------------------------------------------------------------- notebook 2
def evaluate_cells():
    C = [md(r"""
# Scenario 31 — out-of-domain evaluation

Scores every available checkpoint on the **6,804** held-out scenario-31
samples, replacing the n=50 figures the generalisation finding currently rests
on.

Needs the preprocessed scenario-31 dataset produced by
`scenario31_preprocess.ipynb`, plus the checkpoint datasets. GPU, under an hour.
""")]
    C.append(md("## 1. Setup"))
    C.append(code(rf'''
import os, sys, json, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd, torch

PROJECT = Path("/kaggle/working/project")
if not (PROJECT / "amber" / "model.py").exists():
    r = subprocess.run(["git", "clone", "--depth", "1", "{REPO}", str(PROJECT)],
                       capture_output=True, text=True,
                       env={{**os.environ, "GIT_TERMINAL_PROMPT": "0"}})
    assert r.returncode == 0, (r.stderr or "")[:400]
os.chdir(PROJECT); sys.path.insert(0, str(PROJECT))

from amber.ablation import config_label, parse_modalities
from amber.config import TrainConfig
from amber.dataset import AmberDataset
from amber.metrics import evaluate
from amber.predict import load_checkpoint
from amber.train import make_loader, pick_device, to_device
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
'''))

    C.append(md("## 2. Locate the index and the checkpoints"))
    C.append(code(r'''
IN = Path("/kaggle/input")
cands = [d for d in (p.parent for p in IN.rglob("samples.csv"))
         if (d / "gps_norm.json").exists()]
INDEX_DIR = None
for d in cands:
    s = pd.read_csv(d / "samples.csv", usecols=["split"], nrows=5000)
    if (s["split"] == "scenario31_full").any():
        INDEX_DIR = d
        break
assert INDEX_DIR, f"no scenario-31 index among {cands}"
ROOT = INDEX_DIR.parent.parent
print("index:", INDEX_DIR)

SLUGS = ["gps", "gps_radar", "gps_lidar", "gps_radar_lidar", "gps_image",
         "gps_image_lidar", "gps_radar_image", "gps_radar_image_lidar"]
CKPTS = {}
for d, _dn, fn in os.walk(IN, followlinks=True):
    for f in fn:
        if f == "best.pt" and Path(d).name in SLUGS:
            CKPTS.setdefault(Path(d).name, Path(d) / f)
        elif f.endswith(".pt") and f[:-3] in SLUGS:
            CKPTS.setdefault(f[:-3], Path(d) / f)
print(f"{len(CKPTS)}/8 checkpoints:", ", ".join(sorted(CKPTS)))
assert CKPTS, "no checkpoints attached"
'''))

    C.append(md("## 3. Evaluate"))
    C.append(code(r'''
device = pick_device()
train_cfg = TrainConfig()

@torch.no_grad()
def score(slug):
    model, cfg = load_checkpoint(CKPTS[slug], device)
    mods = parse_modalities(slug.split("_"))
    ds = AmberDataset(INDEX_DIR, "scenario31_full", cfg, root=ROOT, modalities=mods)
    loader = make_loader(ds, 32, shuffle=False, workers=2)
    lg, tg = [], []
    for b in loader:
        b = to_device(b, device)
        lg.append(model(b, use_cma=False).logits.float().cpu())
        tg.append(b["target"].cpu())
    lg, tg = torch.cat(lg), torch.cat(tg)
    lab = tg >= 0
    m = evaluate(lg[lab], tg[lab], ks=train_cfg.topk, delta=train_cfg.dba_delta)
    del model; torch.cuda.empty_cache()
    return m, int(lab.sum())

rows = []
for slug in sorted(CKPTS):
    t0 = time.time()
    m, n = score(slug)
    rows.append({"config": config_label(parse_modalities(slug.split("_"))),
                 "slug": slug, "Top-1": m["top1"], "Top-3": m["top3"],
                 "Top-5": m.get("top5", np.nan), "DBA": m["dba"], "n": n,
                 "has_camera": "image" in slug})
    print(f"{slug:<26} DBA {m['dba']:.4f}  Top-1 {m['top1']:.4f}  "
          f"n={n}  ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows).sort_values("DBA", ascending=False)
OUT = Path("/kaggle/working/outputs/scenario31")
(OUT).mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "scenario31_results.csv", index=False)
print("\n" + res.round(4).to_string(index=False))
'''))

    C.append(md(r"""
## 4. The generalisation finding, at 136× the evidence

The n=50 result was: camera configurations averaged 0.8764 in domain and
**0.0567** here, camera-free ones 0.7701 and 0.1940, with GPS alone the best of
all eight. If that ordering holds at n=6,804 it stops being an anecdote.
"""))
    C.append(code(r'''
cam = res[res.has_camera].DBA.mean()
nocam = res[~res.has_camera].DBA.mean()
print(f"camera configurations      mean DBA {cam:.4f}  (n={res.has_camera.sum()})")
print(f"camera-free configurations mean DBA {nocam:.4f}  (n={(~res.has_camera).sum()})")
print(f"\nbest overall: {res.iloc[0].config}  DBA {res.iloc[0].DBA:.4f}")
print(f"GPS alone rank: "
      f"{int((res.slug == 'gps').argmax()) + 1} of {len(res)}")

prior = {"camera_mean_n50": 0.0567, "camera_free_mean_n50": 0.1940}
print(f"\nversus the n=50 figures: camera {prior['camera_mean_n50']:.4f} -> {cam:.4f}, "
      f"camera-free {prior['camera_free_mean_n50']:.4f} -> {nocam:.4f}")
json.dump({"camera_mean": float(cam), "camera_free_mean": float(nocam),
           "n_samples": int(res.n.iloc[0]), "prior_n50": prior,
           "best_config": res.iloc[0].config},
          open(OUT / "generalisation_summary.json", "w"), indent=2)
for p in sorted(OUT.rglob("*")):
    print(" ", p.relative_to(OUT), f"({p.stat().st_size/1e3:.1f} KB)")
'''))
    return C


def main():
    for name, cells in [("scenario31_preprocess", preprocess_cells()),
                        ("scenario31_evaluate", evaluate_cells())]:
        p = HERE / f"{name}.ipynb"
        p.write_text(json.dumps(nb(cells), indent=1))
        print(f"wrote {p.name} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
