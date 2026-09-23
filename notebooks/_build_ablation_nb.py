"""Generate the two modality-ablation notebooks.

Both notebooks are identical apart from the configurations they train, so they
are emitted from one template -- that is what guarantees the two halves of the
experiment use the same protocol, which is the whole point of a controlled
ablation.

The split is balanced by encoder cost so the two run in about the same wall
clock, taking ResNet18 as 1 unit and ResNet34 as 1.8:

  part 1: GPS (0) + GPS/Radar (1) + GPS/Radar/Camera (2.8) + full (3.8) = 7.6
  part 2: GPS/Camera (1.8) + GPS/LiDAR (1) + GPS/Radar/LiDAR (2) +
          GPS/Camera/LiDAR (2.8)                                       = 7.6

Part 1 also carries the key comparisons (GPS -> +Radar -> +Camera -> full), so
it is the half to run first if only one can be run.

Note: code-cell bodies must not contain triple double-quotes, since the cells
are r\"\"\"...\"\"\" literals here -- use ''' for any docstring inside a cell.
"""
import json
import pathlib

PARTS = {
    1: [["gps"], ["gps", "radar"], ["gps", "radar", "camera"],
        ["gps", "radar", "camera", "lidar"]],
    2: [["gps", "camera"], ["gps", "lidar"], ["gps", "radar", "lidar"],
        ["gps", "camera", "lidar"]],
}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").split("\n")}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.strip("\n").split("\n")}


def build(part: int):
    experiments = PARTS[part]
    other = 2 if part == 1 else 1
    C = []

    C.append(md(rf"""
# Modality Ablation — part {part} of 2

**Which modalities are actually worth paying for, relative to the cheap
Radar + GPS configuration?**

This is the prerequisite experiment for the adaptive-routing direction: before
building a model that decides *when* to acquire an expensive modality, we need
to know what each modality is worth once the cheap ones are already in hand.

The two parts are independent and can run **in parallel on two Kaggle
sessions**. They share one protocol and write to the same output layout, so the
results merge into a single table.

| | configurations | encoder cost |
|---|---|---|
| **part 1** | GPS · GPS+Radar · GPS+Radar+Camera · full | 7.6 |
| **part 2** | GPS+Camera · GPS+LiDAR · GPS+Radar+LiDAR · GPS+Camera+LiDAR | 7.6 |

**This notebook runs part {part}:** {', '.join(' + '.join(e) for e in experiments)}

Part 1 carries the key comparisons (GPS → +Radar → +Camera → full), so run it
first if you can only run one.

---

## Method

Every configuration trains **independently from scratch** under an identical
protocol — same split, samples, labels, window, seed, optimiser, schedule,
epochs, batch size and metrics. The only thing that varies is which modalities
the model may use.

This reuses AMBER's own missing-modality mechanism rather than building a
different architecture per configuration: a configuration forces the eq. (3)
availability vector, and the eq. (23)/(30) masks do the rest. Verified
property — a modality marked unavailable has **exactly zero** influence on the
logits, so the restriction is faithful rather than approximate.

The mask can only ever *remove* a modality, never add one. A sample that
genuinely lacks radar stays without radar. So "GPS + Radar" on scenario 34 is
partly "GPS only" in practice, and the notebook reports the effective
availability alongside each result so the numbers can be read honestly.

## Two methodological decisions, stated rather than buried

**Historical beam indices are disabled.** AMBER treats the previous 4 beams as
a fifth input modality (its eq. 13), but they are *not* part of the DeepSense6G
2022 challenge input specification, and they are absent for 100 % of the
official test split and 100 % of the adaptation split. Leaving them on would
let a signal that vanishes at inference dominate the ablation — AMBER's own
Table V has BeamIdx+GPS alone at 58.81 % Top-1. `USE_HISTORICAL_BEAMS` exposes
this; it is `False` for the challenge-compliant comparison.

**Modality dropout is off.** The main AMBER run trains with random modality
masking, but with only one or two modalities enabled that behaves differently
per configuration — for a single-modality config the "keep at least one"
fallback makes it a no-op. Leaving it on would vary the protocol across
configurations, which is exactly what a controlled ablation must not do.
"""))

    C.append(md("## 1. Configuration"))
    C.append(code(rf"""
from pathlib import Path

PART = {part}

# Modality configurations trained by THIS notebook. Edit freely -- a
# configuration whose result already exists is skipped, so re-running is cheap.
EXPERIMENTS = {json.dumps(experiments)}

# Historical beam indices are an AMBER addition, not a challenge input, and are
# absent from the entire test and adaptation splits. See the header.
USE_HISTORICAL_BEAMS = False

EPOCHS      = 10      # see the note below
BATCH_SIZE  = 16      # AMBER Table II
LR          = 1e-4    # AMBER Table II
POOL        = 4       # VA = HA
SEED        = 2022
WORKERS     = 4
MODALITY_DROPOUT = 0.0   # off for a controlled ablation -- see the header

# Where results land. Stable filenames so a later notebook can load them.
OUT = Path("/kaggle/working/outputs/modality_ablation")
RUNS = OUT / "runs"
for d in (OUT, OUT / "plots", OUT / "per_config", RUNS):
    d.mkdir(parents=True, exist_ok=True)

print(f"part {{PART}}: {{len(EXPERIMENTS)}} configurations")
for e in EXPERIMENTS:
    print("   ", " + ".join(e))
print(f"\nhistorical beams: {{USE_HISTORICAL_BEAMS}}   epochs: {{EPOCHS}}   "
      f"modality dropout: {{MODALITY_DROPOUT}}")
"""))

    C.append(md(r"""
### Why 10 epochs

The reference 20-epoch AMBER run peaked at epoch 13 and gained nothing over
epochs 14–20, so 10 epochs with a cosine schedule *sized for 10* anneals fully
and lands close to the plateau. That matters here because the budget is eight
configurations rather than one.

Sizing: the full configuration costs ~12 min/epoch at 8,844 training samples.
This notebook's configurations total ~7.6 encoder-cost units against the full
configuration's 3.8, so ≈ 24 min per epoch across all four, i.e. **≈ 4 h for 10
epochs** — inside Kaggle's ~9 h GPU session with margin.

Raising `EPOCHS` is scientifically preferable if you have the budget; just keep
it **identical across both parts**, or the comparison breaks.
"""))

    C.append(md("## 2. Environment"))
    C.append(code(r"""
import socket
import subprocess
import sys

import torch

print(f"torch {torch.__version__}   cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  {p.total_memory / 1e9:.1f} GB")
else:
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU, then Run All again.")


def has_internet(host="github.com", port=443, timeout=6):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


if not has_internet():
    raise SystemExit("Internet is off; the code cannot be cloned and pretrained "
                     "ResNet weights cannot be downloaded. Settings -> Internet -> On.")
print("internet: on")
"""))

    C.append(md("## 3. Locate and stage the canonical preprocessed dataset"))
    C.append(code(r"""
import os
import shutil

import pandas as pd

INPUT_ROOT = Path("/kaggle/input")
SOURCES = ("development", "adaptation", "test")


def walk_dirs(base, follow=True):
    for dirpath, dirnames, filenames in os.walk(base, followlinks=follow):
        yield Path(dirpath), dirnames, filenames


def find_index(base):
    for cand in (base / "index" / "index", base / "index", base):
        if (cand / "samples.csv").exists():
            return cand
    for d, _dirs, files in walk_dirs(base):
        if "samples.csv" in files:
            return d
    return None


def find_source(base, name):
    for cand in (base / name / name, base / name):
        if cand.is_dir() and any(cand.glob("scenario*")):
            return cand
    return None


INDEX_SRC = find_index(INPUT_ROOT)
if INDEX_SRC is None:
    raise SystemExit("samples.csv not found under /kaggle/input. Attach the "
                     "preprocessed dataset via + Add Input.")
CANDIDATE_ROOTS = [INDEX_SRC.parent, INDEX_SRC.parent.parent, INDEX_SRC]
SOURCE_DIRS = {s: next((d for d in (find_source(r, s) for r in CANDIDATE_ROOTS) if d), None)
               for s in SOURCES}
for s in [k for k, v in SOURCE_DIRS.items() if v is None]:
    for d, dirnames, _f in walk_dirs(INPUT_ROOT):
        if d.name == s and any(n.startswith("scenario") for n in dirnames):
            SOURCE_DIRS[s] = d
            break

STAGE = Path("/kaggle/working/data/processed_amber")
STAGE.mkdir(parents=True, exist_ok=True)
INDEX_DIR = STAGE / "index"
if not (INDEX_DIR / "samples.csv").exists():
    shutil.copytree(INDEX_SRC, INDEX_DIR, dirs_exist_ok=True)
for name, src in SOURCE_DIRS.items():
    if src is not None and not (STAGE / name).exists():
        (STAGE / name).symlink_to(src)

print(f"index: {INDEX_SRC}")
for s, d in SOURCE_DIRS.items():
    print(f"  {s:<12} {d}")
assert SOURCE_DIRS["development"], "development source not found -- training needs it"
"""))

    C.append(md("## 4. Repository code"))
    C.append(code(r"""
REPO = "https://github.com/SathishAdithiyaaSV/MultiModal-Beam-Prediction.git"
PROJECT = Path("/kaggle/working/project")

if not (PROJECT / "amber" / "model.py").exists():
    r = subprocess.run(["git", "clone", "--depth", "1", REPO, str(PROJECT)],
                       capture_output=True, text=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        raise SystemExit(f"clone failed: {(r.stderr or '').strip()[:300]}")
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from amber.ablation import (active_parameters, config_label, effective_availability,
                            enabled_mask, measure_gflops, measure_latency,
                            parse_modalities)
from amber.config import MODALITIES, ModelConfig, TrainConfig
from amber.dataset import AmberDataset
from amber.predict import load_checkpoint
from amber.train import build_parser, make_loader, to_device, train

print(f"cwd {Path.cwd()}")
print(f"modalities in the model: {MODALITIES}")
"""))

    C.append(md(r"""
## 5. The dataset this experiment consumes

Reported so the results can be read against what the model actually received.
Note the availability columns: scenario 34 has radar for only ~40 % of its
samples, and beam history is absent from the adaptation and test splits
entirely.
"""))
    C.append(code(r"""
samples = pd.read_csv(INDEX_DIR / "samples.csv")
usable = samples[samples.split.isin(["train", "val", "adaptation", "test"])]

print("split sizes:")
print(samples.split.value_counts().to_string(), "\n")
print("availability by split (the eq. 3 mask):")
print(usable.groupby("split")[[f"m_{m}" for m in MODALITIES]].mean().round(3).to_string())
print("\nsamples per split and scenario:")
print(usable.groupby(["split", "scenario"]).size().to_string())
"""))

    C.append(md(r"""
**Which split is reported.** `val` (2,198 labelled, scenarios 32/33/34) is the
primary result. `adaptation` (100 labelled, scenarios 31/32/33) is a secondary
check and the only source of scenario-31 numbers. `test` is **unlabelled**, so
it cannot be scored — it is excluded from this experiment entirely.

Scenario 31 therefore rests on 50 adaptation samples. That is thin, and the
notebook flags it rather than presenting those numbers as equally solid.
"""))

    C.append(md("## 6. Modality configurations"))
    C.append(code(r"""
def resolve(cfg_modalities):
    '''Canonical modality tuple for a configuration, honouring the beam switch.'''
    mods = list(parse_modalities(cfg_modalities))
    if USE_HISTORICAL_BEAMS and "beam" not in mods:
        mods.append("beam")
    if not USE_HISTORICAL_BEAMS and "beam" in mods:
        mods.remove("beam")
    return tuple(mods)


def slug(mods):
    return "_".join(mods)


CONFIGS = []
for spec in EXPERIMENTS:
    mods = resolve(spec)
    CONFIGS.append({"modalities": mods, "label": config_label(mods), "slug": slug(mods)})

print(f"{'label':<36}{'slug':<28}modalities")
for c in CONFIGS:
    print(f"{c['label']:<36}{c['slug']:<28}{c['modalities']}")
"""))

    C.append(md(r"""
## 7. Train + evaluate one configuration

Each configuration calls the repository's own `amber.train.train` with
`--modalities`, so the architecture, losses, schedule and metrics are exactly
the ones used everywhere else in the project. Nothing about the model is
reimplemented here.

`skip_unavailable_encoders` is switched on so a configuration does not push
zeros through the encoders of modalities it cannot use. Verified numerically
equivalent (Δlogits = 0.0); it exists so the measured latency, FLOPs and active
parameter count reflect what the configuration would really cost to deploy.
"""))
    C.append(code(r"""
import gc
import json
import time


def cost_of(ckpt_path, mods):
    '''Active parameters, latency and FLOPs for a trained configuration.'''
    device = torch.device("cuda")
    model, mcfg = load_checkpoint(ckpt_path, device)
    ds = AmberDataset(INDEX_DIR, "val", mcfg, modalities=mods)
    loader = make_loader(ds, BATCH_SIZE, shuffle=False, workers=0)
    batch = to_device(next(iter(loader)), device)
    batch["availability"] = batch["availability"] * enabled_mask(mods, device)

    out = active_parameters(model, mods)
    out["latency_ms_per_sample"] = measure_latency(model, batch, device)
    out["gflops_per_sample"] = measure_gflops(model, batch)
    del model, loader, ds, batch
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_configuration(cfg):
    '''Train one configuration from scratch and record metrics + cost.'''
    result_path = OUT / "per_config" / f"{cfg['slug']}.json"
    if result_path.exists():
        print(f"[skip] {cfg['label']} -- result already at {result_path.name}")
        return json.loads(result_path.read_text())

    print(f"\n{'=' * 78}\n{cfg['label']}   ({', '.join(cfg['modalities'])})\n{'=' * 78}",
          flush=True)
    args = build_parser().parse_args([
        "--index-dir", str(INDEX_DIR),
        "--run-dir", str(RUNS),
        "--name", cfg["slug"],
        "--modalities", *cfg["modalities"],
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--pool", str(POOL),
        "--seed", str(SEED),
        "--workers", str(WORKERS),
        "--modality-dropout", str(MODALITY_DROPOUT),
        "--device", "cuda",
        "--skip-unavailable-encoders",
        "--resume",
        "--splits", "val", "adaptation",
    ])
    started = time.time()
    train(args)
    train_seconds = time.time() - started

    run_dir = RUNS / cfg["slug"]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    history = pd.read_csv(run_dir / "history.csv")

    record = {
        "part": PART,
        "label": cfg["label"],
        "slug": cfg["slug"],
        "modalities": list(cfg["modalities"]),
        "use_historical_beams": USE_HISTORICAL_BEAMS,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "pool": POOL,
        "seed": SEED, "modality_dropout": MODALITY_DROPOUT,
        "metrics": metrics,
        "train_seconds": train_seconds,
        "best_epoch": int(history.loc[history.val_top1.idxmax(), "epoch"]),
        "effective_availability_train": effective_availability(
            samples[samples.split == "train"], cfg["modalities"]),
        "cost": cost_of(run_dir / "best.pt", cfg["modalities"]),
    }
    result_path.write_text(json.dumps(record, indent=2))
    print(f"\nsaved -> {result_path}")

    gc.collect()
    torch.cuda.empty_cache()
    return record
"""))

    C.append(md(rf"""
## 8. Run this part's configurations

Results are written after **every** configuration, so an interrupted session
loses at most the one in flight, and re-running skips whatever finished.
"""))
    C.append(code(r"""
records = []
for cfg in CONFIGS:
    records.append(run_configuration(cfg))
print(f"\n{len(records)} configurations complete for part {PART}")
"""))

    C.append(md(rf"""
## 9. This part's results

A local summary so the outcome is visible immediately. The **complete
8-configuration table, the deltas against GPS + Radar, the scenario comparison,
the plots and the interpretation all live in
`modality_ablation_report.ipynb`**, which merges both parts and needs no GPU.

That separation is deliberate: the deltas are measured against GPS + Radar,
which part 1 trains, so no single training notebook can complete the
cross-part analysis on its own.
"""))
    C.append(code(r"""
def flatten(r, split="val"):
    m = r["metrics"].get(split, {}).get("overall", {})
    c = r["cost"]
    return {
        "Configuration": r["label"],
        "Top-1": m.get("top1"), "Top-3": m.get("top3"),
        "Top-5": m.get("top5"), "DBA": m.get("dba"), "n": m.get("n"),
        "Params (M)": c["params_active"] / 1e6,
        "Latency (ms)": c["latency_ms_per_sample"],
        "GFLOPs": c["gflops_per_sample"],
        "Train (min)": r["train_seconds"] / 60.0,
        "Best epoch": r["best_epoch"],
    }


part_results = pd.DataFrame([flatten(r) for r in records])
part_results.to_csv(OUT / f"results_part{PART}.csv", index=False)
print(f"part {PART} -- validation split\n")
print(part_results.round(4).to_string(index=False))
"""))

    C.append(md("## 10. Save outputs"))
    C.append(code(rf"""
config_record = {{
    "part": PART,
    "experiments": EXPERIMENTS,
    "use_historical_beams": USE_HISTORICAL_BEAMS,
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "pool": POOL,
    "seed": SEED, "modality_dropout": MODALITY_DROPOUT,
    "eval_splits": ["val", "adaptation"],
    "note": ("test is unlabelled and cannot be scored; historical beams disabled "
             "as they are not a challenge input and are absent from the whole "
             "test and adaptation splits"),
}}
(OUT / f"config_part{{PART}}.json").write_text(json.dumps(config_record, indent=2))

print(f"{{OUT}}:")
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.suffix in (".json", ".csv"):
        print(f"  {{p.relative_to(OUT)}}  ({{p.stat().st_size / 1e3:.1f}} KB)")

print()
print("NEXT")
print("----")
print(f"1. Save Version, then download outputs/modality_ablation/per_config/*.json "
      f"({{len(records)}} files from this part).")
print("2. Do the same for part {other}.")
print("3. Upload all 8 json files as one Kaggle dataset, attach it to")
print("   modality_ablation_report.ipynb and Run All. That produces the complete")
print("   table, deltas, scenario comparison, plots and interpretation.")
print("   No GPU needed -- it also runs locally.")
"""))

    return {
        "cells": C,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    for part in (1, 2):
        out = here / f"modality_ablation_part{part}.ipynb"
        out.write_text(json.dumps(build(part), indent=1))
        print(f"wrote {out} ({len(build(part)['cells'])} cells)")
